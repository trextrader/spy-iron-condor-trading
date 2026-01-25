from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    # Repo path setup for direct execution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
except Exception:
    pass

from intelligence.condor_brain import CondorBrain
from intelligence.canonical_feature_registry import (
    FEATURE_COLS_V22,
    VERSION_V22,
    apply_semantic_nan_fill,
    get_neutral_fill_value_v22,
)
from intelligence.features.dynamic_features import (
    compute_all_dynamic_features,
    compute_all_primitive_features_v22,
)
from intelligence.rule_engine.dsl_parser import RuleDSLParser
from intelligence.rule_engine.executor import RuleExecutionEngine

OUT_DIR = "artifacts/learned_conditions"
RULESET_PATH = "docs/Complete_Ruleset_DSL.yaml"


# -------------------------
# 1) INTEGRATION POINTS
# -------------------------
def load_model(model_path: str) -> nn.Module:
    """
    TODO: load your trained DeepMamba2/CondorBrain model.
    Must return a torch.nn.Module in eval() mode.
    """
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint)
        input_dim = checkpoint.get("input_dim")
    else:
        state_dict = checkpoint
        input_dim = None

    if input_dim is None:
        raise ValueError("Checkpoint missing input_dim; cannot build CondorBrain.")

    model = CondorBrain(
        d_model=512,
        n_layers=12,
        input_dim=input_dim,
        use_vol_gated_attn=True,
        use_topk_moe=True,
        moe_n_experts=3,
        moe_k=1,
        use_diffusion=True,
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _load_df_and_features(dataset_path: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    if "dt" not in df.columns and "timestamp" in df.columns:
        df.rename(columns={"timestamp": "dt"}, inplace=True)
    if "dt" in df.columns:
        df["dt"] = pd.to_datetime(df["dt"], utc=True)

    missing_v22 = [c for c in FEATURE_COLS_V22 if c not in df.columns]
    if not missing_v22:
        return df

    dt_col = None
    for c in ["dt", "timestamp", "datetime", "date"]:
        if c in df.columns:
            dt_col = c
            break

    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    aux_cols = [c for c in ["spread_ratio", "lag_minutes"] if c in df.columns]

    if dt_col is None:
        df = compute_all_dynamic_features(df, close_col="close", high_col="high", low_col="low")
        df = compute_all_primitive_features_v22(
            df,
            close_col="close",
            high_col="high",
            low_col="low",
            volume_col="volume",
            spread_col="spread_ratio" if "spread_ratio" in df.columns else "close",
            inplace=True,
        )
        return df

    spot_key_cols = ["symbol", dt_col] if "symbol" in df.columns else [dt_col]
    spot_df = df.drop_duplicates(subset=spot_key_cols)[spot_key_cols + ohlcv_cols + aux_cols].copy()
    spot_df = spot_df.sort_values(spot_key_cols).reset_index(drop=True)

    spot_df = compute_all_dynamic_features(spot_df, close_col="close", high_col="high", low_col="low")
    spot_df = compute_all_primitive_features_v22(
        spot_df,
        close_col="close",
        high_col="high",
        low_col="low",
        volume_col="volume",
        spread_col="spread_ratio" if "spread_ratio" in spot_df.columns else "close",
        inplace=True,
    )

    exclude_cols = spot_key_cols + ohlcv_cols + aux_cols
    computed_cols = [c for c in spot_df.columns if c not in exclude_cols]
    merge_df = spot_df[spot_key_cols + computed_cols]
    df = df.merge(merge_df, on=spot_key_cols, how="left", suffixes=("", "_calc"))
    for col in computed_cols:
        calc_col = f"{col}_calc"
        if calc_col in df.columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[calc_col])
            else:
                df[col] = df[calc_col]
            df.drop(columns=[calc_col], inplace=True)
    return df


def _apply_rule_engine(df: pd.DataFrame, ruleset_path: str) -> pd.DataFrame:
    if not os.path.exists(ruleset_path):
        return df
    parser = RuleDSLParser(ruleset_path)
    ruleset = parser.load()
    engine = RuleExecutionEngine(ruleset)

    dt_col = None
    for c in ["dt", "timestamp", "datetime", "date"]:
        if c in df.columns:
            dt_col = c
            break
    spot_key_cols = ["symbol", dt_col] if dt_col and "symbol" in df.columns else ([dt_col] if dt_col else None)
    if spot_key_cols and df.duplicated(subset=spot_key_cols).any():
        use_spot_df = df.drop_duplicates(subset=spot_key_cols).copy()
        use_spot_df = use_spot_df.sort_values(spot_key_cols).reset_index(drop=True)
    else:
        use_spot_df = df

    results = engine.execute(use_spot_df)
    long_signals = []
    short_signals = []
    exit_signals = []
    block_signals = []
    for rule_id, _rule in ruleset.rules.items():
        r_res = results.get(rule_id)
        if r_res is None:
            continue
        long_s = r_res.get("signal_long", pd.Series(False, index=use_spot_df.index)).astype(int)
        short_s = r_res.get("signal_short", pd.Series(False, index=use_spot_df.index)).astype(int)
        exit_s = r_res.get("signal_exit", pd.Series(False, index=use_spot_df.index)).astype(int)
        block_s = r_res.get("blocked", pd.Series(False, index=use_spot_df.index)).astype(int)
        long_signals.append(long_s.values)
        short_signals.append(short_s.values)
        exit_signals.append(exit_s.values)
        block_signals.append(block_s.values)

    if long_signals:
        spot_consensus = pd.DataFrame(
            {
                "rule_long_consensus": np.mean(long_signals, axis=0),
                "rule_short_consensus": np.mean(short_signals, axis=0),
                "rule_exit_consensus": np.mean(exit_signals, axis=0),
                "rule_block_any": np.max(block_signals, axis=0),
            },
            index=use_spot_df.index,
        )
    else:
        spot_consensus = pd.DataFrame(
            {
                "rule_long_consensus": 0.0,
                "rule_short_consensus": 0.0,
                "rule_exit_consensus": 0.0,
                "rule_block_any": 0.0,
            },
            index=use_spot_df.index,
        )

    if spot_key_cols and len(use_spot_df) != len(df):
        spot_join = use_spot_df[spot_key_cols].copy()
        spot_join = spot_join.join(spot_consensus)
        df = df.merge(spot_join, on=spot_key_cols, how="left")
    else:
        for col in spot_consensus.columns:
            df[col] = spot_consensus[col].values
    return df


def load_dataset(
    dataset_path: str,
    model_path: str,
    sample_n: int = 4096,
    seq_len: int = 256,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    TODO: return (X, meta), where:
      X: np.ndarray of shape [N, T, D]
      meta: contains feature_names (len D), and optional timestamps/trade_ids.
    """
    checkpoint = torch.load(model_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "feature_cols" not in checkpoint:
        raise ValueError("Checkpoint missing feature_cols; cannot build dataset.")
    feature_cols = list(checkpoint["feature_cols"])

    df = _load_df_and_features(dataset_path)
    df = _apply_rule_engine(df, RULESET_PATH)

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        for col in missing_cols:
            df[col] = get_neutral_fill_value_v22(col)

    X_np = df[feature_cols].values.astype(np.float32)
    X_np = np.where(np.isfinite(X_np), X_np, np.nan)
    X_np = apply_semantic_nan_fill(X_np, feature_cols)

    if "median" in checkpoint and "mad" in checkpoint:
        mu = np.asarray(checkpoint["median"], dtype=np.float32).squeeze()
        mad = np.asarray(checkpoint["mad"], dtype=np.float32).squeeze()
        if mu.ndim != 1:
            mu = mu.reshape(-1)
        if mad.ndim != 1:
            mad = mad.reshape(-1)
        if mu.shape[0] == X_np.shape[1] and mad.shape[0] == X_np.shape[1]:
            mad = np.maximum(mad, 1e-6)
            X_np = (X_np - mu) / (1.4826 * mad)

    X_np = np.clip(X_np, -10.0, 10.0)

    max_start = len(X_np) - seq_len
    if max_start <= 0:
        raise ValueError("Dataset too small for sequence length.")
    rng = np.random.RandomState(42)
    idx = rng.choice(max_start, size=min(sample_n, max_start), replace=False)
    Xs = np.stack([X_np[i : i + seq_len] for i in idx], axis=0)
    meta = {
        "feature_names": feature_cols,
        "sequence_len": seq_len,
        "sampled": True,
        "version": VERSION_V22,
    }
    return Xs, meta


@torch.no_grad()
def model_forward(model: nn.Module, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    TODO: Run forward pass and return dict with keys:
      - "entry_logit"  : [B] or [B,1]
      - "exit_logit"   : [B] or [B,1]
      - "size_score"   : [B] (continuous) or [B,K] (discrete sizing)
    """
    outputs = model(x)
    pol = outputs[0]
    # Policy head: [call_off, put_off, width, te, prob_profit, expected_roi, max_loss_pct, confidence]
    prob_profit = pol[:, 4]
    expected_roi = pol[:, 5]
    confidence = pol[:, 7]

    eps = 1e-6
    conf_clamped = torch.clamp(confidence, eps, 1 - eps)
    entry_logit = torch.log(conf_clamped / (1 - conf_clamped))
    exit_logit = torch.log((1 - conf_clamped) / conf_clamped)
    size_score = expected_roi if expected_roi is not None else prob_profit

    return {
        "entry_logit": entry_logit,
        "exit_logit": exit_logit,
        "size_score": size_score,
    }


def export_fuzzy_system() -> Optional[Dict[str, Any]]:
    """
    If you have a fuzzy sizing engine:
      export membership functions + rule base + defuzzification config.
    Return dict or None if not applicable.
    """
    return None


# -------------------------
# 2) ATTRIBUTION METHODS
# -------------------------
def integrated_gradients(
    model: nn.Module,
    forward_key: str,
    x: torch.Tensor,
    baseline: torch.Tensor,
    steps: int = 32,
) -> torch.Tensor:
    """
    Integrated gradients on inputs x for a scalar output head forward_key.
    Returns attribution with same shape as x: [B, T, D].
    """
    assert x.shape == baseline.shape
    x = x.requires_grad_(True)

    total_grads = torch.zeros_like(x)
    for i in range(1, steps + 1):
        alpha = float(i) / float(steps)
        xi = baseline + alpha * (x - baseline)
        xi.requires_grad_(True)

        out = model_forward(model, xi)[forward_key]
        out = out.view(-1).sum()

        grads = torch.autograd.grad(out, xi, retain_graph=False, create_graph=False)[0]
        total_grads += grads

    avg_grads = total_grads / float(steps)
    ig = (x - baseline) * avg_grads
    return ig.detach()


def summarize_attribution(
    attributions: np.ndarray,
    feature_names: List[str],
    agg: str = "mean_abs",
) -> pd.DataFrame:
    """
    attributions: [N, T, D] or [N, D]
    Produces feature-level importances.
    """
    if attributions.ndim == 3:
        if agg == "mean_abs":
            imp = np.mean(np.abs(attributions), axis=(0, 1))
        elif agg == "mean":
            imp = np.mean(attributions, axis=(0, 1))
        else:
            raise ValueError("Unsupported agg")
    else:
        if agg == "mean_abs":
            imp = np.mean(np.abs(attributions), axis=0)
        elif agg == "mean":
            imp = np.mean(attributions, axis=0)
        else:
            raise ValueError("Unsupported agg")

    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


# -------------------------
# 3) SURROGATE "LEARNED CONDITIONS"
# -------------------------
def fit_tree_surrogate_classification(X: np.ndarray, y: np.ndarray, max_depth: int = 4):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))
    return clf, acc


def fit_tree_surrogate_regression(X: np.ndarray, y: np.ndarray, max_depth: int = 4):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    reg.fit(Xtr, ytr)
    score = r2_score(yte, reg.predict(Xte))
    return reg, score


def tree_to_markdown_rules(tree, feature_names: List[str], class_names: Optional[List[str]] = None) -> str:
    """
    Emit a readable rule list from a decision tree.
    """
    from sklearn.tree import _tree

    t = tree.tree_
    feat = t.feature
    thr = t.threshold

    def rec(node: int, indent: int) -> List[str]:
        pad = "  " * indent
        if feat[node] != _tree.TREE_UNDEFINED:
            name = feature_names[feat[node]]
            lines = []
            lines.append(f"{pad}- IF `{name} <= {thr[node]:.6g}`:")
            lines += rec(t.children_left[node], indent + 1)
            lines.append(f"{pad}- ELSE  (`{name} > {thr[node]:.6g}`):")
            lines += rec(t.children_right[node], indent + 1)
            return lines
        if hasattr(tree, "classes_"):
            values = t.value[node][0]
            cls = int(np.argmax(values))
            prob = float(values[cls] / (np.sum(values) + 1e-12))
            label = class_names[cls] if class_names else str(tree.classes_[cls])
            return [f"{pad}- THEN => **{label}** (leaf p~{prob:.3f})"]
        pred = float(t.value[node][0][0])
        return [f"{pad}- THEN => **pred~{pred:.6g}**"]

    return "\n".join(rec(0, 0))


# -------------------------
# 4) MAIN EXPORT
# -------------------------
def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_text(path: str, txt: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)

def _check_checkpoint_feature_cols(model_path: str, feature_names: List[str]) -> None:
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "feature_cols" not in checkpoint:
        raise ValueError("Checkpoint missing 'feature_cols'; cannot hard-check inputs.")
    ckpt_cols = list(checkpoint["feature_cols"])
    data_cols = list(feature_names)
    if ckpt_cols == data_cols:
        return
    ckpt_set = set(ckpt_cols)
    data_set = set(data_cols)
    missing_in_data = sorted(ckpt_set - data_set)
    extra_in_data = sorted(data_set - ckpt_set)
    if missing_in_data or extra_in_data:
        msg = (
            "Feature columns mismatch between checkpoint and dataset.\n"
            f"Missing in data: {missing_in_data}\n"
            f"Extra in data: {extra_in_data}\n"
        )
        raise ValueError(msg)
    # Same set, different order
    raise ValueError(
        "Feature column order mismatch between checkpoint and dataset. "
        "Reorder dataset feature_names to match checkpoint feature_cols."
    )


def main(
    model_path: str,
    dataset_path: str,
    baseline_mode: str = "zeros",
    ig_steps: int = 32,
    surrogate_depth: int = 4,
    sample_n: int = 4096,
):
    ensure_out_dir()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path).to(device)
    model.eval()

    X, meta = load_dataset(dataset_path, model_path, sample_n=sample_n)
    feature_names = list(meta["feature_names"])
    _check_checkpoint_feature_cols(model_path, feature_names)

    if meta.get("sampled"):
        Xs = X
    else:
        N = X.shape[0]
        idx = np.random.RandomState(42).choice(N, size=min(sample_n, N), replace=False)
        Xs = X[idx]

    if baseline_mode == "zeros":
        baseline = np.zeros_like(Xs)
    elif baseline_mode == "mean":
        baseline = np.broadcast_to(np.mean(Xs, axis=0, keepdims=True), Xs.shape)
    else:
        raise ValueError("baseline_mode must be zeros|mean")

    inputs_schema = {
        "sequence_shape": list(X.shape[1:]),
        "feature_names": feature_names,
        "notes": "Baked-in inputs consumed by the model (FeatureVector).",
    }
    save_json(os.path.join(OUT_DIR, "inputs_schema.json"), inputs_schema)

    xt = torch.tensor(Xs, dtype=torch.float32, device=device)
    out = model_forward(model, xt)

    entry_prob = torch.sigmoid(out["entry_logit"].view(-1)).cpu().numpy()
    exit_prob = torch.sigmoid(out["exit_logit"].view(-1)).cpu().numpy()

    entry_y = (entry_prob >= 0.5).astype(np.int32)
    exit_y = (exit_prob >= 0.5).astype(np.int32)

    size_raw = out["size_score"].detach().cpu().numpy()
    if size_raw.ndim == 2:
        size_y = np.argmax(size_raw, axis=1).astype(np.int32)
        size_is_classification = True
    else:
        size_y = size_raw.reshape(-1).astype(np.float32)
        size_is_classification = False

    Xflat = Xs[:, -1, :]

    entry_tree, entry_acc = fit_tree_surrogate_classification(Xflat, entry_y, max_depth=surrogate_depth)
    exit_tree, exit_acc = fit_tree_surrogate_classification(Xflat, exit_y, max_depth=surrogate_depth)

    entry_rules = tree_to_markdown_rules(entry_tree, feature_names, class_names=["NO_ENTRY", "ENTRY"])
    exit_rules = tree_to_markdown_rules(exit_tree, feature_names, class_names=["NO_EXIT", "EXIT"])

    save_text(
        os.path.join(OUT_DIR, "entry_rules.md"),
        f"# ENTRY surrogate rules (depth={surrogate_depth})\n\n"
        f"Surrogate accuracy~{entry_acc:.3f}\n\n{entry_rules}\n",
    )
    save_text(
        os.path.join(OUT_DIR, "exit_rules.md"),
        f"# EXIT surrogate rules (depth={surrogate_depth})\n\n"
        f"Surrogate accuracy~{exit_acc:.3f}\n\n{exit_rules}\n",
    )

    if size_is_classification:
        size_tree, size_score = fit_tree_surrogate_classification(Xflat, size_y, max_depth=surrogate_depth)
        size_rules = tree_to_markdown_rules(size_tree, feature_names)
        save_text(
            os.path.join(OUT_DIR, "sizing_rules.md"),
            f"# SIZING surrogate rules (classification, depth={surrogate_depth})\n\n"
            f"Surrogate accuracy~{size_score:.3f}\n\n{size_rules}\n",
        )
    else:
        size_tree, size_score = fit_tree_surrogate_regression(Xflat, size_y, max_depth=surrogate_depth)
        size_rules = tree_to_markdown_rules(size_tree, feature_names)
        save_text(
            os.path.join(OUT_DIR, "sizing_rules.md"),
            f"# SIZING surrogate rules (regression, depth={surrogate_depth})\n\n"
            f"Surrogate R^2~{size_score:.3f}\n\n{size_rules}\n",
        )

    baseline_t = torch.tensor(baseline, dtype=torch.float32, device=device)
    ig_entry = integrated_gradients(model, "entry_logit", xt, baseline_t, steps=ig_steps).cpu().numpy()
    ig_exit = integrated_gradients(model, "exit_logit", xt, baseline_t, steps=ig_steps).cpu().numpy()
    ig_size = integrated_gradients(model, "size_score", xt, baseline_t, steps=ig_steps).cpu().numpy()

    df_entry = summarize_attribution(ig_entry, feature_names, agg="mean_abs")
    df_exit = summarize_attribution(ig_exit, feature_names, agg="mean_abs")
    df_size = summarize_attribution(ig_size, feature_names, agg="mean_abs")

    df_entry.to_csv(os.path.join(OUT_DIR, "attribution_entry.csv"), index=False)
    df_exit.to_csv(os.path.join(OUT_DIR, "attribution_exit.csv"), index=False)
    df_size.to_csv(os.path.join(OUT_DIR, "attribution_sizing.csv"), index=False)

    fuzzy = export_fuzzy_system()
    if fuzzy is not None:
        save_json(os.path.join(OUT_DIR, "fuzzy_system.json"), fuzzy)

    print(f"[OK] Exported learned condition artifacts to: {OUT_DIR}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--baseline", default="zeros", choices=["zeros", "mean"])
    ap.add_argument("--ig-steps", type=int, default=32)
    ap.add_argument("--surrogate-depth", type=int, default=4)
    ap.add_argument("--sample-n", type=int, default=4096)
    args = ap.parse_args()

    main(
        model_path=args.model,
        dataset_path=args.data,
        baseline_mode=args.baseline,
        ig_steps=args.ig_steps,
        surrogate_depth=args.surrogate_depth,
        sample_n=args.sample_n,
    )
