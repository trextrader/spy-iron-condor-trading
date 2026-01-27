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
        model_config = checkpoint.get("model_config", {})
    else:
        state_dict = checkpoint
        input_dim = None
        model_config = {}

    if input_dim is None:
        # Fallback to hardcoded assumption if old checkpoint
        print("[WARN] Checkpoint missing input_dim, assuming 52 (V2.2)")
        input_dim = 52

    # Robustly infer architecture from config or state_dict
    d_model = model_config.get("d_model", 512)
    n_layers = model_config.get("n_layers", 16) # Default to 16 for this run
    use_vol = model_config.get("use_vol_gated_attn", True)
    
    # Infer diffusion from state_dict if not in config
    has_diffusion_weights = any("diffusion_head" in k for k in state_dict.keys())
    use_diff = model_config.get("use_diffusion", has_diffusion_weights)
    
    # Infer MoE from state_dict
    has_moe_weights = any("moe_head" in k for k in state_dict.keys())
    use_topk = model_config.get("use_topk_moe", has_moe_weights)
    moe_experts = model_config.get("moe_n_experts", 3)
    moe_k = model_config.get("moe_k", 1)

    print(f"[Model] Loading CondorBrain: {d_model}d x {n_layers}L | Diffusion={use_diff} | MoE={use_topk}")

    model = CondorBrain(
        d_model=d_model,
        n_layers=n_layers,
        input_dim=input_dim,
        use_vol_gated_attn=use_vol,
        use_topk_moe=use_topk,
        moe_n_experts=moe_experts,
        moe_k=moe_k,
        use_diffusion=use_diff,
        diffusion_steps=50, # Default, not critical for inference unless using sampling loop
        diffusion_input_dim=10, 
        diffusion_horizon=1
    )
    
    # Strip 'module.' or '_orig_mod.' prefixes if present (torch.compile artifact)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if name.startswith("module."):
            name = name[7:]
        if name.startswith("_orig_mod."):
            name = name[10:]
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=False)
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


def model_forward(model: nn.Module, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Run forward pass and return dict with keys:
      - "entry_logit"  : [B] or [B,1]
      - "exit_logit"   : [B] or [B,1]
      - "size_score"   : [B] (continuous) or [B,K] (discrete sizing)

    New 10-output format (2026-01-26):
      [0] call_off, [1] put_off, [2] width, [3] dte,
      [4] prob_profit, [5] expected_roi, [6] max_loss_pct, [7] confidence,
      [8] entry_logit (NEW), [9] exit_logit (NEW)
    """
    outputs = model(x)
    pol = outputs[0]

    # Policy head indices (from CondorExpertHead)
    prob_profit = pol[:, 4]
    expected_roi = pol[:, 5]
    confidence = pol[:, 7]

    # NEW: Use explicit entry/exit logits if available (10-output model)
    if pol.size(1) >= 10:
        entry_logit = pol[:, 8]  # Explicit entry logit
        exit_logit = pol[:, 9]   # Explicit exit logit
    else:
        # FALLBACK: Legacy 8-output model - derive from confidence (will collapse!)
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
def compute_temporal_features(Xs: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Compute temporal aggregations from sequence data for surrogate tree fitting.

    Instead of just using the last timestep, compute multiple temporal summaries:
    - last: value at final timestep (original behavior)
    - mean: mean over entire sequence
    - std: standard deviation over sequence
    - min: minimum value
    - max: maximum value
    - slope: linear regression slope (trend)
    - last5_mean: mean of last 5 timesteps (recent momentum)

    Args:
        Xs: (N, T, D) sequence data
        feature_names: List of D feature names

    Returns:
        X_temporal: (N, D*7) flattened temporal features
        temporal_names: List of D*7 feature names
    """
    N, T, D = Xs.shape

    # Compute temporal aggregations
    last = Xs[:, -1, :]  # (N, D)
    mean = Xs.mean(axis=1)  # (N, D)
    std = Xs.std(axis=1)  # (N, D)
    min_val = Xs.min(axis=1)  # (N, D)
    max_val = Xs.max(axis=1)  # (N, D)

    # Last 5 timesteps mean (or all if T < 5)
    last5 = Xs[:, max(0, T-5):, :].mean(axis=1)  # (N, D)

    # Slope (linear regression coefficient)
    # Use simple (y[-1] - y[0]) / (T-1) as proxy for speed
    slope = (Xs[:, -1, :] - Xs[:, 0, :]) / max(T - 1, 1)  # (N, D)

    # Concatenate all temporal features
    X_temporal = np.concatenate([
        last, mean, std, min_val, max_val, last5, slope
    ], axis=1)  # (N, D*7)

    # Generate feature names
    suffixes = ['_last', '_mean', '_std', '_min', '_max', '_last5', '_slope']
    temporal_names = []
    for suffix in suffixes:
        for name in feature_names:
            temporal_names.append(f"{name}{suffix}")

    return X_temporal.astype(np.float32), temporal_names


def fit_tree_surrogate_classification(X: np.ndarray, y: np.ndarray, max_depth: int = 4):
    # Handle edge case: if all y are same class, can't stratify
    if len(np.unique(y)) < 2:
        # Return dummy tree with 100% accuracy on constant class
        clf = DecisionTreeClassifier(max_depth=1, random_state=42)
        clf.fit(X[:10], y[:10])  # Fit on small subset
        return clf, 1.0

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
    batch_size: int = 64,
    ig_batch_size: int = 8,
):
    ensure_out_dir()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path).to(device)
    model.eval()

    X, meta = load_dataset(dataset_path, model_path, sample_n=sample_n)
    
    # -----------------------------------------------------------
    # CRITICAL FIX: Robust Normalization (Match Training Logic)
    # -----------------------------------------------------------
    if not meta.get("normalized", False):
        print("[Audit] Robust-normalizing features (Median/MAD) to match training distribution...")
        
        # 1. IVR Robustness
        feature_names = list(meta["feature_names"])
        if 'ivr' in feature_names:
            ivr_idx = feature_names.index('ivr')
            # Check last timestep of each sequence for scale
            # X shape is usually (N, L, F) or (N, F). 
            # load_dataset returns sequences (N, L, F).
            # We must normalize the feature dimension (last dim).
            
            # Flatten for stats calculation
            N, L, F = X.shape
            X_flat = X.reshape(-1, F)
            
            ivr_vals = X_flat[:, ivr_idx]
            finite = np.isfinite(ivr_vals)
            if finite.any() and np.nanmax(ivr_vals) <= 1.5:
                 print("[Audit] Detected IVR 0-1 scale. Boosting to 0-100.")
                 X_flat[:, ivr_idx] = np.where(finite, ivr_vals * 100.0, 50.0)
            X_flat[:, ivr_idx] = np.clip(X_flat[:, ivr_idx], 0.0, 100.0)
            
            # 2. Robust Z-Score
            # Helper functions (inline)
            def robust_zscore_fit(arr, eps=1e-6):
                med = np.nanmedian(arr, axis=0)
                mad = np.nanmedian(np.abs(arr - med), axis=0)
                return med, np.maximum(mad, eps)

            def robust_zscore_transform(arr, med, scale, clip_val=10.0):
                return np.clip((arr - med) / scale, -clip_val, clip_val)
            
            med, scale = robust_zscore_fit(X_flat)
            X_flat_norm = robust_zscore_transform(X_flat, med, scale, clip_val=10.0)
            
            # Reshape back to sequences
            X = X_flat_norm.reshape(N, L, F)
            print("[Audit] Features normalized.")
    # -----------------------------------------------------------
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

    # DEBUG: Check stats of input Xs
    print(f"[DEBUG] Xs stats: shape={Xs.shape}, min={Xs.min():.4f}, max={Xs.max():.4f}, mean={Xs.mean():.4f}")
    if np.abs(Xs.mean()) > 10.0 or np.abs(Xs.max()) > 50.0:
        print("[WARNING] Input features seem UNNORMALIZED! Model expects Z-scores.")

    # Batched forward to avoid GPU OOM
    entry_logits = []
    exit_logits = []
    size_scores = []
    total = Xs.shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        if start % (batch_size * 10) == 0:
            print(f"[Forward] batch {start//batch_size + 1} / {int(np.ceil(total / batch_size))}")
        xb = torch.tensor(Xs[start:end], dtype=torch.float32, device=device)
        with torch.no_grad():
            out = model_forward(model, xb)
        entry_logits.append(out["entry_logit"].detach().cpu())
        exit_logits.append(out["exit_logit"].detach().cpu())
        size_scores.append(out["size_score"].detach().cpu())
        del xb, out
        if device.type == "cuda":
            torch.cuda.empty_cache()

    entry_logit = torch.cat(entry_logits, dim=0).view(-1)
    exit_logit = torch.cat(exit_logits, dim=0).view(-1)
    size_raw = torch.cat(size_scores, dim=0)

    entry_prob = torch.sigmoid(entry_logit).numpy()
    exit_prob = torch.sigmoid(exit_logit).numpy()

    print(f"[DEBUG] entry_prob: min={entry_prob.min():.4f}, mean={entry_prob.mean():.4f}, max={entry_prob.max():.4f}")
    print(f"[DEBUG] entry_prob >= 0.5 rate: {(entry_prob >= 0.5).mean():.4f}")

    entry_y = (entry_prob >= 0.5).astype(np.int32)
    exit_y = (exit_prob >= 0.5).astype(np.int32)

    if size_raw.ndim == 2:
        size_y = np.argmax(size_raw.numpy(), axis=1).astype(np.int32)
        size_is_classification = True
    else:
        size_y = size_raw.numpy().reshape(-1).astype(np.float32)
        size_is_classification = False

    # 2026-01-26: Use temporal feature aggregations instead of just last timestep
    # This helps surrogate trees capture temporal patterns that sequence models learn
    X_temporal, temporal_feature_names = compute_temporal_features(Xs, feature_names)
    print(f"[Surrogate] Using {len(temporal_feature_names)} temporal features ({len(feature_names)} base * 7 aggregations)")

    entry_tree, entry_acc = fit_tree_surrogate_classification(X_temporal, entry_y, max_depth=surrogate_depth)
    exit_tree, exit_acc = fit_tree_surrogate_classification(X_temporal, exit_y, max_depth=surrogate_depth)

    entry_rules = tree_to_markdown_rules(entry_tree, temporal_feature_names, class_names=["NO_ENTRY", "ENTRY"])
    exit_rules = tree_to_markdown_rules(exit_tree, temporal_feature_names, class_names=["NO_EXIT", "EXIT"])

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
        size_tree, size_score = fit_tree_surrogate_classification(X_temporal, size_y, max_depth=surrogate_depth)
        size_rules = tree_to_markdown_rules(size_tree, temporal_feature_names)
        save_text(
            os.path.join(OUT_DIR, "sizing_rules.md"),
            f"# SIZING surrogate rules (classification, depth={surrogate_depth})\n\n"
            f"Surrogate accuracy~{size_score:.3f}\n\n{size_rules}\n",
        )
    else:
        size_tree, size_score = fit_tree_surrogate_regression(X_temporal, size_y, max_depth=surrogate_depth)
        size_rules = tree_to_markdown_rules(size_tree, temporal_feature_names)
        save_text(
            os.path.join(OUT_DIR, "sizing_rules.md"),
            f"# SIZING surrogate rules (regression, depth={surrogate_depth})\n\n"
            f"Surrogate R^2~{size_score:.3f}\n\n{size_rules}\n",
        )

    # Integrated gradients in small batches; stream aggregation to avoid OOM
    sum_abs_entry = np.zeros(len(feature_names), dtype=np.float64)
    sum_abs_exit = np.zeros(len(feature_names), dtype=np.float64)
    sum_abs_size = np.zeros(len(feature_names), dtype=np.float64)
    count = 0

    for start in range(0, total, ig_batch_size):
        end = min(start + ig_batch_size, total)
        if start % (ig_batch_size * 10) == 0:
            print(f"[IG] batch {start//ig_batch_size + 1} / {int(np.ceil(total / ig_batch_size))}")
        xb = torch.tensor(Xs[start:end], dtype=torch.float32, device=device)
        baseline_b = torch.tensor(baseline[start:end], dtype=torch.float32, device=device)
        ig_entry = integrated_gradients(model, "entry_logit", xb, baseline_b, steps=ig_steps).cpu().numpy()
        ig_exit = integrated_gradients(model, "exit_logit", xb, baseline_b, steps=ig_steps).cpu().numpy()
        ig_size = integrated_gradients(model, "size_score", xb, baseline_b, steps=ig_steps).cpu().numpy()
        sum_abs_entry += np.sum(np.abs(ig_entry), axis=(0, 1))
        sum_abs_exit += np.sum(np.abs(ig_exit), axis=(0, 1))
        sum_abs_size += np.sum(np.abs(ig_size), axis=(0, 1))
        count += ig_entry.shape[0] * ig_entry.shape[1]
        del xb, baseline_b, ig_entry, ig_exit, ig_size
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def _summarize(sum_abs):
        imp = (sum_abs / max(count, 1)).astype(np.float64)
        df = pd.DataFrame({"feature": feature_names, "importance": imp})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    df_entry = _summarize(sum_abs_entry)
    df_exit = _summarize(sum_abs_exit)
    df_size = _summarize(sum_abs_size)

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
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--ig-batch-size", type=int, default=8)
    args = ap.parse_args()

    main(
        model_path=args.model,
        dataset_path=args.data,
        baseline_mode=args.baseline,
        ig_steps=args.ig_steps,
        surrogate_depth=args.surrogate_depth,
        sample_n=args.sample_n,
        batch_size=args.batch_size,
        ig_batch_size=args.ig_batch_size,
    )
