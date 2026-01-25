from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

OUT_DIR = "artifacts/learned_conditions"


# -------------------------
# 1) INTEGRATION POINTS
# -------------------------
def load_model(model_path: str) -> nn.Module:
    """
    TODO: load your trained DeepMamba2/CondorBrain model.
    Must return a torch.nn.Module in eval() mode.
    """
    raise NotImplementedError("Implement load_model(model_path) for your repo.")


def load_dataset(dataset_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    TODO: return (X, meta), where:
      X: np.ndarray of shape [N, T, D]
      meta: contains feature_names (len D), and optional timestamps/trade_ids.
    """
    raise NotImplementedError("Implement load_dataset(dataset_path) to produce X[T,D] sequences.")


@torch.no_grad()
def model_forward(model: nn.Module, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    TODO: Run forward pass and return dict with keys:
      - "entry_logit"  : [B] or [B,1]
      - "exit_logit"   : [B] or [B,1]
      - "size_score"   : [B] (continuous) or [B,K] (discrete sizing)
    """
    raise NotImplementedError("Implement model_forward(model, x) for your model outputs.")


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

    model = load_model(model_path)
    model.eval()

    X, meta = load_dataset(dataset_path)
    feature_names = list(meta["feature_names"])
    _check_checkpoint_feature_cols(model_path, feature_names)

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

    xt = torch.tensor(Xs, dtype=torch.float32)
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

    baseline_t = torch.tensor(baseline, dtype=torch.float32)
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
