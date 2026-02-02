#!/usr/bin/env python
# intelligence/audit_cde_vs_mamba2_comparison.py

import os
import sys
import argparse
import math
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# Add project root to sys.path to allow absolute imports when run as a script
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.tree import DecisionTreeRegressor, export_text
from scipy.spatial.distance import cosine, jensenshannon
from scipy.stats import wasserstein_distance, spearmanr

from intelligence.canonical_feature_registry import FEATURE_COLS_V22, INPUT_DIM_V22
from intelligence.model_adapter import load_model_any

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-6

HEAD_NAMES = [
    "call_offset","put_offset","wing_width","dte",
    "pop","roi","max_loss","confidence","entry","exit"
]
ROI_HEAD = 5  # roi

# ----------------------------
# Data: streaming window sampler
# ----------------------------

def stream_sample_windows_csv(
    csv_path: str,
    feature_cols: List[str],
    seq_len: int,
    n_samples: int,
    seed: int,
    max_rows: int = 0,
    dtype=np.float32,
) -> np.ndarray:
    """One-pass streaming sampler that returns windows: (n_samples, seq_len, D)"""
    rng = np.random.default_rng(seed)
    D = len(feature_cols)

    buf = np.zeros((seq_len, D), dtype=dtype)
    buf_count = 0

    windows = np.zeros((n_samples, seq_len, D), dtype=dtype)
    seen_windows = 0

    usecols = feature_cols
    dtypes = {c: "float32" for c in usecols}

    rows_read = 0
    print(f"  [Stream] Reading CSV: {csv_path} (seq_len={seq_len})", flush=True)
    
    # Use chunksize to keep memory footprint low
    for chunk in pd.read_csv(csv_path, usecols=usecols, dtype=dtypes, chunksize=100_000):
        Xc = chunk.to_numpy(copy=False)
        m = Xc.shape[0]

        if max_rows and rows_read + m > max_rows:
            Xc = Xc[: max(0, max_rows - rows_read)]
            m = Xc.shape[0]

        for i in range(m):
            buf[buf_count % seq_len] = np.nan_to_num(Xc[i], nan=0.0, posinf=0.0, neginf=0.0)
            buf_count += 1
            rows_read += 1

            if buf_count >= seq_len:
                if seen_windows < n_samples:
                    idx = seen_windows
                    start = buf_count % seq_len
                    if start == 0:
                        windows[idx] = buf
                    else:
                        windows[idx] = np.concatenate([buf[start:], buf[:start]], axis=0)
                else:
                    j = rng.integers(0, seen_windows + 1)
                    if j < n_samples:
                        start = buf_count % seq_len
                        if start == 0:
                            windows[j] = buf
                        else:
                            windows[j] = np.concatenate([buf[start:], buf[:start]], axis=0)
                seen_windows += 1

            if max_rows and rows_read >= max_rows: break
        if max_rows and rows_read >= max_rows: break

    if seen_windows == 0:
        raise RuntimeError(f"No windows formed. Rows={rows_read}, seq_len={seq_len}")

    actual_samples = min(seen_windows, n_samples)
    print(f"  [Stream] Sampled {actual_samples:,} windows from {rows_read:,} rows.", flush=True)
    return windows[:actual_samples]

# ----------------------------
# Metrics: Physics Parity
# ----------------------------

def batch_predict(adapter, X: np.ndarray, batch_size: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    N = X.shape[0]
    ys, ps = [], []
    for s in range(0, N, batch_size):
        xb = torch.from_numpy(X[s:s+batch_size]).to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda"), dtype=torch.bfloat16):
            out = adapter.predict(xb)
            y, pred = out["y"], out["pred_logits"]
        ys.append(y.detach().float().cpu().numpy())
        if pred is not None: ps.append(pred.detach().float().cpu().numpy())
    return np.concatenate(ys, axis=0), (np.concatenate(ps, axis=0) if ps else None)

def lipschitz_stability(adapter, X: np.ndarray, batch_size: int, sigma: float = 0.01) -> float:
    """E[ ||f(X+delta) - f(X)|| / (||delta|| + eps) ]"""
    N = X.shape[0]
    y0, _ = batch_predict(adapter, X, batch_size)
    
    delta = np.random.normal(0, sigma, X.shape).astype(np.float32)
    y1, _ = batch_predict(adapter, X + delta, batch_size)
    
    dy = np.abs(y1[:, ROI_HEAD] - y0[:, ROI_HEAD])
    d_norm = np.linalg.norm(delta.reshape(N, -1), axis=1)
    return float(np.mean(dy / (d_norm + EPS)))

def gradient_analysis(adapter, X: np.ndarray, batch_size: int, max_batches: int = 50) -> Dict[str, Any]:
    """Level saliency (dy/dX) and Increment saliency (dy/dDeltaX)"""
    N, T, D = X.shape
    sal = np.zeros((D,), dtype=np.float64)
    inc_sal = np.zeros((T, D), dtype=np.float64)
    
    batches = 0
    g_batch = max(1, batch_size // 4)
    for s in tqdm(range(0, N, g_batch), desc=f"Gradients ({adapter.info.name})", leave=False):
        xb = torch.from_numpy(X[s:s+g_batch]).to(DEVICE).requires_grad_(True)
        out = adapter.model(xb)
        if isinstance(out, tuple): out = out[0]
        roi = out[:, ROI_HEAD].sum()
        
        adapter.model.zero_grad(set_to_none=True)
        roi.backward()
        
        if xb.grad is not None:
            g = xb.grad.detach().float().cpu().numpy()
            sal += np.abs(g).mean(axis=(0, 1))
            # dy/dDeltaX_t = sum_{k=t}^T (dy/dX_k)
            g_inc = np.abs(np.cumsum(g[:, ::-1, :], axis=1)[:, ::-1, :])
            inc_sal += g_inc.mean(axis=0)
            
        batches += 1
        if batches >= max_batches: break
    
    denom = max(1, batches)
    return {
        "level_saliency": (sal / denom).astype(np.float32).tolist(),
        "increment_saliency": (inc_sal.mean(axis=0) / denom).astype(np.float32).tolist()
    }

# ----------------------------
# Metrics: Standard T4-Safe
# ----------------------------

def fisher_layerwise(adapter, X: np.ndarray, batch_size: int, max_batches: int = 50) -> Dict[str, float]:
    layer_F = {}
    batches = 0
    g_batch = max(1, batch_size // 4)
    for s in tqdm(range(0, X.shape[0], g_batch), desc=f"Fisher ({adapter.info.name})", leave=False):
        xb = torch.from_numpy(X[s:s+g_batch]).to(DEVICE)
        out = adapter.model(xb)
        if isinstance(out, tuple): out = out[0]
        L = -((out - out.mean(dim=0, keepdim=True)) ** 2).sum()
        adapter.model.zero_grad(set_to_none=True)
        L.backward()
        for name, p in adapter.model.named_parameters():
            if p.grad is None: continue
            layer = name.split(".")[0]
            v = float(p.grad.detach().float().pow(2).sum().item())
            layer_F[layer] = layer_F.get(layer, 0.0) + v
        batches += 1
        if batches >= max_batches: break
    for k in layer_F: layer_F[k] /= max(1, batches)
    return layer_F

# ----------------------------
# Main Script
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="T4-Safe Physics Audit")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="reports/audit_physics.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    adapters = []
    print("\n[STEP 1/5] Loading Models...", flush=True)
    for p in args.models:
        ad, _ = load_model_any(p, DEVICE, INPUT_DIM_V22)
        adapters.append(ad)
        print(f"  + {ad.info.name} (CDE={ad.info.use_cde})", flush=True)

    T_common = min(a.info.seq_len for a in adapters)
    X = stream_sample_windows_csv(args.data, FEATURE_COLS_V22, T_common, args.samples, args.seed)

    results = {"generated": datetime.now().isoformat(), "T": T_common, "N": len(X), "models": {}, "pairwise": {}}

    print("\n[STEP 3/5] Computing Architectural & Physics Metrics...", flush=True)
    for ad in adapters:
        print(f"  > Processing {ad.info.name}...", flush=True)
        y, pred = batch_predict(ad, X, args.batch_size)
        
        entry = {
            "info": ad.info.__dict__,
            "stability": lipschitz_stability(ad, X, args.batch_size),
            "gradients": gradient_analysis(ad, X, args.batch_size),
            "fisher": fisher_layerwise(ad, X, args.batch_size),
            "output_mean": y.mean(axis=0).tolist()
        }
        
        if pred is not None:
            # Normalized Predicate Distribution
            probs = 1.0 / (1.0 + np.exp(-(pred - np.median(pred, axis=0))))
            # 2.3 Inequality satisfaction rate (alpha=0.5)
            entry["sat_rate"] = (probs >= 0.5).mean(axis=0).tolist()
            m = probs.mean(axis=0)
            entry["predicate_dist"] = (m / (m.sum() + EPS)).tolist()

        results["models"][ad.info.name] = entry

    if len(adapters) >= 2:
        print("\n[STEP 4/5] Cross-model Divergence Analysis...", flush=True)
        names = list(results["models"].keys())
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                n0, n1 = names[i], names[j]
                m0, m1 = results["models"][n0], results["models"][n1]
                
                div = {
                    "output_cosine": 1.0 - cosine(m0["output_mean"], m1["output_mean"]),
                    "stability_delta": abs(m0["stability"] - m1["stability"]),
                }
                
                # Saliency Divergence (JSD)
                s0, s1 = np.array(m0["gradients"]["level_saliency"]), np.array(m1["gradients"]["level_saliency"])
                div["saliency_jsd"] = float(jensenshannon(s0/s0.sum(), s1/s1.sum())**2)
                
                if "predicate_dist" in m0 and "predicate_dist" in m1:
                    div["predicate_jsd"] = float(jensenshannon(m0["predicate_dist"], m1["predicate_dist"])**2)
                
                results["pairwise"][f"{n0}_vs_{n1}"] = div

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f: json.dump(results, f, indent=2)
    print(f"\n[OK] Audit Complete. Report saved to: {args.output}")

if __name__ == "__main__":
    main()
