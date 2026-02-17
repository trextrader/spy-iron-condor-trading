"""
chaos_regime.py — Chaos & Regime Classification
Columns: beta1_norm_stub, chaos_membership, te

Chaos membership is derived from vol_energy as a proxy for topological β₁.
Transfer Entropy (te) measures information flow from volume→price.
"""
import numpy as np
import pandas as pd

def _curvature_energy(close: pd.Series, span: int = 64):
    r = np.log(close / close.shift(1))
    dr = r.diff()
    d2r = dr.diff()
    scale = dr.abs().ewm(span=span, adjust=False).mean() + 1e-12
    kappa = (d2r / scale).ewm(span=max(8, span // 4), adjust=False).mean()
    return np.log1p(kappa.abs()).astype(np.float32)

def compute_chaos_membership(close: pd.Series, chaos_thresh: float = 2.0,
                              relax_thresh: float = 1.0) -> pd.Series:
    """
    Chaos membership ∈ [0, 1].
    Based on vol_energy scaled to approximate β₁ (topological invariant).
    Higher vol_energy → more chaotic market structure.
    """
    vol_energy = _curvature_energy(close)
    beta1_proxy = vol_energy * 2  # Scale to useful range
    
    # Fuzzy membership: ramp from relax to chaos threshold
    membership = ((beta1_proxy - relax_thresh) / (chaos_thresh - relax_thresh + 1e-12)).clip(0, 1)
    return membership.astype(np.float32)

def compute_transfer_entropy(close: pd.Series, volume: pd.Series,
                              window: int = 50, bins: int = 5) -> pd.Series:
    """
    Approximate Transfer Entropy: volume → price.
    
    Measures how much knowing past volume reduces uncertainty about future price.
    Uses binned approximation for computational efficiency.
    """
    log_ret = np.log(close / close.shift(1)).fillna(0)
    log_vol = np.log1p(volume).fillna(0)
    
    # Bin the series
    ret_q = pd.qcut(log_ret, bins, labels=False, duplicates='drop').fillna(0)
    vol_q = pd.qcut(log_vol, bins, labels=False, duplicates='drop').fillna(0)
    
    te_values = np.zeros(len(close), dtype=np.float32)
    
    for i in range(window, len(close)):
        # Simplified TE: mutual information between lagged volume and current returns
        ret_win = ret_q.iloc[i-window:i].values
        vol_win = vol_q.iloc[i-window-1:i-1].values
        
        if len(ret_win) == len(vol_win) and len(ret_win) > 0:
            # Joint histogram
            joint = np.histogram2d(vol_win.astype(float), ret_win.astype(float),
                                    bins=bins, density=True)[0]
            joint = joint / (joint.sum() + 1e-12)
            
            # Marginals
            p_vol = joint.sum(axis=1) + 1e-12
            p_ret = joint.sum(axis=0) + 1e-12
            
            # MI ≈ TE (simplified)
            mi = 0.0
            for a in range(min(bins, joint.shape[0])):
                for b in range(min(bins, joint.shape[1])):
                    if joint[a, b] > 1e-12:
                        mi += joint[a, b] * np.log(joint[a, b] / (p_vol[a] * p_ret[b] + 1e-12))
            te_values[i] = max(mi, 0)
    
    return pd.Series(te_values, index=close.index).astype(np.float32)

def compute_beta1_stub() -> float:
    """
    β₁ topological invariant — stubbed for V4.3.
    Full TDA computation deferred to GPU-accelerated pipeline.
    """
    return 0.0

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    
    df = pd.read_csv(args.input)
    c = df['close']
    vol = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)
    
    df['chaos_membership'] = compute_chaos_membership(c)
    df['te'] = compute_transfer_entropy(c, vol)
    df['beta1_norm_stub'] = 0.0
    
    df.to_csv(args.output, index=False)
    print(f"[OK] chaos_membership, te, beta1_norm_stub → {args.output}")
