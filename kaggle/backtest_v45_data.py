"""backtest_v45_data.py — Multi-TF Data Loading and CondorNet v4.3 Inference
===========================================================================
Provides the data layer for condor_brain_backtest_v45.py.

Responsibilities:
  1. Load all 5 datasets (M1, M5, M15, H1, options chain)
  2. Align M1/M15/H1 to M5 canonical clock via timestamp matching
  3. Apply robust z-score normalization (fit on train split, apply to all)
  4. Build per-bar chain grids [N_contracts, 10] for model inference
  5. Run batched CondorNet v4.3 inference → per-bar named output arrays
  6. Provide M5-clocked simulation state for the backtest loop

Dataset stack (all in data/Datasetv4/v43/ by default):
  m1_dataset_v43_final.csv    — 1-min bars  (~92k rows)
  m5_dataset_v43_final.csv    — 5-min bars  (~18k rows) ← canonical clock
  m15_dataset_v43_final.csv   — 15-min bars (~6k rows)
  h1_dataset_v43_final.csv    — 1-hour bars (~1.7k rows)
  options_2025_v43.csv        — options chain (~2.3M rows)
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ── Path bootstrap ────────────────────────────────────────────────────────────
try:
    _repo = Path(__file__).resolve().parent.parent
    if str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))
except NameError:
    pass
for _p in ['/content/spy-iron-condor-trading', '/kaggle/working/spy-iron-condor-trading']:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Schema imports ────────────────────────────────────────────────────────────
try:
    from intelligence.schema_v43 import (
        TF_FEATURE_NAMES, TF_PIVOT_FEATURES, POS_STATE_NAMES,
        CHAIN_FEATURE_NAMES, CHAIN_GRID_CONFIG, N_POS_STATE,
    )
    N_TF_FEAT   = len(TF_FEATURE_NAMES)    # 52
    N_PIV_FEAT  = len(TF_PIVOT_FEATURES)   # 13
    N_CHAIN_FEAT = len(CHAIN_FEATURE_NAMES) # 10
    MAX_CONTRACTS = CHAIN_GRID_CONFIG['max_contracts']  # 120
    _SCHEMA_OK = True
except ImportError as _e:
    warnings.warn(f"[backtest_v45_data] schema_v43 not found ({_e}) — using defaults")
    TF_FEATURE_NAMES  = []
    TF_PIVOT_FEATURES = []
    POS_STATE_NAMES   = []
    N_TF_FEAT         = 52
    N_PIV_FEAT        = 13
    N_CHAIN_FEAT      = 10
    MAX_CONTRACTS     = 120
    _SCHEMA_OK        = False


# =============================================================================
# MultiTFDataBundle — holds all pre-loaded, pre-aligned arrays
# =============================================================================

@dataclass
class MultiTFDataBundle:
    """
    All 5 datasets loaded and aligned to M5 canonical clock.

    Feature arrays are robust-z-score normalized (fit on train 70%).
    Pivot arrays are NOT normalized (sparse; NaN = no event).
    All arrays share the same first dimension N = len(M5 bars).

    chain_grid_by_date : for CondorNet v4.3 inference
    chain_df_by_date   : raw DataFrame slices for find_best_legs()
    """
    # Normalized feature arrays [N, N_TF_FEAT]
    m1_feat:   np.ndarray
    m5_feat:   np.ndarray
    m15_feat:  np.ndarray
    h1_feat:   np.ndarray

    # Raw pivot arrays [N, N_PIV_FEAT] — NaN preserved
    m1_piv:    np.ndarray
    m5_piv:    np.ndarray
    m15_piv:   np.ndarray
    h1_piv:    np.ndarray

    # Options chain — keyed by YYYY-MM-DD date string
    chain_grid_by_date: Dict[str, np.ndarray]   # date → [N_contracts, 10] for inference
    chain_df_by_date:   Dict[str, pd.DataFrame]  # date → raw DataFrame for leg selection

    # Position state (Phase 3) — None if not yet computed
    pos_state: Optional[np.ndarray]             # [N, 11] or None

    # Simulation clock (M5-based)
    m5_timestamps: np.ndarray   # [N] datetime64 — canonical bar timestamps
    m5_dates:      np.ndarray   # [N] str YYYY-MM-DD — for chain lookup
    m5_spot:       np.ndarray   # [N] float32 — M5 close prices (simulation spot)

    # Metadata
    n_bars:      int            # = N
    data_dir:    str            # source directory
    files:       Dict[str, str] = field(default_factory=dict)  # filename map

    # Raw (un-normalized) M5 feature DataFrame — used by select_best_template()
    # for compute_predicate_atoms() which needs real feature values (ivr_zone, rsi_dyn, etc.)
    m5_df_raw:   Optional[pd.DataFrame] = None  # [N rows × TF_FEATURE_NAMES cols]

    # GPU-pinned tensors (populated by preload_to_gpu(); None until called)
    _gpu: Optional[Dict] = None  # keys: m1, m5, m15, h1, m5_piv, m5_piv_mask, pos_state

    def preload_to_gpu(self, device) -> None:
        """
        Pre-convert all fixed-size feature arrays to GPU tensors once at load time.
        Eliminates per-batch numpy→GPU transfers in run_v43_batch_inference.
        Also pre-processes pivot NaN→0 and builds the boolean mask.
        """
        import torch
        if str(device) == 'cpu':
            return  # no-op on CPU — saves RAM
        print(f"[GPU] Pinning feature arrays to {device}...")
        piv = self.m5_piv.astype(np.float32)
        piv_mask = np.isnan(piv)
        piv[piv_mask] = 0.0
        ps = self.pos_state if self.pos_state is not None else np.zeros(
            (self.n_bars, N_POS_STATE), dtype=np.float32)
        self._gpu = {
            'm1':         torch.from_numpy(self.m1_feat).to(device),
            'm5':         torch.from_numpy(self.m5_feat).to(device),
            'm15':        torch.from_numpy(self.m15_feat).to(device),
            'h1':         torch.from_numpy(self.h1_feat).to(device),
            'm5_piv':     torch.from_numpy(piv).to(device),
            'm5_piv_mask':torch.from_numpy(piv_mask).to(device),
            'pos_state':  torch.from_numpy(ps.astype(np.float32)).to(device),
        }
        total_mb = sum(t.element_size() * t.nelement() for t in self._gpu.values()) / 1e6
        print(f"[GPU] Pinned {len(self._gpu)} tensors ({total_mb:.1f} MB) to {device}")


# =============================================================================
# Private helpers
# =============================================================================

def _load_tf_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a TF CSV and extract:
      timestamps : [N] datetime64
      features   : [N, N_TF_FEAT] float32  (missing cols → 0)
      pivots     : [N, N_PIV_FEAT] float64 (NaN preserved)
      close_col  : [N] float32 close prices

    Returns (timestamps, features, pivots, close_prices).
    """
    df = pd.read_csv(csv_path, low_memory=False)

    ts_col = 'timestamp' if 'timestamp' in df.columns else df.columns[0]
    timestamps = pd.to_datetime(df[ts_col], utc=False, errors='coerce').values

    # Feature array
    feat = np.zeros((len(df), N_TF_FEAT), dtype=np.float32)
    for j, col in enumerate(TF_FEATURE_NAMES):
        if col in df.columns:
            feat[:, j] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).values.astype(np.float32)

    # Pivot array (NaN = no event — never imputed)
    piv = np.full((len(df), N_PIV_FEAT), np.nan, dtype=np.float64)
    for j, col in enumerate(TF_PIVOT_FEATURES):
        if col in df.columns:
            piv[:, j] = pd.to_numeric(df[col], errors='coerce').values.astype(np.float64)

    # Close prices for spot simulation
    close = pd.to_numeric(df.get('close', pd.Series(np.zeros(len(df)))),
                          errors='coerce').ffill().values.astype(np.float32)

    return timestamps, feat, piv, close


def _align_to_m5(m5_ts: np.ndarray,
                 other_ts: np.ndarray,
                 other_feat: np.ndarray,
                 other_piv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each M5 timestamp, find the most recent other_ts <= m5_ts (forward-fill).
    Works for both coarser TFs (M15, H1 → repeat last bar) and finer TFs (M1 → use
    last M1 bar in each 5-min window).

    Uses np.searchsorted O(N log N) — safe for large arrays.
    Falls back to zeros/NaN for M5 bars before the other TF begins.
    """
    m5_dt    = pd.to_datetime(m5_ts).asi8     # int64 nanoseconds
    other_dt = pd.to_datetime(other_ts).asi8

    # For each m5 position: last other index where other_dt <= m5_dt
    idxs = np.searchsorted(other_dt, m5_dt, side='right') - 1
    valid = idxs >= 0

    aligned_feat = np.zeros((len(m5_ts), other_feat.shape[1]), dtype=np.float32)
    aligned_piv  = np.full((len(m5_ts), other_piv.shape[1]), np.nan, dtype=np.float64)

    aligned_feat[valid] = other_feat[idxs[valid]]
    aligned_piv[valid]  = other_piv[idxs[valid]]

    return aligned_feat, aligned_piv


def _robust_zscore(feat: np.ndarray, fit_frac: float = 0.70) -> np.ndarray:
    """
    Robust z-score normalization: (X - median) / (1.4826 × MAD).
    Fit on first fit_frac rows (train split), apply to all rows.
    Clip to ±10. Passthrough for constant columns.
    """
    N = len(feat)
    fit_end = max(1, int(N * fit_frac))
    X_fit = feat[:fit_end]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mu  = np.nanmedian(X_fit, axis=0)
        mad = np.nanmedian(np.abs(X_fit - mu), axis=0)

    mad = np.where(mad < 1e-6, 1.0, mad)  # avoid div-by-zero on constant cols
    out = (feat - mu) / (1.4826 * mad)
    return np.clip(out, -10.0, 10.0).astype(np.float32)


def _build_chain_grid(row_group: pd.DataFrame,
                      spot: float,
                      max_contracts: int = MAX_CONTRACTS) -> np.ndarray:
    """
    Convert a raw options chain snapshot (one date) to model input grid.

    Returns float32 [N_contracts, 10] where 10 = CHAIN_FEATURE_NAMES:
      [moneyness, days_to_exp, iv, delta, gamma, theta, vega, bid, ask, oi_norm]

    Selects top max_contracts by open_interest.
    """
    df = row_group.copy()
    needed = ['strike', 'expiry_date', 'iv', 'delta', 'gamma',
              'theta', 'vega', 'bid', 'ask', 'open_interest']
    for col in needed:
        if col not in df.columns:
            df[col] = 0.0

    # Sort by OI descending, take top max_contracts
    df['open_interest'] = pd.to_numeric(df['open_interest'], errors='coerce').fillna(0.0)
    df = df.nlargest(max_contracts, 'open_interest').reset_index(drop=True)

    if len(df) == 0:
        return np.zeros((0, N_CHAIN_FEAT), dtype=np.float32)

    # Parse expiry for days_to_exp
    try:
        ts_col = 'timestamp' if 'timestamp' in df.columns else df.columns[0]
        ref_date = pd.to_datetime(df[ts_col].iloc[0]).date()
    except Exception:
        ref_date = None

    exp_dates = pd.to_datetime(df['expiry_date'], errors='coerce')
    if ref_date is not None:
        days_to_exp = (exp_dates - pd.Timestamp(ref_date)).dt.days.fillna(0).values.astype(np.float32)
    else:
        days_to_exp = np.zeros(len(df), dtype=np.float32)

    strike = pd.to_numeric(df['strike'], errors='coerce').fillna(spot).values.astype(np.float32)
    moneyness = np.log(np.clip(spot, 1e-4, None) / np.clip(strike, 1e-4, None)).astype(np.float32)

    iv    = pd.to_numeric(df['iv'],    errors='coerce').fillna(0.0).values.astype(np.float32)
    delta = pd.to_numeric(df['delta'], errors='coerce').fillna(0.0).values.astype(np.float32)
    gamma = pd.to_numeric(df['gamma'], errors='coerce').fillna(0.0).values.astype(np.float32)
    theta = pd.to_numeric(df['theta'], errors='coerce').fillna(0.0).values.astype(np.float32)
    vega  = pd.to_numeric(df['vega'],  errors='coerce').fillna(0.0).values.astype(np.float32)
    bid   = pd.to_numeric(df['bid'],   errors='coerce').fillna(0.0).values.astype(np.float32)
    ask   = pd.to_numeric(df['ask'],   errors='coerce').fillna(0.0).values.astype(np.float32)
    oi    = pd.to_numeric(df['open_interest'], errors='coerce').fillna(0.0).values
    oi_max = max(float(np.log1p(oi).max()), 1e-6)
    oi_norm = (np.log1p(oi) / oi_max).astype(np.float32)

    grid = np.stack([moneyness, days_to_exp, iv, delta, gamma,
                     theta, vega, bid, ask, oi_norm], axis=1).astype(np.float32)
    return grid


# =============================================================================
# Public API — load_multi_tf_bundle
# =============================================================================

def load_multi_tf_bundle(
    data_dir: str = 'data/Datasetv4/v43',
    m1_file:  str = 'm1_dataset_v43_final.csv',
    m5_file:  str = 'm5_dataset_v43_final.csv',
    m15_file: str = 'm15_dataset_v43_final.csv',
    h1_file:  str = 'h1_dataset_v43_final.csv',
    chain_file: str = 'options_2025_v43.csv',
    norm_fit_frac: float = 0.70,
    verbose: bool = True,
) -> MultiTFDataBundle:
    """
    Load all 5 datasets and align to M5 canonical clock.

    Steps:
      1. Load M5 (canonical clock)
      2. Load M1, M15, H1 and align to M5 via timestamp matching
      3. Normalize all TF features (robust z-score, fit on train 70%)
      4. Load options chain — build inference grids + raw DataFrames per date
      5. Load position state (Phase 3) if present in M5 CSV
      6. Return MultiTFDataBundle
    """
    data_dir = Path(data_dir)

    def _p(f): return str(data_dir / f)

    # ── 1. Load M5 (canonical clock) ────────────────────────────────────────
    if verbose: print(f"[DATA] Loading M5 (canonical) from {m5_file}...")
    m5_ts, m5_raw_feat, m5_piv, m5_close = _load_tf_csv(_p(m5_file))
    N = len(m5_ts)
    m5_dates = pd.to_datetime(m5_ts).strftime('%Y-%m-%d').values.astype(str)
    if verbose: print(f"  M5: {N:,} bars, {m5_raw_feat.shape[1]} features")
    # Build raw (un-normalized) M5 DataFrame for predicate atom evaluation
    m5_df_raw = pd.DataFrame(m5_raw_feat, columns=TF_FEATURE_NAMES) if TF_FEATURE_NAMES else None

    # Also load pos_state from M5 CSV if present
    pos_state: Optional[np.ndarray] = None
    try:
        df_m5_check = pd.read_csv(_p(m5_file), usecols=POS_STATE_NAMES or ['ps_pnl_pct'],
                                  nrows=1, low_memory=False)
        if all(c in df_m5_check.columns for c in POS_STATE_NAMES):
            df_m5_ps = pd.read_csv(_p(m5_file), usecols=POS_STATE_NAMES, low_memory=False)
            pos_state = df_m5_ps[POS_STATE_NAMES].fillna(0.0).values.astype(np.float32)
            if verbose: print(f"  M5: Phase 3 pos_state loaded {pos_state.shape}")
        else:
            if verbose: print("  M5: Phase 3 pos_state absent — pass None to model")
    except Exception:
        pass

    # ── 2. Load M1 ───────────────────────────────────────────────────────────
    if verbose: print(f"[DATA] Loading M1 from {m1_file}...")
    m1_ts, m1_raw_feat, m1_piv, _ = _load_tf_csv(_p(m1_file))
    if verbose: print(f"  M1: {len(m1_ts):,} bars → aligning to M5 clock...")
    m1_feat_aligned, m1_piv_aligned = _align_to_m5(m5_ts, m1_ts, m1_raw_feat, m1_piv)

    # ── 3. Load M15 ──────────────────────────────────────────────────────────
    if verbose: print(f"[DATA] Loading M15 from {m15_file}...")
    m15_ts, m15_raw_feat, m15_piv, _ = _load_tf_csv(_p(m15_file))
    if verbose: print(f"  M15: {len(m15_ts):,} bars → aligning to M5 clock...")
    m15_feat_aligned, m15_piv_aligned = _align_to_m5(m5_ts, m15_ts, m15_raw_feat, m15_piv)

    # ── 4. Load H1 ───────────────────────────────────────────────────────────
    if verbose: print(f"[DATA] Loading H1 from {h1_file}...")
    h1_ts, h1_raw_feat, h1_piv, _ = _load_tf_csv(_p(h1_file))
    if verbose: print(f"  H1: {len(h1_ts):,} bars → aligning to M5 clock...")
    h1_feat_aligned, h1_piv_aligned = _align_to_m5(m5_ts, h1_ts, h1_raw_feat, h1_piv)

    # ── 5. Normalize all TF features ─────────────────────────────────────────
    if verbose: print(f"[DATA] Normalizing features (robust z-score, fit_frac={norm_fit_frac})...")
    m5_feat_norm   = _robust_zscore(m5_raw_feat,       fit_frac=norm_fit_frac)
    m1_feat_norm   = _robust_zscore(m1_feat_aligned,   fit_frac=norm_fit_frac)
    m15_feat_norm  = _robust_zscore(m15_feat_aligned,  fit_frac=norm_fit_frac)
    h1_feat_norm   = _robust_zscore(h1_feat_aligned,   fit_frac=norm_fit_frac)
    if verbose: print(f"  Normalization complete.")

    # ── 6. Load options chain ────────────────────────────────────────────────
    chain_grid_by_date: Dict[str, np.ndarray] = {}
    chain_df_by_date:   Dict[str, pd.DataFrame] = {}
    chain_path = _p(chain_file)

    if os.path.exists(chain_path):
        if verbose: print(f"[DATA] Loading options chain from {chain_file}...")
        chain_df_all = pd.read_csv(chain_path, low_memory=False)

        ts_col = 'timestamp' if 'timestamp' in chain_df_all.columns else chain_df_all.columns[0]
        chain_df_all['_date'] = pd.to_datetime(
            chain_df_all[ts_col], errors='coerce').dt.strftime('%Y-%m-%d')

        # Build per-date spot price from M5 close for moneyness computation
        m5_spot_by_date: Dict[str, float] = {}
        for date_str, spot_val in zip(m5_dates, m5_close):
            if date_str not in m5_spot_by_date:
                m5_spot_by_date[date_str] = float(spot_val)

        n_dates = 0
        for date_str, grp in chain_df_all.groupby('_date'):
            chain_df_by_date[date_str] = grp.reset_index(drop=True)
            spot = m5_spot_by_date.get(str(date_str), 500.0)
            chain_grid_by_date[str(date_str)] = _build_chain_grid(grp, spot, MAX_CONTRACTS)
            n_dates += 1

        if verbose:
            print(f"  Chain: {len(chain_df_all):,} rows across {n_dates} dates")
    else:
        if verbose:
            print(f"  [WARN] Chain file not found: {chain_path} — inference will use empty chain")

    return MultiTFDataBundle(
        m1_feat=m1_feat_norm,
        m5_feat=m5_feat_norm,
        m15_feat=m15_feat_norm,
        h1_feat=h1_feat_norm,
        m1_piv=m1_piv_aligned,
        m5_piv=m5_piv,
        m15_piv=m15_piv_aligned,
        h1_piv=h1_piv_aligned,
        chain_grid_by_date=chain_grid_by_date,
        chain_df_by_date=chain_df_by_date,
        pos_state=pos_state,
        m5_timestamps=m5_ts,
        m5_dates=m5_dates,
        m5_spot=m5_close,
        n_bars=N,
        data_dir=str(data_dir),
        files={
            'm1': m1_file, 'm5': m5_file,
            'm15': m15_file, 'h1': h1_file,
            'chain': chain_file,
        },
        m5_df_raw=m5_df_raw,
    )


# =============================================================================
# Public API — run_v43_batch_inference
# =============================================================================

def run_v43_batch_inference(
    bundle:     MultiTFDataBundle,
    model,                          # CondorNetV43 instance
    device,                         # torch.device
    seq_len:    int   = 64,
    batch_size: int   = 32,
    verbose:    bool  = True,
    max_bars:   int   = 0,          # 0 = all bars; >0 = limit for fast debug runs
) -> Dict[str, np.ndarray]:
    """
    Run CondorNet v4.3 batch inference over all M5 bars.

    For each bar i in [seq_len, N), constructs input window [seq_len] and
    runs model forward. Returns per-bar output arrays, all length N with
    zeros/defaults for bars < seq_len (warmup window).

    Output dict keys:
      entry_signal   [N] float32  ∈ [0,1] — entry confidence
      exit_signal    [N] float32  ∈ [0,1] — exit probability
      pop            [N] float32  ∈ [0,1] — probability of profit
      ev             [N] float32          — expected value
      position_size  [N] float32  ∈ [0,1] — fuzzy position size multiplier
      strategy_idx   [N] int32    ∈ [0,9] — argmax of strategy logits (9=abstain)
      abstain        [N] bool             — True if max softmax < threshold
    """
    import torch
    try:
        from tqdm import tqdm
        _tqdm = tqdm
    except ImportError:
        _tqdm = lambda x, **kw: x  # noqa

    N = bundle.n_bars if max_bars <= 0 else min(bundle.n_bars, max_bars + seq_len)
    results: Dict[str, np.ndarray] = {
        'entry_signal':   np.full(N, 0.5,  dtype=np.float32),
        'exit_signal':    np.full(N, 0.5,  dtype=np.float32),
        'pop':            np.full(N, 0.5,  dtype=np.float32),
        'ev':             np.zeros(N,       dtype=np.float32),
        'position_size':  np.full(N, 0.5,  dtype=np.float32),
        'strategy_idx':   np.full(N, 9,    dtype=np.int32),   # 9 = abstain
        'abstain':        np.ones(N,        dtype=bool),
    }

    model.eval()
    _EMPTY_CHAIN  = np.zeros((1, N_CHAIN_FEAT), dtype=np.float32)
    _EMPTY_CMASK  = np.ones(1, dtype=bool)

    bar_indices = list(range(seq_len, N))
    if verbose:
        print(f"[INFERENCE] Running CondorNet v4.3 on {len(bar_indices):,} bars "
              f"(seq_len={seq_len}, batch={batch_size})...")

    for b_start in _tqdm(range(0, len(bar_indices), batch_size), desc="v4.3 Inference"):
        b_indices = bar_indices[b_start: b_start + batch_size]
        B = len(b_indices)

        # ── Build batch inputs ───────────────────────────────────────────────
        chain_list, cmask_list = [], []
        for i in b_indices:
            date_str = bundle.m5_dates[i]
            chain_arr = bundle.chain_grid_by_date.get(str(date_str))
            if chain_arr is None or len(chain_arr) == 0:
                chain_list.append(_EMPTY_CHAIN.copy())
                cmask_list.append(_EMPTY_CMASK.copy())
            else:
                if len(chain_arr) > MAX_CONTRACTS:
                    chain_arr = chain_arr[:MAX_CONTRACTS]
                chain_list.append(chain_arr)
                cmask_list.append(np.zeros(len(chain_arr), dtype=bool))

        # ── Stack fixed-size tensors (GPU fast-path if pre-pinned) ───────────
        if bundle._gpu is not None:
            # GPU fast-path: torch.stack of contiguous slice views (avoids non-contiguous
            # tensor bug from 2D advanced indexing which corrupts model outputs)
            g = bundle._gpu
            x_m1_t  = torch.stack([g['m1'][i - seq_len:i]       for i in b_indices])
            x_m5_t  = torch.stack([g['m5'][i - seq_len:i]       for i in b_indices])
            x_m15_t = torch.stack([g['m15'][i - seq_len:i]      for i in b_indices])
            x_h1_t  = torch.stack([g['h1'][i - seq_len:i]       for i in b_indices])
            piv_f_t = torch.stack([g['m5_piv'][i - seq_len:i]      for i in b_indices])
            piv_m_t = torch.stack([g['m5_piv_mask'][i - seq_len:i] for i in b_indices])
            ps_t    = torch.stack([g['pos_state'][i - seq_len:i]    for i in b_indices])
        else:
            # CPU fallback (no GPU or preload_to_gpu() not called)
            x_m1_list, x_m5_list, x_m15_list, x_h1_list = [], [], [], []
            piv_feat_list, piv_mask_list, ps_list = [], [], []
            for i in b_indices:
                s = i - seq_len
                x_m1_list.append(bundle.m1_feat[s:i])
                x_m5_list.append(bundle.m5_feat[s:i])
                x_m15_list.append(bundle.m15_feat[s:i])
                x_h1_list.append(bundle.h1_feat[s:i])
                piv = bundle.m5_piv[s:i].astype(np.float32)
                pmask = np.isnan(piv)
                piv[pmask] = 0.0
                piv_feat_list.append(piv)
                piv_mask_list.append(pmask)
                if bundle.pos_state is not None:
                    ps_list.append(bundle.pos_state[s:i])
                else:
                    ps_list.append(np.zeros((seq_len, N_POS_STATE), dtype=np.float32))
            x_m1_t  = torch.tensor(np.stack(x_m1_list),     dtype=torch.float32, device=device)
            x_m5_t  = torch.tensor(np.stack(x_m5_list),     dtype=torch.float32, device=device)
            x_m15_t = torch.tensor(np.stack(x_m15_list),    dtype=torch.float32, device=device)
            x_h1_t  = torch.tensor(np.stack(x_h1_list),     dtype=torch.float32, device=device)
            piv_f_t = torch.tensor(np.stack(piv_feat_list), dtype=torch.float32, device=device)
            piv_m_t = torch.tensor(np.stack(piv_mask_list), dtype=torch.bool,    device=device)
            ps_t    = torch.tensor(np.stack(ps_list),        dtype=torch.float32, device=device)

        # Pad chains to batch-max N_contracts
        max_n = max(len(c) for c in chain_list)
        chain_pad  = np.zeros((B, max_n, N_CHAIN_FEAT), dtype=np.float32)
        cmask_pad  = np.ones((B, max_n), dtype=bool)
        for j, (c, cm) in enumerate(zip(chain_list, cmask_list)):
            chain_pad[j, :len(c)]  = c
            cmask_pad[j, :len(cm)] = cm
        chain_t = torch.tensor(chain_pad, dtype=torch.float32, device=device)
        cmask_t = torch.tensor(cmask_pad, dtype=torch.bool,    device=device)

        # ── Forward pass ─────────────────────────────────────────────────────
        with torch.no_grad():
            out = model(
                x_m1_t, x_m5_t, x_m15_t, x_h1_t,
                chain=chain_t,
                chain_mask=cmask_t,
                pivot_features=piv_f_t,
                pivot_mask=piv_m_t,
                pos_state=ps_t,
                return_diagnostics=False,
            )

        # ── Extract per-bar outputs ───────────────────────────────────────────
        es   = out.entry_signal.cpu().float().numpy()   # [B, 1]
        xs   = out.exit_signal.cpu().float().numpy()    # [B, 1]
        pp   = out.pop.cpu().float().numpy()            # [B, 1]
        ev   = out.ev.cpu().float().numpy()             # [B, 1]
        ps   = out.position_size.cpu().float().numpy()  # [B, 1]
        sidx = out.strategy_type_idx.cpu().numpy()      # [B]
        abt  = out.abstain_mask.cpu().numpy()           # [B] bool

        for j, i in enumerate(b_indices):
            results['entry_signal'][i]  = float(es[j, 0])
            results['exit_signal'][i]   = float(xs[j, 0])
            results['pop'][i]           = float(pp[j, 0])
            results['ev'][i]            = float(ev[j, 0])
            results['position_size'][i] = float(ps[j, 0])
            results['strategy_idx'][i]  = int(sidx[j])
            results['abstain'][i]       = bool(abt[j])

    if verbose:
        n_valid = int((~results['abstain'][seq_len:]).sum())
        n_entry = int((results['entry_signal'][seq_len:] > 0.55).sum())
        print(f"  Done. Non-abstain bars: {n_valid:,} | "
              f"entry_signal > 0.55: {n_entry:,}")

    return results


# =============================================================================
# Public API — normalize_v43_chain
# =============================================================================

def normalize_v43_chain(chain_df: pd.DataFrame, spot: float,
                        min_spread_ratio: float = 0.02) -> pd.DataFrame:
    """
    Normalize v43 options chain column names so synthesize_chain_prices +
    find_best_legs work without modification.

    v43 CSV (options_2025_v43.csv) columns:
        contract_id, type (call/put), implied_volatility, expiration,
        bid, bid_size, ask, ask_size, mark, delta, gamma, theta, vega,
        volume, open_interest, in_the_money

    Expected downstream:
        option_symbol, call_put (C/P), iv, te (DTE days), underlying_price,
        mid, close (legacy compat), spread_ratio

    Since v43 chain is already BSM-priced (bid/ask/mark from SyntheticOptionsEngine),
    no re-synthesis is needed — we just provide the correct column names so
    synthesize_chain_prices can run idempotently if called again inside find_best_legs.
    """
    if chain_df is None or chain_df.empty:
        return chain_df

    df = chain_df.copy()

    # ── Coerce all numeric columns first (CSV loads may give object dtype) ──
    _numeric_cols = [
        'strike', 'last', 'mark', 'bid', 'bid_size', 'ask', 'ask_size',
        'volume', 'open_interest', 'implied_volatility',
        'delta', 'gamma', 'theta', 'vega', 'rho',
    ]
    for _col in _numeric_cols:
        if _col in df.columns:
            df[_col] = pd.to_numeric(df[_col], errors='coerce').fillna(0.0)

    # option_symbol ← contract_id
    if 'contract_id' in df.columns and 'option_symbol' not in df.columns:
        df['option_symbol'] = df['contract_id']

    # call_put ← type  ('call' → 'C', 'put' → 'P')
    if 'type' in df.columns and 'call_put' not in df.columns:
        df['call_put'] = df['type'].str.lower().map({'call': 'C', 'put': 'P'}).fillna('C')

    # iv ← implied_volatility
    if 'implied_volatility' in df.columns and 'iv' not in df.columns:
        df['iv'] = df['implied_volatility']

    # underlying_price ← spot  (used by synthesize_chain_prices BSM formula)
    df['underlying_price'] = float(spot)

    # te (DTE in calendar days) ← expiration − bar_date
    if 'te' not in df.columns and 'expiration' in df.columns:
        try:
            exp_dates = pd.to_datetime(df['expiration'])
            if 'timestamp' in df.columns:
                bar_date = pd.to_datetime(df['timestamp'].iloc[0]).normalize()
            else:
                bar_date = pd.Timestamp.now().normalize()
            df['te'] = (exp_dates - bar_date).dt.days.clip(lower=0).astype(float)
        except Exception:
            df['te'] = 0.0

    # mid ← mark  (already BSM-priced; fallback to (bid+ask)/2)
    if 'mid' not in df.columns:
        if 'mark' in df.columns:
            df['mid'] = df['mark'].clip(lower=0.01)
        elif 'bid' in df.columns and 'ask' in df.columns:
            df['mid'] = ((df['bid'] + df['ask']) / 2.0).clip(lower=0.01)
        else:
            df['mid'] = 0.10

    # close ← mid  (legacy compat: find_best_legs reads chain['close'])
    if 'close' not in df.columns:
        df['close'] = df['mid']

    # spread_ratio ← computed from bid/ask  (used by validate_leg_liquidity)
    if 'spread_ratio' not in df.columns:
        if 'bid' in df.columns and 'ask' in df.columns:
            mid_safe = df['mid'].clip(lower=0.01)
            spread   = (df['ask'] - df['bid']).clip(lower=0.0)
            df['spread_ratio'] = (spread / mid_safe).clip(
                lower=min_spread_ratio, upper=2.0
            )
        else:
            df['spread_ratio'] = min_spread_ratio

    return df


# =============================================================================
# Public API — load_v43_model
# =============================================================================

def load_v43_model(checkpoint_path: str, device, d_tf_in: int = 0, verbose: bool = True):
    """
    Load CondorNet v4.3 from checkpoint.

    Reconstructs model architecture from the 'config' dict saved inside the
    checkpoint (written by condor_train_net_v43.py).  Falls back to schema
    defaults only when the checkpoint has no config key.

    Key-alias remapping mirrors audit_condornet_interpretability_v43.py so
    both scripts accept the same checkpoint files without size-mismatch errors.

    Returns model in eval() mode on device.
    """
    import inspect
    import torch
    from intelligence.condor_brain_net_v43 import CondorNetV43, build_condornet_v43
    from intelligence.schema_v43 import (
        N_PIVOT_FEATURES, CHAIN_GRID_CONFIG, N_STRATEGY_TYPES,
    )

    # ── Step 1: load raw checkpoint (with fallback path search) ─────────────
    # If the exact path is not found, try common alternate names so the backtest
    # works even when the checkpoint was saved under a different filename.
    _resolved_path = checkpoint_path
    if not (checkpoint_path and os.path.exists(checkpoint_path)):
        _models_dir = os.path.dirname(checkpoint_path) if checkpoint_path else "models"
        _fallback_names = [
            "condor_net_v43.pth",          # default training output (most likely)
            "condornet_v43_best.pth",       # alternate training default
            "condor_net_v43_run18.pth",     # underscore variant
            "condornet_v43_run18.pth",      # no-underscore variant
        ]
        _found = None
        for _name in _fallback_names:
            _cand = os.path.join(_models_dir, _name)
            if os.path.exists(_cand):
                _found = _cand
                break
        if _found:
            if verbose:
                print(f"[MODEL] Checkpoint not found at '{checkpoint_path}' — "
                      f"using fallback: {_found}")
            _resolved_path = _found
        else:
            _d = d_tf_in if d_tf_in > 0 else N_TF_FEAT
            if verbose:
                print(f"[MODEL] Checkpoint not found: {checkpoint_path} "
                      f"(tried {len(_fallback_names)} fallback names in '{_models_dir}') "
                      f"— building defaults (d_tf_in={_d})")
                print(f"[MODEL] NOTE: Upload a trained checkpoint to '{_models_dir}/' "
                      f"or run condor_train_net_v43.py to generate one.")
            model = build_condornet_v43(d_tf_in=_d, verbose=False)
            model.to(device).eval()
            return model

    if verbose:
        print(f"[MODEL] Loading checkpoint: {_resolved_path}")
    state = torch.load(_resolved_path, map_location=device, weights_only=False)

    # ── Step 2: recover architecture config saved during training ────────────
    config = dict(state.get('config', {}))

    if config:
        # Remap legacy / alternate key names to CondorNetV43.__init__ names
        key_aliases = {
            'n_pivot_features': 'd_pivot_in',
            'd_pivot':          'd_pivot_proj',
            'chain_d_model':    'd_model_chain',
            'chain_n_heads':    'n_heads_chain',
            'chain_n_layers':   'n_layers_chain',
        }
        # If training saved d_joint but not d_tf_proj, derive d_tf_proj
        if 'd_joint' in config and 'd_tf_proj' not in config:
            config['d_tf_proj'] = config['d_joint'] // 4
        mapped = {key_aliases.get(k, k): v for k, v in config.items()}
        # Override d_tf_in only when caller explicitly provides it
        if d_tf_in > 0:
            mapped['d_tf_in'] = d_tf_in
        if verbose:
            print(f"[MODEL] Restoring from checkpoint config — "
                  f"d_h={mapped.get('d_h')}, d_m={mapped.get('d_m')}, "
                  f"d_r={mapped.get('d_r')}, n_predicates={mapped.get('n_predicates')}, "
                  f"n_sets={mapped.get('n_sets')}, n_super_sets={mapped.get('n_super_sets')}, "
                  f"d_tf_proj={mapped.get('d_tf_proj', 64)}")
    else:
        # No config key — fall back to schema defaults
        _d = d_tf_in if d_tf_in > 0 else N_TF_FEAT
        mapped = {
            'd_tf_in':           _d,
            'd_tf_proj':         64,
            'd_chain':           CHAIN_GRID_CONFIG['d_chain'],
            'd_pivot_proj':      16,
            'd_pivot_in':        N_PIVOT_FEATURES,
            'n_strategy_types':  N_STRATEGY_TYPES,
            'chain_in_features': CHAIN_GRID_CONFIG['in_features'],
            'd_model_chain':     CHAIN_GRID_CONFIG['d_model'],
            'n_heads_chain':     CHAIN_GRID_CONFIG['n_heads'],
            'n_layers_chain':    CHAIN_GRID_CONFIG['n_layers'],
        }
        if verbose:
            print(f"[MODEL] No 'config' in checkpoint — using schema defaults (d_tf_in={_d})")

    # ── Step 3: filter to only valid CondorNetV43 / build_condornet_v43 params
    valid_keys = set(inspect.signature(build_condornet_v43).parameters.keys())
    valid_keys.update(inspect.signature(CondorNetV43.__init__).parameters.keys())
    valid_keys.discard('self')
    valid_keys.discard('kwargs')
    build_cfg = {k: v for k, v in mapped.items() if k in valid_keys}

    if verbose:
        print(f"[MODEL] Building CondorNet v4.3 (d_tf_in={build_cfg.get('d_tf_in', N_TF_FEAT)})...")
    model = build_condornet_v43(**build_cfg)

    # ── Step 4: load weights ─────────────────────────────────────────────────
    sd = state.get('model_state_dict', state.get('state_dict', state))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if verbose:
        print(f"  Loaded. Missing keys: {len(missing)} | Unexpected: {len(unexpected)}")

    model.to(device)
    model.eval()
    return model
