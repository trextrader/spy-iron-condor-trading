"""
optimizer_prep.py — Immutable Tensor Context Builder
====================================================
Pre-processes the v45 bundle into flat CSR tensors for the batch optimizer.
Build once; reuse across all BO rounds.

CSR Layout (date-indexed — one copy per unique trading date):
    date_offsets[D+1]    — start/end index into flat arrays for each date
    bar_to_date_idx[T]   — bar index → date index (-1 = no chain for that bar)
    option_right[N]      — 0=call  1=put   (N = total options across D dates)
    option_strike[N]     — float32 strike price
    option_dte[N]        — float32 days-to-expiry (te column)
    option_delta[N]      — float32 abs(delta)  (NaN when unavailable)
    opt_bid[N]           — float32
    opt_ask[N]           — float32
    opt_mid[N]           — float32

Date-indexed layout stores each date's chain ONCE (N = D × avg_opts_per_date)
instead of once per bar (old: N = T × avg_opts_per_date).  For 5-year runs
this reduces chain tensor size ~78× (bars_per_day ≈ 78).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch


@dataclass
class OptimizerContext:
    """Immutable bundle of tensors for K-candidate batch optimization."""
    device: torch.device

    # ── Bar dimension ─────────────────────────────────────────────────────
    T: int                         # number of M5 bars
    timestamps: torch.Tensor       # [T] int64 unix-seconds
    spot: torch.Tensor             # [T] float32 close price
    gate_entry: torch.Tensor       # [T] float32 v43 entry_signal
    gate_pop: torch.Tensor         # [T] float32 v43 pop
    strategy_idx: torch.Tensor     # [T] int16  predicted strategy class idx
    abstain: torch.Tensor          # [T] bool

    # ── Chain CSR (date-indexed) ───────────────────────────────────────────
    date_offsets: torch.Tensor     # [D+1] int64  — one entry per unique trading date
    bar_to_date_idx: torch.Tensor  # [T]   int32  — bar → date index, -1 = no chain
    option_right: torch.Tensor     # [N] int8   0=call  1=put
    option_strike: torch.Tensor    # [N] float32
    option_dte: torch.Tensor       # [N] float32
    option_delta: torch.Tensor     # [N] float32  NaN where unknown
    opt_bid: torch.Tensor          # [N] float32
    opt_ask: torch.Tensor          # [N] float32
    opt_mid: torch.Tensor          # [N] float32

    # ── Fidelity cutpoints (bar indices into [0, T)) ─────────────────────
    fast_end: int                  # ≈ 25 % of T
    medium_end: int                # ≈ 60 % of T

    # ── Convenience (not tensors) ─────────────────────────────────────────
    bar_dates: List[str]           # [T] ISO date string "YYYY-MM-DD"


def build_optimizer_context(
    v43_outputs: Dict,
    bundle,                          # MultiTFDataBundle  (has .m5_df_raw)
    chain_df_by_date: Dict,          # { "YYYY-MM-DD" : pd.DataFrame }
    device: Optional[torch.device] = None,
) -> OptimizerContext:
    """
    Convert pre-loaded bundle + v43 inference outputs into OptimizerContext.

    Parameters
    ----------
    v43_outputs     : dict with keys 'entry_signal', 'pop', 'strategy_idx', 'abstain'
                      each a 1-D array of length T
    bundle          : MultiTFDataBundle  (.m5_df_raw is a DataFrame)
    chain_df_by_date: { date_str : chain_df }  same format as used in run_backtest
    device          : target device (default: CUDA if available, else CPU)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m5 = bundle.m5_df_raw
    T  = len(m5)

    # ── Timestamps ────────────────────────────────────────────────────────
    # m5_df_raw is a plain feature DataFrame (RangeIndex, no timestamp col).
    # Prefer bundle.m5_timestamps (datetime64 array) as the authoritative source.
    # Fall back to m5.index / column only if bundle timestamps unavailable.
    ts_arr = None
    ts_series = None  # only set when ts_arr cannot be built directly

    if hasattr(bundle, 'm5_timestamps') and bundle.m5_timestamps is not None:
        # Direct datetime64 → unix-seconds (most reliable path)
        _ts = np.asarray(bundle.m5_timestamps[:T], dtype='datetime64[s]')
        ts_arr = _ts.view(np.int64)                 # int64 unix-sec numpy array
    elif isinstance(m5.index, pd.DatetimeIndex):
        ts_arr = m5.index[:T].asi8 // 10 ** 9      # DatetimeIndex → unix-sec
    elif 'timestamp' in m5.columns:
        ts_series = pd.to_datetime(m5['timestamp'])
        ts_arr = ts_series.values.astype('datetime64[s]').view(np.int64)
    else:
        # Last resort: integer RangeIndex — pd.to_datetime(RangeIndex) treats
        # values as nanoseconds from epoch, yielding 0 for all small indices.
        # Instead, use bar index × 300 s (5-min bars) as synthetic clock.
        print("[optimizer_prep] WARN: no timestamp source — using bar-index × 300s")
        ts_arr = np.arange(T, dtype=np.int64) * 300

    ts_int = ts_arr   # kept as alias for OptimizerContext construction below

    # Use bundle.m5_dates when available — same key format as chain_df_by_date
    # (chain is indexed by these exact strings, so we must match precisely)
    if hasattr(bundle, 'm5_dates') and bundle.m5_dates is not None:
        bar_dates = [str(d) for d in bundle.m5_dates[:T]]
    elif ts_series is not None:
        bar_dates = [str(t.date()) for t in ts_series]
    else:
        bar_dates = [str(pd.Timestamp(t, unit='s').date()) for t in ts_arr]

    # ── Spot ──────────────────────────────────────────────────────────────
    if 'close' in m5.columns:
        spot_arr = m5['close'].astype(np.float32).values
    elif hasattr(bundle, 'm5_spot') and bundle.m5_spot is not None:
        spot_arr = np.asarray(bundle.m5_spot[:T], dtype=np.float32)
    else:
        spot_arr = m5.iloc[:, 0].astype(np.float32).values

    # ── V43 gate signals (clip to T in case of slight mismatch) ──────────
    def _to_arr(key, dtype):
        a = np.asarray(v43_outputs[key], dtype=dtype)
        return a[:T] if len(a) >= T else np.pad(a, (0, T - len(a)))

    entry_sig = _to_arr('entry_signal', np.float32)
    pop_sig   = _to_arr('pop',          np.float32)
    sidx_arr  = _to_arr('strategy_idx', np.int16)
    abt_arr   = _to_arr('abstain',      bool)

    # ── CSR chain build ───────────────────────────────────────────────────
    # Pre-process each UNIQUE date once (238 dates vs 18,494 bars).
    # Then build the offset array by reusing cached arrays per bar.
    n_unique = len(set(bar_dates))
    print(f"[optimizer_prep] Pre-processing {n_unique} unique chain dates…")

    def _chain_to_arrays(chain, date_str: str = None):
        n   = len(chain)
        cp  = chain.get('call_put', chain.get('type', pd.Series(['C'] * n, index=chain.index)))
        right  = (cp.astype(str).str.upper().str.startswith('P')).astype(np.int8).values
        strike = pd.to_numeric(chain['strike'], errors='coerce').fillna(0).astype(np.float32).values

        # ── DTE resolution (calendar days to expiry) ───────────────────────────
        # Priority: 'te' col → 'dte' col → compute from 'expiration' → fallback
        dte_col = 'te' if 'te' in chain.columns else ('dte' if 'dte' in chain.columns else None)
        if dte_col:
            dte = pd.to_numeric(chain[dte_col], errors='coerce').fillna(0.0).astype(np.float32).values
        elif 'expiration' in chain.columns and date_str is not None:
            # Same logic as backtest_v45_data.py::prepare_chain_for_backtester line 646-653
            try:
                exp_dates = pd.to_datetime(chain['expiration'])
                bar_date  = pd.to_datetime(date_str)
                dte = (exp_dates - bar_date).dt.days.clip(lower=0).astype(np.float32).values
            except Exception:
                dte = np.zeros(n, np.float32)
        else:
            dte = np.zeros(n, np.float32)   # unknown DTE — will be excluded by target_dte search
        delta_col = 'delta' if 'delta' in chain.columns else None
        delta = (pd.to_numeric(chain[delta_col], errors='coerce').astype(np.float32).values
                 if delta_col else np.full(n, np.nan, np.float32))
        bid = pd.to_numeric(chain.get('bid', pd.Series([0.0]*n)), errors='coerce').fillna(0.0).astype(np.float32).values
        ask = pd.to_numeric(chain.get('ask', pd.Series([0.0]*n)), errors='coerce').fillna(0.0).astype(np.float32).values
        mid_raw = chain.get('mid')
        mid = (pd.to_numeric(mid_raw, errors='coerce').fillna((bid+ask)/2).astype(np.float32).values
               if mid_raw is not None else ((bid + ask) / 2).astype(np.float32))
        return right, strike, dte, delta, bid, ask, mid

    # Cache: date_str → (right, strike, dte, delta, bid, ask, mid)
    _date_cache: dict = {}
    for d in set(bar_dates):
        chain = chain_df_by_date.get(d)
        if chain is not None and len(chain) > 0:
            _date_cache[d] = _chain_to_arrays(chain, date_str=d)

    print(f"[optimizer_prep] {len(_date_cache)} dates with chain data  "
          f"(0 = key mismatch; check bundle.m5_dates vs chain_df_by_date keys)")

    # Build date-indexed CSR: one copy per unique date (not one per bar).
    # For 5-year runs this cuts chain_N from ~889M to ~11.4M (78× reduction).
    _unique_dates_ordered = sorted(_date_cache.keys())
    D = len(_unique_dates_ordered)
    _date_to_didx = {d: i for i, d in enumerate(_unique_dates_ordered)}

    rights, strikes, dtes, deltas, bids, asks, mids = [], [], [], [], [], [], []
    date_offsets_list = [0]
    for d in _unique_dates_ordered:
        r, s, dte_a, da, b, a, m = _date_cache[d]
        rights.append(r);  strikes.append(s); dtes.append(dte_a)
        deltas.append(da); bids.append(b);    asks.append(a);    mids.append(m)
        date_offsets_list.append(date_offsets_list[-1] + len(r))

    # bar → date index mapping (-1 when bar's date has no chain data)
    bar_to_date_np = np.array(
        [_date_to_didx.get(d, -1) for d in bar_dates], dtype=np.int32
    )
    date_offsets_arr = np.array(date_offsets_list, dtype=np.int64)

    if rights:
        r_flat  = np.concatenate(rights)
        s_flat  = np.concatenate(strikes)
        d_flat  = np.concatenate(dtes)
        da_flat = np.concatenate(deltas)
        bi_flat = np.concatenate(bids)
        as_flat = np.concatenate(asks)
        mi_flat = np.concatenate(mids)
    else:
        r_flat  = np.empty(0, np.int8)
        s_flat  = np.empty(0, np.float32)
        d_flat  = np.empty(0, np.float32)
        da_flat = np.empty(0, np.float32)
        bi_flat = np.empty(0, np.float32)
        as_flat = np.empty(0, np.float32)
        mi_flat = np.empty(0, np.float32)

    dev = device
    chain_N = len(r_flat)
    bars_with_chain = int((bar_to_date_np >= 0).sum())
    print(f"[optimizer_prep] T={T} bars  D={D} dates  chain_N={chain_N:,}  "
          f"bars_with_chain={bars_with_chain:,}  device={dev}")

    # ── Verbose tensor summary ────────────────────────────────────────────
    print(f"[optimizer_prep] Tensor summary:")
    print(f"  timestamps      shape=({T},)      dtype=int64    {ts_arr.nbytes/1e6:.2f} MB")
    print(f"  spot            shape=({T},)      dtype=float32  {spot_arr.nbytes/1e6:.2f} MB")
    print(f"  gate_entry      shape=({T},)      dtype=float32  {entry_sig.nbytes/1e6:.2f} MB")
    print(f"  gate_pop        shape=({T},)      dtype=float32  {pop_sig.nbytes/1e6:.2f} MB")
    print(f"  strategy_idx    shape=({T},)      dtype=int16    {sidx_arr.nbytes/1e6:.2f} MB")
    print(f"  bar_to_date_idx shape=({T},)      dtype=int32    {bar_to_date_np.nbytes/1e6:.2f} MB")
    print(f"  date_offsets    shape=({D+1},)    dtype=int64    {date_offsets_arr.nbytes/1e6:.2f} MB")
    print(f"  option_right    shape=({chain_N},) dtype=int8    {r_flat.nbytes/1e6:.2f} MB")
    print(f"  option_strike   shape=({chain_N},) dtype=float32 {s_flat.nbytes/1e6:.2f} MB")
    print(f"  option_dte      shape=({chain_N},) dtype=float32 {d_flat.nbytes/1e6:.2f} MB")
    print(f"  option_delta    shape=({chain_N},) dtype=float32 {da_flat.nbytes/1e6:.2f} MB")
    print(f"  opt_bid/ask     shape=({chain_N},) dtype=float32 {bi_flat.nbytes/1e6:.2f} MB each")
    total_mb = (ts_arr.nbytes + spot_arr.nbytes + entry_sig.nbytes + pop_sig.nbytes
                + sidx_arr.nbytes + bar_to_date_np.nbytes + date_offsets_arr.nbytes
                + r_flat.nbytes + s_flat.nbytes + d_flat.nbytes + da_flat.nbytes
                + bi_flat.nbytes + as_flat.nbytes + mi_flat.nbytes) / 1e6
    print(f"  → Total context: {total_mb:.1f} MB on {dev}  "
          f"(date-indexed: {D} dates × avg {chain_N//max(D,1):,} opts/date)")
    if chain_N > 0:
        print(f"  → Calls: {(r_flat==0).sum():,}  Puts: {(r_flat==1).sum():,}")
        print(f"  → DTE range: [{d_flat.min():.1f}, {d_flat.max():.1f}]  "
              f"delta NaN frac: {np.isnan(da_flat).mean()*100:.1f}%")
        print(f"  → Strike range: [{s_flat.min():.2f}, {s_flat.max():.2f}]")
        print(f"  → Bid range: [{bi_flat[bi_flat>0].min() if (bi_flat>0).any() else 0:.2f}, {bi_flat.max():.2f}]")
    else:
        print("  [WARN] chain_N=0 — no options data reached the tensor! Check bar_dates vs chain_df_by_date keys.")

    return OptimizerContext(
        device          = dev,
        T               = T,
        timestamps      = torch.from_numpy(ts_arr).to(dev),
        spot            = torch.from_numpy(spot_arr).to(dev),
        gate_entry      = torch.from_numpy(entry_sig).to(dev),
        gate_pop        = torch.from_numpy(pop_sig).to(dev),
        strategy_idx    = torch.from_numpy(sidx_arr).to(dev),
        abstain         = torch.from_numpy(abt_arr.astype(np.uint8)).bool().to(dev),
        date_offsets    = torch.from_numpy(date_offsets_arr).to(dev),
        bar_to_date_idx = torch.from_numpy(bar_to_date_np).to(dev),
        option_right    = torch.from_numpy(r_flat).to(dev),
        option_strike   = torch.from_numpy(s_flat).to(dev),
        option_dte      = torch.from_numpy(d_flat).to(dev),
        option_delta    = torch.from_numpy(da_flat).to(dev),
        opt_bid         = torch.from_numpy(bi_flat).to(dev),
        opt_ask         = torch.from_numpy(as_flat).to(dev),
        opt_mid         = torch.from_numpy(mi_flat).to(dev),
        fast_end        = max(1, int(T * 0.25)),
        medium_end      = max(1, int(T * 0.60)),
        bar_dates       = bar_dates,
    )
