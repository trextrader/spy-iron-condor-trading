"""
optimizer_prep.py — Immutable Tensor Context Builder
====================================================
Pre-processes the v45 bundle into flat CSR tensors for the batch optimizer.
Build once; reuse across all BO rounds.

CSR Layout (chain per M5 bar):
    bar_offsets[T+1]  — start/end index into flat arrays for each M5 bar
    option_right[N]   — 0=call  1=put
    option_strike[N]  — float32 strike price
    option_dte[N]     — float32 days-to-expiry (te column)
    option_delta[N]   — float32 abs(delta)  (NaN when unavailable)
    opt_bid[N]        — float32
    opt_ask[N]        — float32
    opt_mid[N]        — float32
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

    # ── Chain CSR ─────────────────────────────────────────────────────────
    bar_offsets: torch.Tensor      # [T+1] int64
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
    if isinstance(m5.index, pd.DatetimeIndex):
        ts_series = m5.index
    elif 'timestamp' in m5.columns:
        ts_series = pd.to_datetime(m5['timestamp'])
    else:
        ts_series = pd.to_datetime(m5.index)
    ts_int = ts_series.astype(np.int64) // 10 ** 9  # unix seconds

    bar_dates = [str(t.date()) for t in ts_series]

    # ── Spot ──────────────────────────────────────────────────────────────
    close_col = 'close' if 'close' in m5.columns else m5.columns[0]
    spot_arr  = m5[close_col].astype(np.float32).values

    # ── V43 gate signals (clip to T in case of slight mismatch) ──────────
    def _to_arr(key, dtype):
        a = np.asarray(v43_outputs[key], dtype=dtype)
        return a[:T] if len(a) >= T else np.pad(a, (0, T - len(a)))

    entry_sig = _to_arr('entry_signal', np.float32)
    pop_sig   = _to_arr('pop',          np.float32)
    sidx_arr  = _to_arr('strategy_idx', np.int16)
    abt_arr   = _to_arr('abstain',      bool)

    # ── CSR chain build ───────────────────────────────────────────────────
    rights, strikes, dtes, deltas, bids, asks, mids = [], [], [], [], [], [], []
    offsets = [0]

    for date_str in bar_dates:
        chain = chain_df_by_date.get(date_str)
        if chain is None or len(chain) == 0:
            offsets.append(offsets[-1])
            continue

        n   = len(chain)
        cp  = chain.get('call_put', chain.get('type', pd.Series(['C'] * n, index=chain.index)))
        right = (cp.astype(str).str.upper().str.startswith('P')).astype(np.int8).values
        strike = pd.to_numeric(chain['strike'], errors='coerce').fillna(0).astype(np.float32).values

        dte_col = 'te' if 'te' in chain.columns else ('dte' if 'dte' in chain.columns else None)
        dte = (pd.to_numeric(chain[dte_col], errors='coerce').fillna(1.0).astype(np.float32).values
               if dte_col else np.ones(n, np.float32))

        delta_col = 'delta' if 'delta' in chain.columns else None
        delta = (pd.to_numeric(chain[delta_col], errors='coerce').astype(np.float32).values
                 if delta_col else np.full(n, np.nan, np.float32))

        bid = pd.to_numeric(chain.get('bid', pd.Series([0.0]*n)), errors='coerce').fillna(0.0).astype(np.float32).values
        ask = pd.to_numeric(chain.get('ask', pd.Series([0.0]*n)), errors='coerce').fillna(0.0).astype(np.float32).values
        mid_raw = chain.get('mid')
        if mid_raw is not None:
            mid = pd.to_numeric(mid_raw, errors='coerce').fillna((bid+ask)/2).astype(np.float32).values
        else:
            mid = ((bid + ask) / 2).astype(np.float32)

        rights.append(right);  strikes.append(strike); dtes.append(dte)
        deltas.append(delta);  bids.append(bid);       asks.append(ask); mids.append(mid)
        offsets.append(offsets[-1] + n)

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

    offsets_arr = np.array(offsets, dtype=np.int64)
    dev = device

    print(f"[optimizer_prep] T={T} bars  chain_N={len(r_flat):,}  device={dev}")

    return OptimizerContext(
        device        = dev,
        T             = T,
        timestamps    = torch.from_numpy(ts_int.values.astype(np.int64)).to(dev),
        spot          = torch.from_numpy(spot_arr).to(dev),
        gate_entry    = torch.from_numpy(entry_sig).to(dev),
        gate_pop      = torch.from_numpy(pop_sig).to(dev),
        strategy_idx  = torch.from_numpy(sidx_arr).to(dev),
        abstain       = torch.from_numpy(abt_arr.astype(np.uint8)).bool().to(dev),
        bar_offsets   = torch.from_numpy(offsets_arr).to(dev),
        option_right  = torch.from_numpy(r_flat).to(dev),
        option_strike = torch.from_numpy(s_flat).to(dev),
        option_dte    = torch.from_numpy(d_flat).to(dev),
        option_delta  = torch.from_numpy(da_flat).to(dev),
        opt_bid       = torch.from_numpy(bi_flat).to(dev),
        opt_ask       = torch.from_numpy(as_flat).to(dev),
        opt_mid       = torch.from_numpy(mi_flat).to(dev),
        fast_end      = max(1, int(T * 0.25)),
        medium_end    = max(1, int(T * 0.60)),
        bar_dates     = bar_dates,
    )
