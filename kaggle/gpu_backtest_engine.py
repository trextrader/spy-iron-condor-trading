"""
gpu_backtest_engine.py — Multi-position GPU Backtest Simulation
===============================================================
Replicates run_backtest() shared-capital semantics on GPU:
  - N_pos concurrent position slots (vs K independent simulations in optimizer)
  - Shared scalar equity (vs K separate equity arrays)
  - One entry per gate-fire bar (model predicts one strategy_idx; global cooldown)
  - Per-slot family dispatch for MtM and intrinsic value

Financial equivalences vs run_backtest():
  - DOLLAR_STOP, PROFIT_TARGET, TIME_EXIT/EXPIRY, FRIDAY_CLOSEOUT → same thresholds
  - ExecutionRealityEngine slippage → not ported (no option symbol tracking on GPU)
  - DecisionTraceLogger → not ported (diagnostic only)

Results are financially equivalent but not bit-for-bit identical to run_backtest().
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

# ── Shared constants (mirror optimizer_engine.py) ────────────────────────────
STARTING_EQUITY        = 100_000.0
IC_MULTIPLIER          = 100
LEVERAGE_FACTOR        = 2.0
MAX_OPTIONS_ALLOC_PCT  = 0.40
MIN_BARS_COOLDOWN      = 5
ENTRY_THRESHOLD        = 0.55
POP_THRESHOLD          = 0.50

# ── Engine family codes (int8 stored per position slot) ──────────────────────
FC_SHORT_CALL      = 0
FC_SHORT_PUT       = 1
FC_IRON_BUTTERFLY  = 2
FC_IRON_CONDOR     = 3

# class_name (from strategy CONFIG) → engine family string
_CLASS_TO_ENGINE_FAMILY: Dict[str, str] = {
    'single_call':       'short_call',
    'single_put':        'short_put',
    'iron_butterfly':    'iron_butterfly',
    'custom_multi_leg':  'iron_butterfly',   # symmetric two-sided
    'straddle':          'iron_butterfly',
    'strangle':          'iron_condor',
    'iron_condor':       'iron_condor',
    'bull_call_spread':  'iron_condor',      # approximated as IC
    'bear_put_spread':   'iron_condor',      # approximated as IC
}

# engine family string → int8 code
_FAMILY_CODE: Dict[str, int] = {
    'short_call':      FC_SHORT_CALL,
    'short_put':       FC_SHORT_PUT,
    'iron_butterfly':  FC_IRON_BUTTERFLY,
    'iron_condor':     FC_IRON_CONDOR,
}

# int8 code → engine family string (for mark_to_market_gpu dispatch)
_CODE_TO_FAMILY: Dict[int, str] = {v: k for k, v in _FAMILY_CODE.items()}


def _get_engine_family(class_name: str) -> str:
    return _CLASS_TO_ENGINE_FAMILY.get(class_name, 'iron_butterfly')


def _resolve_template_gpu(sidx: int, strategy_configs: Dict) -> str:
    """Given strategy class index, return first matching template_id."""
    for tid, cfg in strategy_configs.items():
        if cfg.get('class_idx') == sidx:
            return tid
    return 'unknown'


def _intrinsic_debit_vec(
    spot_t: torch.Tensor,        # scalar float32
    ss_call_t: torch.Tensor,     # [N_pos] float32  short-call strike
    ss_put_t: torch.Tensor,      # [N_pos] float32  short-put strike
    width_t: torch.Tensor,       # [N_pos] float32  wing width
    family_code_t: torch.Tensor, # [N_pos] int8
    open_mask_t: torch.Tensor,   # [N_pos] bool
) -> torch.Tensor:               # [N_pos] float32
    """
    Intrinsic-value MtM debit for all open position slots.

    Computes the cost-to-close (debit) for each slot assuming spot = spot_t.
    Uses the same intrinsic formulas as optimizer_engine.py GPU path.

    Returns zeros for closed slots (open_mask_t == False).
    """
    sc = spot_t                    # scalar
    _ic  = torch.clamp(sc - ss_call_t, min=0.0)          # short call leg
    _ip  = torch.clamp(ss_put_t - sc,  min=0.0)          # short put leg
    _iwc = torch.clamp(sc - (ss_call_t + width_t), min=0.0)   # long call wing
    _iwp = torch.clamp((ss_put_t - width_t) - sc,  min=0.0)   # long put wing

    is_sc = (family_code_t == FC_SHORT_CALL)
    is_sp = (family_code_t == FC_SHORT_PUT)
    # IC / IB / spreads — 4-leg net debit
    _ic_net = _ic + _ip - _iwc - _iwp

    debit = torch.where(is_sc, _ic,
            torch.where(is_sp, _ip, _ic_net))

    return torch.where(open_mask_t, debit, torch.zeros_like(debit))


def _chain_mtm_debit(
    r_t:  torch.Tensor,   # [M] chain slice option_right
    s_t:  torch.Tensor,   # [M] chain slice option_strike
    d_t:  torch.Tensor,   # [M] chain slice option_dte
    b_t:  torch.Tensor,   # [M] chain slice opt_bid
    a_t:  torch.Tensor,   # [M] chain slice opt_ask
    open_mask_t:    torch.Tensor,  # [N_pos] bool
    family_code_t:  torch.Tensor,  # [N_pos] int8
    ss_call_t:      torch.Tensor,  # [N_pos] float64
    ss_put_t:       torch.Tensor,  # [N_pos] float64
    width_t:        torch.Tensor,  # [N_pos] float64
    dte_t:          torch.Tensor,  # [N_pos] float64  remaining DTE
    spot: float,
    dev: torch.device,
) -> torch.Tensor:                 # [N_pos] float32
    """
    Chain-based MtM debit for open slots. Calls mark_to_market_gpu once per
    unique family in open slots, then falls back to intrinsic for NaN results.
    """
    import gpu_strike_selector as _gss

    N_pos = len(open_mask_t)
    debit_t = torch.full((N_pos,), float('nan'), dtype=torch.float32, device=dev)

    # One mark_to_market_gpu call per unique family in open slots
    for fc_val in torch.unique(family_code_t[open_mask_t]).tolist():
        fc_int = int(fc_val)
        family_mask = open_mask_t & (family_code_t == fc_int)
        engine_family = _CODE_TO_FAMILY.get(fc_int, 'iron_butterfly')
        raw = _gss.mark_to_market_gpu(
            r_t, s_t, d_t, b_t, a_t,
            ss_call_t.float(), ss_put_t.float(), width_t.float(),
            family_mask, dte_t.float(), engine_family,
            return_tensors=True,
        )   # [N_pos] float32, NaN where not found or slot closed
        debit_t = torch.where(torch.isfinite(raw) & family_mask, raw, debit_t)

    # Intrinsic fallback for still-NaN open slots
    spot_t = torch.tensor(spot, dtype=torch.float32, device=dev)
    intr_t = _intrinsic_debit_vec(
        spot_t, ss_call_t.float(), ss_put_t.float(), width_t.float(),
        family_code_t, open_mask_t,
    )
    debit_t = torch.where(open_mask_t & ~torch.isfinite(debit_t), intr_t, debit_t)

    return debit_t


def run_backtest_gpu(
    ctx: "OptimizerContext",
    strategy_configs: Dict[str, dict],
    allowed_template_ids: Optional[Set[str]],
    device: torch.device,
    max_positions: int = 5,
    friday_closeout: bool = True,
    verbose: bool = False,
    limit: Optional[int] = None,
) -> Tuple[List[float], List[dict]]:
    """
    GPU multi-position backtest with shared capital.

    Replaces run_backtest() semantics on GPU:
      - Up to max_positions concurrent position slots sharing one equity pool
      - Entries driven by v43 model gates (same thresholds as run_backtest)
      - Exits: DOLLAR_STOP, PROFIT_TARGET, TIME/EXPIRY, FRIDAY_CLOSEOUT
      - Template selection: resolves strategy_idx → template_id → engine_family

    Parameters
    ----------
    ctx                 : OptimizerContext from build_optimizer_context().
    strategy_configs    : _STRATEGY_CONFIGS dict loaded from kaggle/strategies/.
    allowed_template_ids: None = all templates; set = only these templates eligible.
    device              : torch.device (should be 'cuda').
    max_positions       : max concurrent open position slots (default 5).
    friday_closeout     : force-close all positions on Friday ≥15h (default True).
    verbose             : print entry/exit events.

    Returns
    -------
    equity_curve  : List[float]  one equity value per M5 bar (realised equity).
    trade_events  : List[dict]   OPEN and CLOSE event dicts for reporting.
    """
    import gpu_strike_selector as _gss

    dev = device
    N_pos = max_positions
    T     = ctx.T if not limit else min(ctx.T, limit)

    # ── Pull bar-level arrays to CPU numpy (small; used in Python gate checks) ──
    spot_np        = ctx.spot.cpu().numpy()
    gate_e_np      = ctx.gate_entry.cpu().numpy()
    gate_p_np      = ctx.gate_pop.cpu().numpy()
    sidx_np        = ctx.strategy_idx.cpu().numpy()
    abstain_np     = ctx.abstain.cpu().numpy().astype(bool)
    bar_offsets_np = ctx.bar_offsets.cpu().numpy()
    ts_np          = ctx.timestamps.cpu().numpy()   # int64 unix-seconds

    # ── GPU state tensors ─────────────────────────────────────────────────────
    equity_t        = torch.tensor(STARTING_EQUITY, dtype=torch.float64, device=dev)
    peak_t          = torch.tensor(STARTING_EQUITY, dtype=torch.float64, device=dev)
    max_dd_t        = torch.tensor(0.0,             dtype=torch.float64, device=dev)

    open_mask_t     = torch.zeros(N_pos, dtype=torch.bool,    device=dev)
    family_code_t   = torch.zeros(N_pos, dtype=torch.int8,    device=dev)
    entry_credit_t  = torch.zeros(N_pos, dtype=torch.float64, device=dev)
    entry_ss_call_t = torch.zeros(N_pos, dtype=torch.float64, device=dev)
    entry_ss_put_t  = torch.zeros(N_pos, dtype=torch.float64, device=dev)
    entry_width_t   = torch.zeros(N_pos, dtype=torch.float64, device=dev)
    entry_dte_t     = torch.zeros(N_pos, dtype=torch.float64, device=dev)
    entry_bar_t     = torch.full((N_pos,), -999, dtype=torch.int64, device=dev)
    # Per-slot exit thresholds (written at entry from template config)
    stop_loss_t     = torch.full((N_pos,), 600.0,  dtype=torch.float64, device=dev)
    profit_tgt_t    = torch.full((N_pos,), 1500.0, dtype=torch.float64, device=dev)
    hold_days_t     = torch.full((N_pos,), 30.0,   dtype=torch.float64, device=dev)
    max_dte_exit_t  = torch.zeros(N_pos,            dtype=torch.float64, device=dev)

    # ── Output accumulators ──────────────────────────────────────────────────
    equity_curve:  List[float] = []
    trade_events:  List[dict]  = []
    last_entry_bar = -999

    print(f"[gpu_backtest] T={T}  N_pos={max_positions}  device={dev}  "
          f"templates={len(strategy_configs)}"
          + (f"  filter={len(allowed_template_ids)}" if allowed_template_ids else "  filter=ALL"))

    # ── Bar loop ──────────────────────────────────────────────────────────────
    for i in range(T):
        spot   = float(spot_np[i])
        ts_i   = int(ts_np[i])          # unix seconds
        es     = float(gate_e_np[i])
        pop    = float(gate_p_np[i])
        sidx   = int(sidx_np[i])
        abt    = bool(abstain_np[i])

        # Chain slice for this bar
        s_off    = int(bar_offsets_np[i])
        e_off    = int(bar_offsets_np[i + 1])
        has_chain = (s_off < e_off)
        if has_chain:
            _r_t  = ctx.option_right[s_off:e_off]
            _s_t  = ctx.option_strike[s_off:e_off]
            _d_t  = ctx.option_dte[s_off:e_off]
            _da_t = ctx.option_delta[s_off:e_off]
            _b_t  = ctx.opt_bid[s_off:e_off]
            _a_t  = ctx.opt_ask[s_off:e_off]

        # ── A. Friday closeout — force close all before weekend ────────────
        _dow  = int((ts_i // 86400 + 4) % 7)  # 0=Mon..4=Fri  (unix epoch was Thu)
        _hour = int((ts_i % 86400) // 3600)
        if friday_closeout and _dow == 4 and _hour >= 15 and bool(open_mask_t.any()):
            spot_t32 = torch.tensor(spot, dtype=torch.float32, device=dev)
            intr_t   = _intrinsic_debit_vec(
                spot_t32, entry_ss_call_t.float(), entry_ss_put_t.float(),
                entry_width_t.float(), family_code_t, open_mask_t,
            )
            _pnl_fc_t = torch.where(
                open_mask_t,
                (entry_credit_t.float() - intr_t).double() * IC_MULTIPLIER,
                torch.zeros(N_pos, dtype=torch.float64, device=dev),
            )
            equity_t   = equity_t + _pnl_fc_t.sum()
            peak_t     = torch.maximum(peak_t, equity_t)
            for j in range(N_pos):
                if bool(open_mask_t[j].item()):
                    _p = float(_pnl_fc_t[j].item())
                    trade_events.append({'action': 'CLOSE', 'idx': i,
                                         'pnl': _p, 'pnl_pct': _p / STARTING_EQUITY * 100,
                                         'spot': spot, 'reason': 'FRIDAY_CLOSEOUT'})
                    if verbose:
                        print(f"  [CLOSE:FRIDAY_CLOSEOUT] slot={j} bar={i} pnl={_p:+,.0f}")
            open_mask_t = torch.zeros(N_pos, dtype=torch.bool, device=dev)

        # ── B. Process exits for open positions ────────────────────────────
        if bool(open_mask_t.any()):
            # MtM debit: chain where available, intrinsic fallback
            if has_chain:
                _debit_t = _chain_mtm_debit(
                    _r_t, _s_t, _d_t, _b_t, _a_t,
                    open_mask_t, family_code_t,
                    entry_ss_call_t, entry_ss_put_t, entry_width_t, entry_dte_t,
                    spot, dev,
                )
            else:
                spot_t32 = torch.tensor(spot, dtype=torch.float32, device=dev)
                _debit_t = _intrinsic_debit_vec(
                    spot_t32, entry_ss_call_t.float(), entry_ss_put_t.float(),
                    entry_width_t.float(), family_code_t, open_mask_t,
                )

            # Unrealized P&L per slot
            _unreal_t = torch.where(
                open_mask_t,
                (entry_credit_t.float() - _debit_t).double() * IC_MULTIPLIER,
                torch.zeros(N_pos, dtype=torch.float64, device=dev),
            )

            # Mark-to-market drawdown update
            _eq_mark  = equity_t + _unreal_t.sum()
            peak_t    = torch.maximum(peak_t, _eq_mark)
            _dd_now   = torch.where(
                peak_t > 0,
                (peak_t - _eq_mark) / peak_t * 100.0,
                torch.tensor(0.0, device=dev),
            )
            max_dd_t  = torch.maximum(max_dd_t, _dd_now)

            # Days held and DTE remaining per slot
            _eb_idx    = entry_bar_t.clamp(min=0).cpu().numpy()
            _eb_ts_arr = ts_np[_eb_idx]
            _days_held = (ts_i - torch.tensor(_eb_ts_arr, dtype=torch.float64, device=dev)) / 86400.0
            _days_held = torch.where(open_mask_t, _days_held,
                                     torch.zeros(N_pos, dtype=torch.float64, device=dev))
            _dte_rem   = torch.where(open_mask_t, entry_dte_t - _days_held,
                                     torch.zeros_like(_days_held))

            # Exit condition masks
            _sl_hit   = open_mask_t & (_unreal_t <= -stop_loss_t.abs())
            _pt_hit   = open_mask_t & (_unreal_t >= profit_tgt_t) & (_unreal_t > 0)
            _exp_ex   = open_mask_t & (_dte_rem <= 0)
            _hd_ex    = open_mask_t & (_days_held >= hold_days_t)
            _dte_ex   = open_mask_t & (max_dte_exit_t > 0) & (_dte_rem <= max_dte_exit_t)
            _exit_t   = _sl_hit | _pt_hit | _exp_ex | _hd_ex | _dte_ex

            if bool(_exit_t.any()):
                _pnl_t   = torch.where(_exit_t, _unreal_t,
                                       torch.zeros(N_pos, dtype=torch.float64, device=dev))
                equity_t = equity_t + _pnl_t.sum()
                peak_t   = torch.maximum(peak_t, equity_t)
                for j in range(N_pos):
                    if bool(_exit_t[j].item()):
                        _p = float(_pnl_t[j].item())
                        _reason = ('DOLLAR_STOP'    if bool(_sl_hit[j].item()) else
                                   'PROFIT_TARGET'  if bool(_pt_hit[j].item()) else
                                   'EXPIRY'         if bool(_exp_ex[j].item()) else
                                   'MAX_HOLD'       if bool(_hd_ex[j].item()) else 'DTE_EXIT')
                        trade_events.append({'action': 'CLOSE', 'idx': i,
                                             'pnl': _p,
                                             'pnl_pct': _p / STARTING_EQUITY * 100,
                                             'spot': spot, 'reason': _reason})
                        if verbose:
                            print(f"  [CLOSE:{_reason}] slot={j} bar={i} "
                                  f"pnl={_p:+,.0f} equity={float(equity_t):,.0f}")
                open_mask_t = open_mask_t & ~_exit_t

        # Record realised equity after exits
        equity_curve.append(float(equity_t.item()))

        # ── C. Entry gate ────────────────────────────────────────────────
        if not has_chain:
            continue
        gate_ok = (es > ENTRY_THRESHOLD) and (pop > POP_THRESHOLD) and (not abt)
        if not gate_ok:
            continue
        if (i - last_entry_bar) < MIN_BARS_COOLDOWN:
            continue

        # Resolve template and check filter
        tmpl_id = _resolve_template_gpu(sidx, strategy_configs)
        if allowed_template_ids is not None and tmpl_id not in allowed_template_ids:
            continue
        cfg = strategy_configs.get(tmpl_id, {})
        if not cfg:
            continue
        engine_family = _get_engine_family(cfg.get('class_name', 'iron_butterfly'))

        # Find empty position slot
        _empty = (~open_mask_t).nonzero(as_tuple=False)
        if len(_empty) == 0:
            continue
        j = int(_empty[0, 0].item())

        # Capital check (40% of equity × 2x leverage = MAX_DEPLOY)
        _max_deploy = float(equity_t.item()) * LEVERAGE_FACTOR * MAX_OPTIONS_ALLOC_PCT
        _is_naked   = engine_family in ('short_call', 'short_put')
        _est_width  = float(cfg.get('spread_width', 5.0) or 5.0)
        _est_margin = (spot * 0.02 * IC_MULTIPLIER if _is_naked
                       else _est_width * IC_MULTIPLIER)
        if _est_margin > _max_deploy:
            continue

        # Structure selection (K=1 call to select_entry_for_bar)
        _td_t = torch.tensor([float(cfg.get('target_dte',    21.0) or 21.0)],
                             dtype=torch.float32, device=dev)
        _sd_t = torch.tensor([float(cfg.get('short_delta',   0.20) or 0.20)],
                             dtype=torch.float32, device=dev)
        _sw_t = torch.tensor([float(cfg.get('spread_width',  5.0)  or 5.0)],
                             dtype=torch.float32, device=dev)
        _g = _gss.select_entry_for_bar(
            _r_t, _s_t, _d_t, _da_t, _b_t, _a_t, spot,
            _td_t, _sd_t, _sw_t, engine_family,
            return_tensors=True,
        )
        if not bool(_g['valid'][0].item()):
            continue

        # Write entry to slot j
        open_mask_t[j]     = True
        entry_credit_t[j]  = float(_g['credit'][0].item())
        entry_ss_call_t[j] = float(_g['ss_call'][0].item())
        entry_ss_put_t[j]  = float(_g['ss_put'][0].item())
        entry_width_t[j]   = float(_g['actual_width'][0].item())
        entry_dte_t[j]     = float(_g['actual_dte'][0].item())
        entry_bar_t[j]     = i
        family_code_t[j]   = _FAMILY_CODE.get(engine_family, FC_IRON_BUTTERFLY)
        stop_loss_t[j]     = float(cfg.get('stop_loss_dollar', 600.0) or 600.0)
        profit_tgt_t[j]    = float(cfg.get('profit_target',  1500.0) or 1500.0)
        hold_days_t[j]     = float(cfg.get('hold_days',         30.0) or 30.0)
        max_dte_exit_t[j]  = float(cfg.get('max_dte_exit',       0.0) or 0.0)
        last_entry_bar     = i

        _cred = float(_g['credit'][0].item())
        trade_events.append({
            'action': 'OPEN', 'idx': i, 'template_id': tmpl_id,
            'spot': spot, 'credit': _cred,
            'ss_call': float(_g['ss_call'][0].item()),
            'ss_put':  float(_g['ss_put'][0].item()),
            'dte':     float(_g['actual_dte'][0].item()),
            'pnl_pct': 0.0,   # placeholder for reporting compat
        })
        if verbose:
            print(f"  [OPEN] slot={j} bar={i} tmpl={tmpl_id} family={engine_family} "
                  f"credit={_cred:.4f} spot={spot:.2f}")

    # ── Post-loop: force-close remaining open positions ────────────────────
    if bool(open_mask_t.any()):
        spot_t32 = torch.tensor(float(spot_np[-1]), dtype=torch.float32, device=dev)
        intr_t   = _intrinsic_debit_vec(
            spot_t32, entry_ss_call_t.float(), entry_ss_put_t.float(),
            entry_width_t.float(), family_code_t, open_mask_t,
        )
        _fc_pnl_t = torch.where(
            open_mask_t,
            (entry_credit_t.float() - intr_t).double() * IC_MULTIPLIER,
            torch.zeros(N_pos, dtype=torch.float64, device=dev),
        )
        equity_t = equity_t + _fc_pnl_t.sum()
        for j in range(N_pos):
            if bool(open_mask_t[j].item()):
                _p = float(_fc_pnl_t[j].item())
                trade_events.append({'action': 'CLOSE', 'idx': T - 1,
                                     'pnl': _p, 'pnl_pct': _p / STARTING_EQUITY * 100,
                                     'spot': float(spot_np[-1]),
                                     'reason': 'END_OF_SIM'})

    net_pnl = float(equity_t.item()) - STARTING_EQUITY
    n_closes = sum(1 for e in trade_events if e['action'] == 'CLOSE')
    print(f"[gpu_backtest] Done  T={T}  trades={n_closes}  "
          f"net={net_pnl:+,.0f}  ({net_pnl/STARTING_EQUITY*100:+.2f}%)  "
          f"max_dd={float(max_dd_t.item()):.2f}%")

    return equity_curve, trade_events
