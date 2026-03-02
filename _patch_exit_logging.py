"""
_patch_exit_logging.py  (v2 — correct indentation)
Rewrites all 5 exit verbose blocks + HOLD block in condor_brain_backtest_v45.py:
  - Removes `if verbose_sim:` guards from exit prints → always visible
  - Enriches exit prints: early-exit flag, premium_captured %, DTE left
  - Makes HOLD periodic (every HOLD_PRINT_INTERVAL bars) with hold reasoning
  - Adds exit signal + spot-to-strike distances to HOLD display
Run: py -3 _patch_exit_logging.py
"""
import shutil, sys

TARGET = 'kaggle/condor_brain_backtest_v45.py'
shutil.copy(TARGET, TARGET + '.prebak_exitlog')

with open(TARGET, 'r', encoding='utf-8') as f:
    lines = f.readlines()

applied = {}

def replace_block(lines, start_anchor, end_anchor, new_lines_str, label):
    """Replace the block from the line containing start_anchor up to and
    including the line containing end_anchor with new_lines_str."""
    start_idx = None
    end_idx   = None
    for i, l in enumerate(lines):
        if start_anchor in l and start_idx is None:
            start_idx = i
        if start_idx is not None and end_anchor in l:
            end_idx = i
            break
    if start_idx is None or end_idx is None:
        print(f"  {label}: FAILED (start={start_idx}, end={end_idx})")
        applied[label] = False
        return lines
    new = [ln + '\n' for ln in new_lines_str.split('\n')]
    # strip trailing empty line from split artefact
    if new and new[-1] == '\n':
        new = new[:-1]
    lines = lines[:start_idx] + new + lines[end_idx + 1:]
    print(f"  {label}: applied (lines {start_idx+1}–{end_idx+1})")
    applied[label] = True
    return lines

# ── Each block: (unique start anchor, unique end anchor, replacement text) ────

# ─── PER_TRADE_STOP (17sp / 21sp) ────────────────────────────────────────────
P, P4 = ' '*17, ' '*21
BLOCK_PTS_OLD_START = '_trace_close(tr, realized_pnl, "PER_TRADE_STOP")'
BLOCK_PTS_OLD_END   = 'held={_dh:.1f}d  equity_after=${equity:,.2f}")\n                 continue'
# We'll match by unique anchor pair
BLOCK_PTS_NEW = (
P  + '_trace_close(tr, realized_pnl, "PER_TRADE_STOP")\n' +
P  + '_dh = (pd.Timestamp(ts) - pd.Timestamp(tr.entry_dt)).total_seconds() / 86400\n' +
P  + '_max_credit = tr.net_credit * tr.qty * IC_MULTIPLIER\n' +
P  + '_prem_cap   = (realized_pnl / _max_credit * 100) if _max_credit != 0 else 0.0\n' +
P  + 'print(f"\\n  \u274c CLOSED [PER_TRADE_STOP] [{tr.trade_id}]  bar={i}  {str(ts)[:19]}"\n' +
P  + '      f"\\n     EARLY EXIT \u2014 stopped out at {_dh:.1f}d held / {tr.dte_entry:.0f}d DTE"\n' +
P  + '      f"  ({max(0.0, tr.dte_entry - _dh):.1f}d left on contract)"\n' +
P  + '      f"\\n     credit=${tr.net_credit:.4f}/shr  stop_thresh=${_per_trade_stop:,.2f}"\n' +
P  + '      f"  unrealized=${tr.unrealized_pnl:,.2f}"\n' +
P  + '      f"\\n     realized=${realized_pnl:,.2f}  premium_captured={_prem_cap:+.1f}%"\n' +
P  + '      f"  fill_valid={exit_valid}  equity_after=${equity:,.2f}")\n' +
P  + 'continue'
).rstrip('\n')

# ─── RISK_STOP (17sp / 21sp) ─────────────────────────────────────────────────
BLOCK_RSK_NEW = (
P  + '_trace_close(tr, realized_pnl, "RISK_STOP")\n' +
P  + '_dh = (pd.Timestamp(ts) - pd.Timestamp(tr.entry_dt)).total_seconds() / 86400\n' +
P  + '_max_credit2 = tr.net_credit * tr.qty * IC_MULTIPLIER\n' +
P  + '_prem_cap2   = (realized_pnl / _max_credit2 * 100) if _max_credit2 != 0 else 0.0\n' +
P  + 'print(f"\\n  \U0001f6d1 CLOSED [RISK_STOP_5PCT] [{tr.trade_id}]  bar={i}  {str(ts)[:19]}"\n' +
P  + '      f"\\n     EARLY EXIT \u2014 portfolio 5%% equity stop at {_dh:.1f}d held / {tr.dte_entry:.0f}d DTE"\n' +
P  + '      f"  ({max(0.0, tr.dte_entry - _dh):.1f}d left on contract)"\n' +
P  + '      f"\\n     unrealized=${tr.unrealized_pnl:,.2f}  fill_valid={exit_valid}"\n' +
P  + '      f"\\n     realized=${realized_pnl:,.2f}  premium_captured={_prem_cap2:+.1f}%"\n' +
P  + '      f"  equity_after=${equity:,.2f}")\n' +
P  + 'continue  # Trade is gone'
).rstrip('\n')

# ─── PROFIT_50PCT (21sp / 25sp) ──────────────────────────────────────────────
P2, P24 = ' '*21, ' '*25
BLOCK_PRF_NEW = (
P2 + '_trace_close(tr, realized_pnl, "PROFIT_50PCT")\n' +
P2 + '_dh = (pd.Timestamp(ts) - pd.Timestamp(tr.entry_dt)).total_seconds() / 86400\n' +
P2 + '_max_credit3 = tr.net_credit * tr.qty * IC_MULTIPLIER\n' +
P2 + '_prem_cap3   = (realized_pnl / _max_credit3 * 100) if _max_credit3 != 0 else 0.0\n' +
P2 + 'print(f"\\n  \u2705 CLOSED [PROFIT_50PCT] [{tr.trade_id}]  bar={i}  {str(ts)[:19]}"\n' +
P2 + '      f"\\n     EARLY EXIT \u2014 took 50%% profit at {_dh:.1f}d held / {tr.dte_entry:.0f}d DTE"\n' +
P2 + '      f"  ({max(0.0, tr.dte_entry - _dh):.1f}d left on contract)"\n' +
P2 + '      f"\\n     credit=${tr.net_credit:.4f}/shr  target=${profit_target:,.2f}"\n' +
P2 + '      f"  unrealized=${tr.unrealized_pnl:,.2f}"\n' +
P2 + '      f"\\n     realized=${realized_pnl:,.2f}  premium_captured={_prem_cap3:+.1f}%"\n' +
P2 + '      f"  fill_valid={exit_valid}  equity_after=${equity:,.2f}")\n' +
P2 + 'continue'
).rstrip('\n')

# ─── EXPIRED (17sp / 21sp) ───────────────────────────────────────────────────
BLOCK_EXP_NEW = (
P  + '_trace_close(tr, realized_pnl, "EXPIRED")\n' +
P  + '_max_credit4 = tr.net_credit * tr.qty * IC_MULTIPLIER\n' +
P  + '_prem_cap4   = (realized_pnl / _max_credit4 * 100) if _max_credit4 != 0 else 0.0\n' +
P  + 'print(f"\\n  \u23f0 CLOSED [EXPIRED] [{tr.trade_id}]  bar={i}  {str(ts)[:19]}"\n' +
P  + '      f"\\n     FULL DTE \u2014 held to expiry {days_held:.1f}d / {tr.dte_entry:.0f}d DTE (0d left)"\n' +
P  + '      f"\\n     credit=${tr.net_credit:.4f}/shr  dte_entry={tr.dte_entry:.1f}d"\n' +
P  + '      f"  days_held={days_held:.1f}"\n' +
P  + '      f"\\n     realized=${realized_pnl:,.2f}  premium_captured={_prem_cap4:+.1f}%"\n' +
P  + '      f"  fill_valid={exit_valid}  equity_after=${equity:,.2f}")\n' +
P  + 'continue'
).rstrip('\n')

# ─── HOLD (13sp / 17sp) ──────────────────────────────────────────────────────
P3, P34 = ' '*13, ' '*17
BLOCK_HLD_NEW = (
P3 + 'active_trades.append(tr)\n' +
P3 + '# Periodic HOLD status (every HOLD_PRINT_INTERVAL bars per position)\n' +
P3 + 'if not hasattr(tr, \'_hold_bar_cnt\'): tr._hold_bar_cnt = 0\n' +
P3 + 'tr._hold_bar_cnt += 1\n' +
P3 + 'if verbose_sim and (tr._hold_bar_cnt == 1 or tr._hold_bar_cnt % HOLD_PRINT_INTERVAL == 0):\n' +
P34 + '_dh       = (pd.Timestamp(ts) - pd.Timestamp(tr.entry_dt)).total_seconds() / 86400\n' +
P34 + '_dte_rem  = max(0.0, tr.dte_entry - _dh)\n' +
P34 + '_mc_hold  = tr.net_credit * tr.qty * IC_MULTIPLIER\n' +
P34 + '_prem_pct = (_mc_hold and tr.unrealized_pnl / _mc_hold * 100) or 0.0\n' +
P34 + '_stop_thr = -(tr.net_credit * IC_STOP_LOSS_MULT * tr.qty * IC_MULTIPLIER)\n' +
P34 + '_stop_gap = tr.unrealized_pnl - _stop_thr   # positive = room before stop\n' +
P34 + '_tgt_pnl  = _mc_hold * 0.50\n' +
P34 + '_to_tgt   = _tgt_pnl - tr.unrealized_pnl    # positive = room to target\n' +
P34 + '_sc_dist  = ((tr.legs.get(\'short_call\', spot) - spot) / spot * 100\n' +
P34 + '             if tr.legs else 0.0)\n' +
P34 + '_sp_dist  = ((spot - tr.legs.get(\'short_put\', spot)) / spot * 100\n' +
P34 + '             if tr.legs else 0.0)\n' +
P34 + '_xs_str   = ""\n' +
P34 + 'if v43_outputs is not None and i < len(v43_outputs[\'exit_signal\']):\n' +
P34 + '    _xs = float(v43_outputs[\'exit_signal\'][i])\n' +
P34 + '    _xs_str = f"  exit_sig={_xs:.3f}"\n' +
P34 + '_hold_reasons = []\n' +
P34 + 'if _prem_pct < 50.0:  _hold_reasons.append(f"prem_only {_prem_pct:.1f}% (<50%)")\n' +
P34 + 'if _dte_rem > 0:      _hold_reasons.append(f"{_dte_rem:.1f}d DTE remaining")\n' +
P34 + 'if _stop_gap > 0:     _hold_reasons.append(f"${_stop_gap:,.0f} from stop")\n' +
P34 + 'if _to_tgt > 0:       _hold_reasons.append(f"${_to_tgt:,.0f} to profit target")\n' +
P34 + '_hold_why = " | ".join(_hold_reasons) if _hold_reasons else "holding"\n' +
P34 + 'print(f"  [HOLD] [{tr.trade_id}] bar={i} {str(ts)[:19]}"\n' +
P34 + '      f"  held={_dh:.1f}d/{tr.dte_entry:.0f}dte  dte_rem={_dte_rem:.1f}d\\n"\n' +
P34 + '      f"    unrealized=${tr.unrealized_pnl:+,.2f}  prem_captured={_prem_pct:+.1f}%"\n' +
P34 + '      f"  spot={spot:.2f}\\n"\n' +
P34 + '      f"    SC_dist={_sc_dist:+.2f}%  SP_dist={_sp_dist:+.2f}%"\n' +
P34 + '      f"  stop_gap=${_stop_gap:,.0f}  to_target=${_to_tgt:,.0f}{_xs_str}\\n"\n' +
P34 + '      f"    WHY HOLDING: {_hold_why}")'
).rstrip('\n')

# ── Now apply each block by searching for unique anchors ─────────────────────
# Use line-number based replacement: find start anchor, scan for end anchor

def patch_block(lines, start_anchor, end_anchor, new_block_str, label):
    """Find lines containing start_anchor and then end_anchor, replace all."""
    si = None
    for i, l in enumerate(lines):
        if start_anchor in l and si is None:
            si = i
        if si is not None and end_anchor in l:
            ei = i
            new = new_block_str.split('\n')
            result = lines[:si] + [ln + '\n' for ln in new] + lines[ei + 1:]
            print(f"  {label}: OK (lines {si+1}–{ei+1})")
            applied[label] = True
            return result
    print(f"  {label}: FAILED — start_anchor={repr(start_anchor)[:50]}")
    applied[label] = False
    return lines

# PER_TRADE_STOP: unique anchor = _trace_close...PER_TRADE_STOP; end = 17sp continue AFTER it
lines = patch_block(lines,
    start_anchor='_trace_close(tr, realized_pnl, "PER_TRADE_STOP")',
    end_anchor='held={_dh:.1f}d  equity_after=${equity:,.2f}")\n',
    new_block_str=BLOCK_PTS_NEW,
    label='PER_TRADE_STOP')

# RISK_STOP
lines = patch_block(lines,
    start_anchor='_trace_close(tr, realized_pnl, "RISK_STOP")',
    end_anchor='continue  # Trade is gone',
    new_block_str=BLOCK_RSK_NEW,
    label='RISK_STOP')

# PROFIT_50PCT
lines = patch_block(lines,
    start_anchor='_trace_close(tr, realized_pnl, "PROFIT_50PCT")',
    end_anchor='held={_dh:.1f}d  equity_after=${equity:,.2f}")\n',
    new_block_str=BLOCK_PRF_NEW,
    label='PROFIT_50PCT')

# EXPIRED
lines = patch_block(lines,
    start_anchor='_trace_close(tr, realized_pnl, "EXPIRED")',
    end_anchor='equity_after=${equity:,.2f}")\n',
    new_block_str=BLOCK_EXP_NEW,
    label='EXPIRED')

# HOLD
lines = patch_block(lines,
    start_anchor='active_trades.append(tr)',
    end_anchor='f"  spot={spot:.2f}")',
    new_block_str=BLOCK_HLD_NEW,
    label='HOLD')

with open(TARGET, 'w', encoding='utf-8') as f:
    f.writelines(lines)

ok = sum(1 for v in applied.values() if v)
print(f"\n{ok}/{len(applied)} patches applied.")
