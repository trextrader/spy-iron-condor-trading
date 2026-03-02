#!/usr/bin/env python3
"""
patch_sim_guards.py  (v3)
Applies remaining simulation guard patches to condor_brain_backtest_v45.py.
--max-positions arg + run_backtest signature already applied directly.
This script handles:
  1. Expiry gate after dte computation
  2. Max positions gate in entry logic
  3. Force-close open trades at sim end
  4. Gate summary additions

Run once on Lightning AI:
    python patch_sim_guards.py
"""
import shutil, sys, os

TARGET = "kaggle/condor_brain_backtest_v45.py"
if not os.path.exists(TARGET):
    print(f"ERROR: {TARGET} not found. Run from repo root.")
    sys.exit(1)

shutil.copy(TARGET, TARGET + ".bak3")
with open(TARGET, "r", encoding="utf-8") as f:
    lines = f.readlines()

applied = []

# ── PATCH 1: Expiry gate after dte computation ────────────────────────────────
for idx, line in enumerate(lines):
    if "dte = float(np.clip(dte_raw" in line:
        if not any("expiry_past_sim" in lines[idx + j] for j in range(1, 6)):
            ind = len(line) - len(line.lstrip())
            p = " " * ind
            block = (
                f"{p}# Gate: skip if position would expire after sim window\n"
                f"{p}_expiry_dt = pd.Timestamp(ts) + pd.Timedelta(days=dte)\n"
                f"{p}if _expiry_dt > sim_end_dt:\n"
                f"{p}    run_backtest._entry_dbg['expiry_past_sim'] = run_backtest._entry_dbg.get('expiry_past_sim', 0) + 1\n"
                f"{p}    if verbose_sim:\n"
                f"{p}        print(f'  -- SKIP expiry_past_sim: expiry {{_expiry_dt.date()}} > sim_end {{sim_end_dt.date()}}')\n"
                f"{p}    continue\n"
            )
            lines.insert(idx + 1, block)
            applied.append(f"P1 expiry gate at line {idx+1}")
        else:
            applied.append("P1 expiry gate: already present")
        break

# ── PATCH 2: Max positions gate ───────────────────────────────────────────────
# Insert before the capital_block check (elif (currently_deployed + ...) > live_max_deploy)
for idx, line in enumerate(lines):
    if "currently_deployed + estimated_margin" in line and "live_max_deploy" in line and "elif" in line:
        if not any("max_positions" in lines[idx + j] for j in range(-3, 1)):
            ind = len(line) - len(line.lstrip())
            p = " " * ind
            block = (
                f"{p}elif len(open_trades) >= max_positions:\n"
                f"{p}    run_backtest._entry_dbg['max_pos_block'] = run_backtest._entry_dbg.get('max_pos_block', 0) + 1\n"
                f"\n"
            )
            lines.insert(idx, block)
            applied.append(f"P2 max_pos gate at line {idx+1}")
        else:
            applied.append("P2 max_pos gate: already present")
        break

# ── PATCH 3: Force-close open trades at sim end ───────────────────────────────
for idx, line in enumerate(lines):
    if "# End Simulation" in line:
        if not any("SIM_END" in lines[idx + j] for j in range(1, 15)):
            block = (
                "    # Force-close any positions still open at simulation end\n"
                "    if open_trades:\n"
                "        print(f'\\n[SIM_END] Force-closing {len(open_trades)} open position(s) at MTM.')\n"
                "    for tr in open_trades:\n"
                "        _fc_pnl = getattr(tr, 'unrealized_pnl_real', tr.unrealized_pnl)\n"
                "        equity += _fc_pnl\n"
                "        closed_trades.append({\n"
                "            'trade_id':    tr.trade_id,\n"
                "            'entry_dt':    tr.entry_dt,\n"
                "            'exit_dt':     str(spot_timestamps[num_bars - 1]),\n"
                "            'pnl':         round(_fc_pnl, 4),\n"
                "            'pnl_pct':     round(tr.pnl_pct * 100, 4),\n"
                "            'reason':      'SIM_END',\n"
                "            'max_dd':      round(tr.max_dd_pct * 100, 4),\n"
                "            'exit_details': 'force_closed_at_sim_end',\n"
                "        })\n"
                "        _trace_close(tr, _fc_pnl, 'SIM_END')\n"
                "        print(f'  [SIM_END] {tr.trade_id}: MTM PnL=${_fc_pnl:,.2f}')\n"
                "    open_trades = []\n"
                "\n"
            )
            lines.insert(idx + 1, block)
            applied.append(f"P3 sim_end force-close at line {idx+1}")
        else:
            applied.append("P3 sim_end force-close: already present")
        break

# ── PATCH 4: Gate summary — expiry_past_sim + max_pos_block ──────────────────
# Anchor on the credit_fail summary line (single-line, safe to insert after)
for idx, line in enumerate(lines):
    if "'credit_fail'" in line and "d['credit_fail']" in line and "print" in line:
        if not any("expiry_past_sim" in lines[idx + j] for j in range(1, 5)):
            ind = len(line) - len(line.lstrip())
            p = " " * ind
            block = (
                f"{p}print(\"  \" + \"expiry_past_sim\".ljust(28) + \": \" + str(d.get(\"expiry_past_sim\", 0)).rjust(8) + \"  (\" + f\"{{d.get('expiry_past_sim', 0)/bars*100:.1f}}\" + \"%)\")\n"
                f"{p}print(\"  \" + \"max_pos_block\".ljust(28) + \": \" + str(d.get(\"max_pos_block\", 0)).rjust(8) + \"  (\" + f\"{{d.get('max_pos_block', 0)/bars*100:.1f}}\" + \"%)\")\n"
            )
            lines.insert(idx + 1, block)
            applied.append(f"P4 gate summary at line {idx+1}")
        else:
            applied.append("P4 gate summary: already present")
        break

with open(TARGET, "w", encoding="utf-8") as f:
    f.writelines(lines)

print("\nPatch results:")
for a in applied:
    print(f"  {a}")
print(f"\nBackup: {TARGET}.bak3")
