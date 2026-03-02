#!/usr/bin/env python3
"""
patch_sim_guards.py
Applies three simulation guard patches to condor_brain_backtest_v45.py:
  1. --max-positions CLI arg (default 5)
  2. Entry gate: skip if position expires past sim_end_dt
  3. Force-close all open trades at sim end (SIM_END reason)

Run once on Lightning AI:
    python patch_sim_guards.py
"""
import shutil, sys, os, re

TARGET = "kaggle/condor_brain_backtest_v45.py"
if not os.path.exists(TARGET):
    print(f"ERROR: {TARGET} not found. Run from repo root.")
    sys.exit(1)

shutil.copy(TARGET, TARGET + ".bak2")
with open(TARGET, "r", encoding="utf-8") as f:
    lines = f.readlines()

results = {}

# ─── PATCH 1: Add --max-positions CLI arg ────────────────────────────────────
# Find --limit arg line and insert --max-positions after it
for idx, line in enumerate(lines):
    if '"--limit"' in line and 'Limit the number of bars' in line:
        # Find the end of this argument block (next add_argument call)
        end = idx + 1
        while end < len(lines) and 'add_argument' not in lines[end]:
            end += 1
        insert_at = end
        new_arg = (
            '    parser.add_argument("--max-positions", type=int, default=5,\n'
            '                        help="Max simultaneous open IC positions (default: 5)")\n'
        )
        lines.insert(insert_at, new_arg)
        results['p1_arg'] = f"inserted after line {idx+1}"
        break

# ─── PATCH 2: Wire max_positions into run_backtest signature + body ──────────
# Find run_backtest function signature and add max_positions param
for idx, line in enumerate(lines):
    if 'def run_backtest(' in line:
        # Find the closing ) of the signature
        end = idx
        depth = line.count('(') - line.count(')')
        while depth > 0 and end < idx + 20:
            end += 1
            depth += lines[end].count('(') - lines[end].count(')')
        # Add param before the closing ):
        lines[end] = lines[end].replace('):', ', max_positions=5):')
        results['p2_sig'] = f"param added at line {end+1}"
        break

# Find where run_backtest is called (in main) and wire --max-positions
for idx, line in enumerate(lines):
    if 'run_backtest(' in line and 'args.' in lines[idx] or \
       ('run_backtest(' in line and idx + 1 < len(lines) and 'args.' in lines[idx+1]):
        # Look ahead for the closing ) of the call
        end = idx
        depth = line.count('(') - line.count(')')
        while depth > 0 and end < idx + 30:
            end += 1
            depth += lines[end].count('(') - lines[end].count(')')
        # Insert max_positions before closing )
        closing = lines[end]
        if ')' in closing:
            lines[end] = closing.replace(')', ',\n            max_positions=args.max_positions)', 1)
            results['p2_call'] = f"wired at line {end+1}"
        break

# ─── PATCH 3: Add sim_end_dt computation before sim loop ─────────────────────
for idx, line in enumerate(lines):
    if 'sim_start_idx = SEQ_LEN' in line and 'Warmup' in lines[max(0,idx-2):idx+1][0]:
        # Already inserted? Check next few lines
        already = any('sim_end_dt' in lines[idx+j] for j in range(1, 5))
        if not already:
            insert_line = (
                '\n    # Last bar timestamp — no new entry may expire past this date\n'
                '    sim_end_dt = pd.Timestamp(spot_timestamps[num_bars - 1])\n'
            )
            lines.insert(idx + 1, insert_line)
            results['p3_simend'] = f"inserted after line {idx+1}"
        else:
            results['p3_simend'] = "already present"
        break

# ─── PATCH 4: Entry gate — skip if expiry past sim_end_dt ────────────────────
for idx, line in enumerate(lines):
    if 'dte = float(np.clip(dte_raw' in line:
        already = any('expiry_past_sim' in lines[idx+j] for j in range(1, 8))
        if not already:
            indent = len(line) - len(line.lstrip())
            pad = ' ' * indent
            gate_lines = (
                f'\n'
                f'{pad}# Gate: block entries whose expiry falls after the sim window\n'
                f'{pad}_expiry_dt = pd.Timestamp(ts) + pd.Timedelta(days=dte)\n'
                f'{pad}if _expiry_dt > sim_end_dt:\n'
                f'{pad}    run_backtest._entry_dbg["expiry_past_sim"] = run_backtest._entry_dbg.get("expiry_past_sim", 0) + 1\n'
                f'{pad}    if verbose_sim:\n'
                f'{pad}        print(f"  -- SKIP expiry_past_sim -- expiry {{_expiry_dt.date()}} > sim_end {{sim_end_dt.date()}}")\n'
                f'{pad}    continue\n'
            )
            lines.insert(idx + 1, gate_lines)
            results['p4_gate'] = f"inserted after line {idx+1}"
        else:
            results['p4_gate'] = "already present"
        break

# ─── PATCH 5: Max positions gate in entry logic ───────────────────────────────
# Find capital_block check and insert max_positions gate before it
for idx, line in enumerate(lines):
    if 'capital_block' in line and 'run_backtest._entry_dbg' in line and 'currently_deployed' not in line:
        # Walk back to find the elif/else that starts this block
        already = any('max_positions' in lines[idx+j] for j in range(-5, 2))
        if not already:
            # Find the if/elif line that leads to capital_block
            check_idx = idx
            for j in range(idx, max(0, idx-5), -1):
                if lines[j].strip().startswith('elif') or lines[j].strip().startswith('if'):
                    check_idx = j
                    break
            indent = len(lines[check_idx]) - len(lines[check_idx].lstrip())
            pad = ' ' * indent
            max_pos_check = (
                f'{pad}elif len(open_trades) >= max_positions:\n'
                f'{pad}    run_backtest._entry_dbg["max_pos_block"] = run_backtest._entry_dbg.get("max_pos_block", 0) + 1\n'
            )
            lines.insert(check_idx, max_pos_check)
            results['p5_maxpos'] = f"inserted at line {check_idx+1}"
        else:
            results['p5_maxpos'] = "already present"
        break

# ─── PATCH 6: Force-close open trades at sim end ─────────────────────────────
for idx, line in enumerate(lines):
    if '# End Simulation' in line:
        already = any('SIM_END' in lines[idx+j] for j in range(1, 10))
        if not already:
            force_close = (
                '\n    # Force-close any positions still open at simulation end (mark-to-market)\n'
                '    if open_trades:\n'
                '        print(f"\\n[SIM_END] Force-closing {len(open_trades)} open position(s) at MTM.")\n'
                '    for tr in open_trades:\n'
                '        _fc_pnl = getattr(tr, "unrealized_pnl_real", tr.unrealized_pnl)\n'
                '        equity += _fc_pnl\n'
                '        closed_trades.append({\n'
                '            "trade_id":  tr.trade_id,\n'
                '            "entry_dt":  tr.entry_dt,\n'
                '            "exit_dt":   str(spot_timestamps[num_bars - 1]),\n'
                '            "pnl":       round(_fc_pnl, 4),\n'
                '            "pnl_pct":   round(tr.pnl_pct * 100, 4),\n'
                '            "reason":    "SIM_END",\n'
                '            "max_dd":    round(tr.max_dd_pct * 100, 4),\n'
                '            "exit_details": "force_closed_at_sim_end",\n'
                '        })\n'
                '        _trace_close(tr, _fc_pnl, "SIM_END")\n'
                '        print(f"  [SIM_END] {tr.trade_id}: MTM PnL=${_fc_pnl:,.2f}")\n'
                '    open_trades = []\n'
            )
            lines.insert(idx + 1, force_close)
            results['p6_forceclose'] = f"inserted after line {idx+1}"
        else:
            results['p6_forceclose'] = "already present"
        break

# ─── PATCH 7: Show expiry_past_sim + max_pos_block in gate summary ────────────
for idx, line in enumerate(lines):
    if "'atomicity_fail'" in line and 'entry_dbg' in line and 'print' in line:
        already = any('expiry_past_sim' in lines[idx+j] for j in range(1, 5))
        if not already:
            indent = len(line) - len(line.lstrip())
            pad = ' ' * indent
            summary_lines = (
                f"{pad}print(f\"  {{'expiry_past_sim':<28s}}: {{d.get('expiry_past_sim',0):>8,}}  ({{d.get('expiry_past_sim',0)/bars*100:.1f}}%)\")\n"
                f"{pad}print(f\"  {{'max_pos_block':<28s}}: {{d.get('max_pos_block',0):>8,}}  ({{d.get('max_pos_block',0)/bars*100:.1f}}%)\")\n"
            )
            lines.insert(idx + 1, summary_lines)
            results['p7_summary'] = f"inserted after line {idx+1}"
        else:
            results['p7_summary'] = "already present"
        break

with open(TARGET, "w", encoding="utf-8") as f:
    f.writelines(lines)

print("\nPatch results:")
for k, v in results.items():
    print(f"  {k}: {v}")
print(f"\nBackup saved to {TARGET}.bak2")
print("\nRun with:")
print("  python kaggle/condor_brain_backtest_v45.py --use-v43 --v43-model models/condor_net_v43_run18.pth \\")
print("    --v43-data-dir data/Datasetv4/v43 --limit 5000 --verbose --max-positions 3")
