#!/usr/bin/env python3
"""
patch_entry_display.py  (v2 — line-based, robust)
Applies two display patches to condor_brain_backtest_v45.py:
  1. Adds Strategy / Balance / Wins/Losses / Max DD to ENTRY ATTEMPT header
  2. Replaces the big TRADE OPENED panel with compact line + strategy name

Run once on Lightning AI:
    python patch_entry_display.py
"""
import shutil, sys, os, re

TARGET = "kaggle/condor_brain_backtest_v45.py"
if not os.path.exists(TARGET):
    print(f"ERROR: {TARGET} not found. Run from repo root.")
    sys.exit(1)

shutil.copy(TARGET, TARGET + ".bak")
with open(TARGET, "r", encoding="utf-8") as f:
    lines = f.readlines()

# ── PATCH 1: insert stats block after ENTRY ATTEMPT header ───────────────────
p1_applied = False
for idx, line in enumerate(lines):
    if "ENTRY ATTEMPT bar=" in line and "print(" in line:
        # Replace any unicode dash sequences with ASCII ---
        lines[idx] = re.sub(r"[\u2500\u2014\u2013]{2,}", "---", line)
        indent = "                             "  # 29 spaces (matches surrounding code)
        stats = (
            f'{indent}_ea_close = [r for r in closed_trades if r.get("action") != "OPEN" and r.get("reason")]\n'
            f'{indent}_ea_wins  = sum(1 for r in _ea_close if float(r.get("pnl") or 0) > 0)\n'
            f'{indent}_ea_loss  = sum(1 for r in _ea_close if float(r.get("pnl") or 0) < 0)\n'
            f'{indent}_ea_eq    = equity + sum(t.unrealized_pnl for t in open_trades)\n'
            f'{indent}_ea_dd    = (_ea_eq - peak_equity) / peak_equity * 100 if peak_equity > 0 else 0.0\n'
            f'{indent}_ea_strat = V43_STRATEGY_NAMES[_sidx] if 0 <= _sidx < len(V43_STRATEGY_NAMES) else str(_sidx)\n'
            f'{indent}print(f"  Strategy  : {{_ea_strat}} (idx={{_sidx}})")\n'
            f'{indent}print(f"  Balance   : ${{_ea_eq:,.2f}}")\n'
            f'{indent}print(f"  Wins / Losses: {{_ea_wins}} / {{_ea_loss}}")\n'
            f'{indent}print(f"  Max DD    : {{abs(_ea_dd):.1f}}%")\n'
        )
        lines.insert(idx + 1, stats)
        p1_applied = True
        print(f"PATCH 1 applied at line {idx+1}")
        break

# ── PATCH 2: replace big TRADE OPENED panel with compact format ───────────────
# Find "# Rolling stats panel" comment, then delete everything up to and
# including the closing print(  ...  ) block, replace with compact version.
p2_applied = False
start_idx = None
for idx, line in enumerate(lines):
    if "# Rolling stats panel" in line:
        start_idx = idx
        break

if start_idx is not None:
    # Find the end of the print( block — look for a line that is just spaces + ")"
    # after start_idx
    end_idx = None
    paren_depth = 0
    in_print = False
    for idx in range(start_idx, min(start_idx + 40, len(lines))):
        l = lines[idx].rstrip()
        if "print(" in lines[idx] and not in_print:
            in_print = True
        if in_print:
            paren_depth += lines[idx].count("(") - lines[idx].count(")")
            if paren_depth <= 0:
                end_idx = idx
                break

    if end_idx is not None:
        indent2 = "                                        "  # 40 spaces
        compact = (
            f'{indent2}_strat_name = V43_STRATEGY_NAMES[_sidx] if v43_outputs and 0 <= _sidx < len(V43_STRATEGY_NAMES) else "?"\n'
            f"{indent2}print(f\"\\n  ✅ TRADE OPENED [{{trade_id}}] bar={{i}} ts={{str(ts)[:19]}}\"\n"
            f"{indent2}      f\"\\n     spot={{spot:.2f}} | credit=${{net_credit_val:.4f}}/contract\"\n"
            f"{indent2}      f\" | qty={{filled_qty}} | margin=${{trade_margin:.0f}}\"\n"
            f"{indent2}      f\"\\n     SC={{legs.get('short_call')}} LC={{legs.get('long_call')}}\"\n"
            f"{indent2}      f\" SP={{legs.get('short_put')}} LP={{legs.get('long_put')}}\"\n"
            f"{indent2}      f\"\\n     entry={{_es:.3f}} pop={{_pop:.3f}}\"\n"
            f"{indent2}      f\" strategy={{_strat_name}}({{_sidx if v43_outputs else '?'}})\"\n"
            f"{indent2}      f\" deployed_after=${{currently_deployed_exact+trade_margin:.0f}}\")\n"
        )
        lines[start_idx : end_idx + 1] = [compact]
        p2_applied = True
        print(f"PATCH 2 applied (lines {start_idx+1}–{end_idx+1} replaced)")
    else:
        print("WARN: PATCH 2 — could not find end of print block")
else:
    print("WARN: PATCH 2 — '# Rolling stats panel' not found (may already be patched)")
    # Check if it's already compact (has _strat_name without the big panel)
    if any("_strat_name = V43_STRATEGY_NAMES" in l and "Rolling" not in lines[max(0,i-3):i+1][0]
           for i, l in enumerate(lines)):
        print("  -> Looks like compact format already applied; skipping.")
        p2_applied = True  # treat as success

with open(TARGET, "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"\nPATCH 1 (ENTRY ATTEMPT stats): {'applied' if p1_applied else 'FAILED'}")
print(f"PATCH 2 (TRADE OPENED compact): {'applied' if p2_applied else 'FAILED'}")
print("Backup saved to", TARGET + ".bak")
