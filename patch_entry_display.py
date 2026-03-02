#!/usr/bin/env python3
"""
patch_entry_display.py
Applies two display patches to condor_brain_backtest_v45.py:
  1. Adds Strategy / Balance / Wins/Losses / Max DD to ENTRY ATTEMPT header
  2. Replaces the big TRADE OPENED panel with compact line + strategy name

Run once on Lightning AI:
    python patch_entry_display.py
"""
import re, shutil, sys, os

TARGET = "kaggle/condor_brain_backtest_v45.py"

if not os.path.exists(TARGET):
    print(f"ERROR: {TARGET} not found. Run from repo root.")
    sys.exit(1)

shutil.copy(TARGET, TARGET + ".bak")

with open(TARGET, "r", encoding="utf-8") as f:
    src = f.read()

# ── PATCH 1: insert stats block after ENTRY ATTEMPT header line ───────────────
# Find the ENTRY ATTEMPT print, then the very next print((G) Chain...)
# Insert stats lines between them.

OLD_EA = (
    '                             print(f"\\n  \\u2500\\u2500\\u2500 ENTRY ATTEMPT'
    ' bar={i} ts={str(ts)[:19]} spot={spot:.2f} \\u2500\\u2500\\u2500")\n'
    '                             print(f"  (G) Chain:'
)
NEW_EA = (
    '                             print(f"\\n  --- ENTRY ATTEMPT'
    ' bar={i} ts={str(ts)[:19]} spot={spot:.2f} ---")\n'
    '                             _ea_close = [r for r in closed_trades if r.get("action") != "OPEN" and r.get("reason")]\n'
    '                             _ea_wins  = sum(1 for r in _ea_close if float(r.get("pnl") or 0) > 0)\n'
    '                             _ea_loss  = sum(1 for r in _ea_close if float(r.get("pnl") or 0) < 0)\n'
    '                             _ea_eq    = equity + sum(t.unrealized_pnl for t in open_trades)\n'
    '                             _ea_dd    = (_ea_eq - peak_equity) / peak_equity * 100 if peak_equity > 0 else 0.0\n'
    '                             _ea_strat = V43_STRATEGY_NAMES[_sidx] if 0 <= _sidx < len(V43_STRATEGY_NAMES) else str(_sidx)\n'
    '                             print(f"  Strategy  : {_ea_strat} (idx={_sidx})")\n'
    '                             print(f"  Balance   : ${_ea_eq:,.2f}")\n'
    '                             print(f"  Wins / Losses: {_ea_wins} / {_ea_loss}")\n'
    '                             print(f"  Max DD    : {abs(_ea_dd):.1f}%")\n'
    '                             print(f"  (G) Chain:'
)

# Use regex to handle actual unicode chars in file
ea_pattern = re.compile(
    r'(                             print\(f"\\n  [^\n]+ ENTRY ATTEMPT bar=\{i\}[^\n]+\n)'
    r'(                             print\(f"  \(G\) Chain:)',
    re.UNICODE
)

def ea_replacer(m):
    header_line = m.group(1)
    # Replace unicode dashes with ASCII dashes in header
    header_line = re.sub(r'[\u2500\u2014\u2013\-]{3,}', '---', header_line)
    stats_block = (
        '                             _ea_close = [r for r in closed_trades if r.get("action") != "OPEN" and r.get("reason")]\n'
        '                             _ea_wins  = sum(1 for r in _ea_close if float(r.get("pnl") or 0) > 0)\n'
        '                             _ea_loss  = sum(1 for r in _ea_close if float(r.get("pnl") or 0) < 0)\n'
        '                             _ea_eq    = equity + sum(t.unrealized_pnl for t in open_trades)\n'
        '                             _ea_dd    = (_ea_eq - peak_equity) / peak_equity * 100 if peak_equity > 0 else 0.0\n'
        '                             _ea_strat = V43_STRATEGY_NAMES[_sidx] if 0 <= _sidx < len(V43_STRATEGY_NAMES) else str(_sidx)\n'
        '                             print(f"  Strategy  : {_ea_strat} (idx={_sidx})")\n'
        '                             print(f"  Balance   : ${_ea_eq:,.2f}")\n'
        '                             print(f"  Wins / Losses: {_ea_wins} / {_ea_loss}")\n'
        '                             print(f"  Max DD    : {abs(_ea_dd):.1f}%")\n'
    )
    return header_line + stats_block + m.group(2)

new_src, n1 = ea_pattern.subn(ea_replacer, src)
if n1 == 0:
    print("WARN: ENTRY ATTEMPT pattern not matched — trying line-based approach")
    lines = src.splitlines(keepends=True)
    for idx, line in enumerate(lines):
        if 'ENTRY ATTEMPT bar=' in line:
            # Replace unicode dashes
            lines[idx] = re.sub(r'[\u2500]{3,}', '---', line)
            stats = (
                '                             _ea_close = [r for r in closed_trades if r.get("action") != "OPEN" and r.get("reason")]\n'
                '                             _ea_wins  = sum(1 for r in _ea_close if float(r.get("pnl") or 0) > 0)\n'
                '                             _ea_loss  = sum(1 for r in _ea_close if float(r.get("pnl") or 0) < 0)\n'
                '                             _ea_eq    = equity + sum(t.unrealized_pnl for t in open_trades)\n'
                '                             _ea_dd    = (_ea_eq - peak_equity) / peak_equity * 100 if peak_equity > 0 else 0.0\n'
                '                             _ea_strat = V43_STRATEGY_NAMES[_sidx] if 0 <= _sidx < len(V43_STRATEGY_NAMES) else str(_sidx)\n'
                '                             print(f"  Strategy  : {_ea_strat} (idx={_sidx})")\n'
                '                             print(f"  Balance   : ${_ea_eq:,.2f}")\n'
                '                             print(f"  Wins / Losses: {_ea_wins} / {_ea_loss}")\n'
                '                             print(f"  Max DD    : {abs(_ea_dd):.1f}%")\n'
            )
            lines.insert(idx + 1, stats)
            n1 = 1
            break
    new_src = ''.join(lines)

# ── PATCH 2: replace big TRADE OPENED panel with compact line + strategy name ─
# The big panel starts at "# Rolling stats panel" and ends at the closing )
big_panel_pattern = re.compile(
    r'                                        # Rolling stats panel\n'
    r'                                        _close_recs = .*?\n'
    r'                                        _n_wins     = .*?\n'
    r'                                        _n_losses   = .*?\n'
    r'                                        _cur_eq     = .*?\n'
    r'                                        _dd_pct     = .*?\n'
    r'                                        _strat_name = .*?\n'
    r'                                        print\(\n'
    r'(?:.*?\n)*?'                    # all the f-string lines
    r'                                        \)\n',
    re.DOTALL
)

COMPACT_OPENED = (
    '                                        _strat_name = V43_STRATEGY_NAMES[_sidx] if v43_outputs and 0 <= _sidx < len(V43_STRATEGY_NAMES) else "?"\n'
    '                                        print(f"\\n  ✅ TRADE OPENED [{trade_id}] bar={i} ts={str(ts)[:19]}"\n'
    '                                              f"\\n     spot={spot:.2f} | credit=${net_credit_val:.4f}/contract"\n'
    '                                              f" | qty={filled_qty} | margin=${trade_margin:.0f}"\n'
    '                                              f"\\n     SC={legs.get(\'short_call\')} LC={legs.get(\'long_call\')}"\n'
    '                                              f" SP={legs.get(\'short_put\')} LP={legs.get(\'long_put\')}"\n'
    '                                              f"\\n     entry={_es:.3f} pop={_pop:.3f}"\n'
    '                                              f" strategy={_strat_name}({_sidx if v43_outputs else \'?\'})"\n'
    '                                              f" deployed_after=${currently_deployed_exact+trade_margin:.0f}")\n'
)

new_src2, n2 = big_panel_pattern.subn(COMPACT_OPENED, new_src)
if n2 == 0:
    print("WARN: big panel pattern not matched — TRADE OPENED unchanged")
    new_src2 = new_src
else:
    print(f"PATCH 2 applied ({n2} replacement)")

with open(TARGET, "w", encoding="utf-8") as f:
    f.write(new_src2)

print(f"PATCH 1 (ENTRY ATTEMPT stats): {'applied' if n1 else 'FAILED'}")
print(f"PATCH 2 (TRADE OPENED compact): {'applied' if n2 else 'FAILED'}")
print("Done. Backup saved to", TARGET + ".bak")
