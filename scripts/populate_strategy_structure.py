"""
populate_strategy_structure.py
Add entry-structure and exit-time params to all 58 strategy CONFIG dicts.

New params:
  call_offset_pct  — % OTM for call leg  (None = neural model controls)
  put_offset_pct   — % OTM for put leg   (None = neural model controls)
  spread_width     — spread width points  (None = neural model; N/A single-leg)
  target_dte       — target DTE           (None = neural model controls)
  max_leg_spread   — bid-ask quality gate (0.15 default; 0.20 for naked/synthetic)
  max_dte_exit     — close when DTE <= N  (0 = expire naturally; >0 = early close)
  hold_days        — max calendar days held regardless of DTE

Philosophy:
  None = neural model output is used; set a value to pin that dimension for optimization.
  Optimizer can grid-search any of these per-strategy independently.
"""
import os, re

strat_dir = os.path.join(os.path.dirname(__file__), "..", "kaggle", "strategies")
strat_dir = os.path.normpath(strat_dir)

# (call_offset_pct, put_offset_pct, spread_width, target_dte, max_leg_spread, max_dte_exit, hold_days)
ASSIGNMENTS = {
    # ── Naked / covered single-leg short ────────────────────────────────────────
    "short_call":               (1.5,  None, None,  7, 0.20, 0,  7),
    "short_put":                (None, 1.5,  None,  7, 0.20, 0,  7),
    "covered_call":             (1.5,  None, None, 21, 0.20, 1, 21),
    "cash_secured_put":         (None, 1.5,  None, 21, 0.20, 1, 21),

    # ── Single-leg long directional ─────────────────────────────────────────────
    "long_call":                (2.0,  None, None, 30, 0.15, 5, 20),
    "long_put":                 (None, 2.0,  None, 30, 0.15, 5, 20),
    "protective_put":           (None, 1.0,  None, 30, 0.15, 5, 21),

    # ── 2-leg debit/credit spreads ───────────────────────────────────────────────
    "bear_call_spread_credit":  (1.5,  None,  5,    7, 0.15, 0,  7),
    "bull_put_spread_credit":   (None, 1.5,   5,    7, 0.15, 0,  7),
    "bear_put_spread":          (None, 1.5,   5,    7, 0.15, 0,  7),
    "bull_call_spread":         (1.5,  None,  5,    7, 0.15, 0,  7),

    # ── 4-leg Iron Condor ────────────────────────────────────────────────────────
    "iron_condor":              (1.5,  1.5,   5,   14, 0.15, 1, 14),

    # ── Iron / Inverse Butterfly ─────────────────────────────────────────────────
    "iron_butterfly":           (0.0,  0.0,   5,    7, 0.15, 1,  7),
    "inverse_iron_butterfly":   (0.0,  0.0,   5,   14, 0.15, 1, 14),
    "inverse_iron_condor":      (1.5,  1.5,   5,   14, 0.15, 1, 14),

    # ── Butterflies (call / put) ─────────────────────────────────────────────────
    "short_call_butterfly":     (0.0,  None,  5,    7, 0.15, 0,  5),
    "short_put_butterfly":      (None, 0.0,   5,    7, 0.15, 0,  5),
    "long_call_butterfly":      (0.0,  None,  5,   14, 0.15, 3, 14),
    "long_put_butterfly":       (None, 0.0,   5,   14, 0.15, 3, 14),

    # ── Condors (credit/debit) ───────────────────────────────────────────────────
    "short_call_condor":        (1.5,  None,  5,   14, 0.15, 1, 14),
    "short_put_condor":         (None, 1.5,   5,   14, 0.15, 1, 14),
    "long_call_condor":         (1.5,  None,  5,   21, 0.15, 3, 21),
    "long_put_condor":          (None, 1.5,   5,   21, 0.15, 3, 21),

    # ── Short vol: straddle / strangle / guts ───────────────────────────────────
    "short_straddle":           (0.0,  0.0,  None, 14, 0.15, 1, 14),
    "short_strangle":           (2.0,  2.0,  None, 14, 0.15, 1, 14),
    "short_guts":               (1.0,  1.0,  None, 14, 0.15, 1, 14),

    # ── Long vol: straddle / strangle / guts / strap / strip ────────────────────
    "straddle_long":            (0.0,  0.0,  None, 21, 0.15, 5, 21),
    "strangle_long":            (2.0,  2.0,  None, 21, 0.15, 5, 21),
    "guts":                     (1.0,  1.0,  None, 21, 0.15, 5, 21),
    "strap":                    (0.0,  0.0,  None, 21, 0.15, 5, 21),
    "strip":                    (0.0,  0.0,  None, 21, 0.15, 5, 21),

    # ── Ratio spreads (semi-defined risk) ───────────────────────────────────────
    "call_ratio_spread":        (1.5,  None,  5,   14, 0.15, 1, 14),
    "put_ratio_spread":         (None, 1.5,   5,   14, 0.15, 1, 14),

    # ── Backspreads (long vol, directional) ─────────────────────────────────────
    "call_ratio_backspread":    (1.5,  None,  5,   21, 0.15, 3, 21),
    "put_ratio_backspread":     (None, 1.5,   5,   21, 0.15, 3, 21),

    # ── Ladders (1x2 undefined risk) ────────────────────────────────────────────
    "bear_call_ladder":         (1.5,  None,  5,    7, 0.15, 0,  7),
    "bear_put_ladder":          (None, 1.5,   5,    7, 0.15, 0,  7),
    "bull_call_ladder":         (1.5,  None,  5,    7, 0.15, 0,  7),
    "bull_put_ladder":          (None, 1.5,   5,    7, 0.15, 0,  7),

    # ── Broken wing butterflies ──────────────────────────────────────────────────
    "call_broken_wing":         (1.5,  None,  5,    7, 0.15, 0,  5),
    "put_broken_wing":          (None, 1.5,   5,    7, 0.15, 0,  5),
    "inverse_call_broken_wing": (1.5,  None,  5,   14, 0.15, 3, 14),
    "inverse_put_broken_wing":  (None, 1.5,   5,   14, 0.15, 3, 14),

    # ── Calendar / Diagonal / Double-diagonal ────────────────────────────────────
    "calendar_call":            (0.0,  None, None, 30, 0.15, 7, 21),
    "calendar_put":             (None, 0.0,  None, 30, 0.15, 7, 21),
    "diagonal_call":            (1.5,  None, None, 30, 0.15, 5, 21),
    "diagonal_put":             (None, 1.5,  None, 30, 0.15, 5, 21),
    "double_diagonal":          (1.5,  1.5,  None, 30, 0.15, 7, 21),

    # ── Synthetic / combo ────────────────────────────────────────────────────────
    "long_combo":               (1.5,  1.5,  None, 14, 0.20, 1, 14),
    "short_combo":              (1.5,  1.5,  None, 14, 0.20, 1, 14),
    "long_synthetic_future":    (0.0,  0.0,  None, 14, 0.20, 1, 14),
    "short_synthetic_future":   (0.0,  0.0,  None, 14, 0.20, 1, 14),
    "collar":                   (1.5,  1.5,  None, 21, 0.15, 3, 21),
    "synthetic_put":            (1.5,  1.5,  None, 14, 0.20, 1, 14),

    # ── Covered premium combos ───────────────────────────────────────────────────
    "covered_short_straddle":   (0.0,  0.0,  None, 21, 0.15, 1, 21),
    "covered_short_strangle":   (2.0,  2.0,  None, 21, 0.15, 1, 21),

    # ── Jade lizard variants ─────────────────────────────────────────────────────
    "jade_lizard":              (1.5,  1.5,   5,   14, 0.15, 1, 14),
    "reverse_jade_lizard":      (1.5,  1.5,   5,   14, 0.15, 1, 14),
}

# Insertion marker: add new section after "cooldown_bars" line
INSERT_AFTER = '"cooldown_bars"'

NEW_SECTION_TEMPLATE = """
    # ── Entry: Structure (overrides neural model when not None) ──────────────
    "call_offset_pct":    {call_offset_pct},  # % OTM for call strike (None = neural model)
    "put_offset_pct":     {put_offset_pct},  # % OTM for put strike  (None = neural model)
    "spread_width":       {spread_width},  # Spread width points   (None = neural model / N/A)
    "target_dte":         {target_dte},  # Target DTE at entry   (None = neural model)

    # ── Entry: Quality Filter ─────────────────────────────────────────────────
    "max_leg_spread":     {max_leg_spread},  # Max bid-ask ratio per leg

    # ── Exit: Time ────────────────────────────────────────────────────────────
    "max_dte_exit":       {max_dte_exit},  # Close when DTE remaining <= N (0 = expire naturally)
    "hold_days":          {hold_days},  # Max calendar days held"""


def _fmt(v):
    """Format a value for Python source: None stays None, floats keep decimal."""
    if v is None:
        return "None"
    if isinstance(v, float) and v == int(v):
        return f"{v:.1f}"
    return str(v)


updated, skipped, errors = [], [], []

for fname in sorted(os.listdir(strat_dir)):
    if fname.startswith("_") or not fname.endswith(".py"):
        continue
    tid = fname[:-3]
    if tid not in ASSIGNMENTS:
        errors.append(f"NO_ASSIGNMENT: {tid}")
        continue

    call_off, put_off, width, dte, mls, mde, hd = ASSIGNMENTS[tid]
    fpath = os.path.join(strat_dir, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        text = f.read()

    # Skip if already has the new params
    if '"call_offset_pct"' in text:
        skipped.append(tid)
        continue

    # Find the cooldown_bars line and insert after it
    # Pattern: "cooldown_bars": <value>,  [# optional comment] \n
    pattern = r'("cooldown_bars"\s*:\s*[^\n]+\n)'
    new_block = NEW_SECTION_TEMPLATE.format(
        call_offset_pct=_fmt(call_off),
        put_offset_pct=_fmt(put_off),
        spread_width=_fmt(width),
        target_dte=_fmt(dte),
        max_leg_spread=_fmt(mls),
        max_dte_exit=_fmt(mde),
        hold_days=_fmt(hd),
    )
    new_text, n = re.subn(pattern, r'\1' + new_block + '\n', text, count=1)

    if n == 0:
        errors.append(f"REGEX_MISS (cooldown_bars not found): {tid}")
        continue

    with open(fpath, "w", encoding="utf-8") as f:
        f.write(new_text)
    updated.append(
        f"  {tid:<33} call={_fmt(call_off):<4}  put={_fmt(put_off):<4}  "
        f"w={_fmt(width):<4}  dte={_fmt(dte):<3}  mls={mls}  mde={mde}  hd={hd}"
    )

print(f"UPDATED ({len(updated)}):")
for u in updated:
    print(u)
print(f"\nSKIPPED — already has params ({len(skipped)}): {', '.join(skipped)}")
if errors:
    print(f"\nERRORS ({len(errors)}):")
    for e in errors:
        print(f"  {e}")
