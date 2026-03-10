"""
populate_strategy_exits.py
Pre-populate stop_loss_mult, stop_loss_dollar, profit_target, and max_contracts
for all 58 strategy scripts.

Philosophy (user-specified):
  - SL <= $1000 (tight stops, prefer $500-$750)
  - PT >= 2x SL (rely on positive expectancy, not win rate)
  - Reduce max_contracts for high-cost debit / long-vol strategies
"""
import os, re

strat_dir = os.path.join(os.path.dirname(__file__), "..", "kaggle", "strategies")
strat_dir = os.path.normpath(strat_dir)

# (sl_mult, sl_dollar, profit_target, max_contracts, note)
ASSIGNMENTS = {
    # Already fully set
    "short_call":    "SKIP",
    "short_put":     "SKIP",
    "covered_call":  "SKIP",

    # Only need profit_target (SL already set)
    "cash_secured_put":         (1.5,  500,  2500, 10, "CSP = short put with cash: same exits as short_put"),
    "short_straddle":           (1.5,  750,  2000, 10, "ATM call+put: big credit, 2.67:1 PT/SL"),
    "short_strangle":           (1.5,  750,  2000, 10, "OTM call+put: same family as straddle"),
    "short_guts":               (1.5,  750,  2000, 10, "ITM call+put: large credit, asymmetric risk"),

    # Only need stop_loss_dollar (pt already set from prior analysis)
    "bear_put_spread":          (1.5,  500,  1029, 10, "Debit bear spread: $500 cap, keep $1029 PT"),
    "bull_call_spread":         (1.5,  500,   820, 10, "Debit bull spread: $500 cap, keep $820 PT"),
    "iron_butterfly":           (2.0,  750,  1100, 10, "4-leg defined: $750 cap, keep $1100 PT"),

    # ShortVolIncome credit spreads
    "bear_call_spread_credit":  (1.5,  500,  1500, 10, "2-leg credit spread: 3:1 PT/SL"),
    "bull_put_spread_credit":   (1.5,  500,  1500, 10, "2-leg credit spread: 3:1 PT/SL"),
    "iron_condor":              (2.0,  750,  1500, 10, "4-leg defined: standard IC management"),
    "jade_lizard":              (1.5,  600,  1500, 10, "Short put + bear call spread: credit hybrid"),
    "reverse_jade_lizard":      (1.5,  600,  1500, 10, "Long call + bull put spread: credit hybrid"),
    "call_ratio_spread":        (1.5,  600,  1500, 10, "Sell 2 / buy 1 call: semi-undefined upside"),
    "put_ratio_spread":         (1.5,  600,  1500, 10, "Sell 2 / buy 1 put: semi-undefined downside"),
    "short_call_condor":        (1.5,  500,  1500, 10, "4-leg credit call condor: defined risk"),
    "short_put_condor":         (1.5,  500,  1500, 10, "4-leg credit put condor: defined risk"),
    "covered_short_straddle":   (1.5,  750,  2000,  5, "Covered + short straddle: 5 contracts (has underlying)"),
    "covered_short_strangle":   (1.5,  750,  2000,  5, "Covered + short strangle: 5 contracts"),

    # Directional defined risk: ladders
    "bear_call_ladder":         (1.5,  600,  1500, 10, "1x2 bear call: undefined upside gap risk"),
    "bear_put_ladder":          (1.5,  600,  1500, 10, "1x2 bear put: defined downside, gap risk up"),
    "bull_call_ladder":         (1.5,  600,  1500, 10, "1x2 bull call: defined upside, gap risk down"),
    "bull_put_ladder":          (1.5,  600,  1500, 10, "1x2 bull put: undefined downside gap risk"),

    # Broken wing butterflies
    "call_broken_wing":         (1.5,  500,  1500, 10, "BWB call: asymmetric, defined both sides"),
    "put_broken_wing":          (1.5,  500,  1500, 10, "BWB put: asymmetric, defined both sides"),
    "inverse_call_broken_wing": (1.5,  500,  1500, 10, "Inverted BWB call: long gamma variant"),
    "inverse_put_broken_wing":  (1.5,  500,  1500, 10, "Inverted BWB put: long gamma variant"),

    # Butterfly / condor (credit body)
    "short_call_butterfly":     (1.5,  500,  1500, 10, "Sell ATM call body, buy wing calls: limited credit"),
    "short_put_butterfly":      (1.5,  500,  1500, 10, "Sell ATM put body, buy wing puts: limited credit"),

    # Diagonals
    "diagonal_call":            (1.5,  500,  1200, 10, "Bull diagonal: time value + direction"),
    "diagonal_put":             (1.5,  500,  1200, 10, "Bear diagonal: time value + direction"),

    # Synthetic / combo
    "long_combo":               (1.5,  600,  1500, 10, "Long call + short put: synthetic long"),
    "long_synthetic_future":    (1.5,  600,  1500, 10, "ATM long call + short put: full synthetic"),
    "short_combo":              (1.5,  600,  1500, 10, "Short call + long put: synthetic short"),
    "short_synthetic_future":   (1.5,  600,  1500, 10, "ATM short call + long put: full synthetic short"),
    "collar":                   (1.5,  600,  1500, 10, "Long stock + long put + short call: hedge"),
    "protective_put":           (1.5,  500,  1500, 10, "Long stock + long put: defined floor"),
    "synthetic_put":            (1.5,  600,  1500, 10, "Short call + long combo: synthetic put exposure"),

    # Long debit single-leg
    "long_call":                (1.5,  500,  2000,  5, "Buy call: cut at $500, target $2000 (4:1)"),
    "long_put":                 (1.5,  500,  2000,  5, "Buy put: cut at $500, target $2000 (4:1)"),

    # Long butterflies / condors (debit)
    "long_call_butterfly":      (1.5,  500,  1000, 10, "Debit call butterfly: tight range play"),
    "long_call_condor":         (1.5,  500,  1000, 10, "Debit call condor: wider range play"),
    "long_put_butterfly":       (1.5,  500,  1000, 10, "Debit put butterfly: tight range play"),
    "long_put_condor":          (1.5,  500,  1000, 10, "Debit put condor: wider range play"),

    # Long vol (LongVolConvex) — expensive positions, fewer contracts
    "straddle_long":            (1.5,  750,  2500,  5, "Buy ATM straddle: big move needed, 3.3:1"),
    "strangle_long":            (1.5,  750,  2500,  5, "Buy OTM strangle: cheaper but needs bigger move"),
    "guts":                     (1.5,  750,  2500,  5, "Buy ITM call+put: expensive, needs strong move"),
    "inverse_iron_butterfly":   (1.5,  750,  2000,  5, "Long body short wings: long vol defined risk"),
    "inverse_iron_condor":      (1.5,  750,  2000,  5, "Long body short wings wider: long vol"),
    "call_ratio_backspread":    (1.5,  500,  2000,  5, "Buy 2 / sell 1 call: net long vol, unlimited up"),
    "put_ratio_backspread":     (1.5,  500,  2000,  5, "Buy 2 / sell 1 put: net long vol, unlimited down"),
    "strap":                    (1.5,  750,  2500,  5, "2 calls + 1 put: long vol bullish bias"),
    "strip":                    (1.5,  750,  2500,  5, "2 puts + 1 call: long vol bearish bias"),

    # Term structure / calendar
    "calendar_call":            (1.5,  500,  1000, 10, "Call time spread: theta decay, limited move"),
    "calendar_put":             (1.5,  500,  1000, 10, "Put time spread: theta decay, limited move"),
    "double_diagonal":          (1.5,  600,  1200, 10, "2 diagonals: wider term structure play"),
}


def update_field(text, key, new_val):
    """Replace  "key":  <old>,  preserving inline comments."""
    pattern = r'("' + key + r'"\s*:\s*)([\w\.]+)(,\s*(?:#[^\n]*)?)(\n)'
    def replacer(m):
        return f'{m.group(1)}{new_val}{m.group(3)}{m.group(4)}'
    new_text, n = re.subn(pattern, replacer, text)
    return new_text, n > 0


updated, skipped, errors = [], [], []

for fname in sorted(os.listdir(strat_dir)):
    if fname.startswith("_") or not fname.endswith(".py"):
        continue
    tid = fname[:-3]
    if tid not in ASSIGNMENTS:
        errors.append(f"NO_ASSIGNMENT: {tid}")
        continue
    asgn = ASSIGNMENTS[tid]
    if asgn == "SKIP":
        skipped.append(tid)
        continue

    sl_mult, sl_dollar, profit_target, max_contracts, note = asgn
    fpath = os.path.join(strat_dir, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        text = f.read()

    orig = text
    text, ok1 = update_field(text, "stop_loss_mult",   str(sl_mult))
    text, ok2 = update_field(text, "stop_loss_dollar",  str(sl_dollar))
    text, ok3 = update_field(text, "profit_target",     str(profit_target))
    text, ok4 = update_field(text, "max_contracts",     str(max_contracts))

    if text != orig:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(text)
        updated.append(
            f"  {tid:<33} sl_x={sl_mult}  sl_$=${sl_dollar:<4}  pt=${profit_target:<5}  max_c={max_contracts}"
            f"  fields_hit=[{int(ok1)}{int(ok2)}{int(ok3)}{int(ok4)}]"
        )
    else:
        errors.append(f"REGEX_MISS (no change written): {tid}")

print(f"UPDATED ({len(updated)}):")
for u in updated:
    print(u)
print(f"\nSKIPPED — already set ({len(skipped)}): {', '.join(skipped)}")
if errors:
    print(f"\nERRORS ({len(errors)}):")
    for e in errors:
        print(f"  {e}")
