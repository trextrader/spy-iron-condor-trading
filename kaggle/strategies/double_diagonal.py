"""
double_diagonal.py — Double Diagonal
=======================================

FAMILY: D) Term Structure — exploit differential theta between near and far expirations
CLASS:  custom_multi_leg (index 8)
DIRECTION: Neutral

WHAT IT IS:
    Calendar + strangle. Sell near OTM strangle, buy far OTM strangle. Exploits term structure IV edge.

LEG STRUCTURE:
#   Leg 1: LONG P rank0 [far]
#   Leg 2: SHORT P rank1 [near]
#   Leg 3: SHORT C rank2 [near]
#   Leg 4: LONG C rank3 [far]

RISK PROFILE:
    Risk:   Limited to net debit paid.
    Reward: Profits from theta differential + IV normalization.

WHEN IT FIRES (predicate atoms):
    adx_weak & consol_high & ivr_high & regime_stable & friction_ok

HOW IT'S USED IN THE BACKTESTER:
    1. CondorNet v4.3 predicts strategy class 'custom_multi_leg' (index 8)
    2. All templates mapped to this class are scored
    3. Eligibility predicate checks market atoms (above)
    4. If eligible, template is scored: 0.40*pop + 0.35*tanh(3*ev) - 0.15*tanh(2*ml) + 0.10*dte_aff
    5. Highest-scoring eligible template wins; legs are built from chain
    6. Fill engine executes the trade with qty, margin, and exit rules from CONFIG below
"""

CONFIG = {
    # ── Identity ──────────────────────────────────────────────────────────
    "template_id":        "double_diagonal",
    "class_name":         "custom_multi_leg",
    "class_idx":          8,
    "family":             "TermStructure",

    # ── Leg Structure ─────────────────────────────────────────────────────
    "legs": [
        {"type": "P", "side": "LONG", "qty": 1, "rank": 0},
        {"type": "P", "side": "SHORT", "qty": 1, "rank": 1},
        {"type": "C", "side": "SHORT", "qty": 1, "rank": 2},
        {"type": "C", "side": "LONG", "qty": 1, "rank": 3}
    ],

    # ── Sizing ────────────────────────────────────────────────────────────
    "max_contracts":      10,          # Max contracts per trade
    "margin_pct":         0.50,        # Max % of buying power to deploy per trade
    "margin_type":        "width",   # "width" (spread) or "pct_spot" (naked)
    "margin_spot_pct":    0.5,      # If pct_spot: margin = spot * this * 100

    # ── Exit: Stop-Loss ───────────────────────────────────────────────────
    "stop_loss_mult":     2.0,   # Close when loss >= N * |credit| * qty * 100
    "stop_loss_dollar":   None,  # Hard dollar cap per trade (None = use multiplier only)

    # ── Exit: Profit Target ───────────────────────────────────────────────
    "profit_target":      None,    # Dollar target (None = use 50% of credit)

    # ── Template Preference ───────────────────────────────────────────────
    "fallback_template":  None,        # If no template eligible, use this template_id

    # ── Entry Gate Overrides ──────────────────────────────────────────────
    "entry_threshold":    None,        # Override entry_logit threshold (default 0.55)
    "pop_threshold":      None,        # Override PoP threshold (default 0.50)

    # ── Position Limits ───────────────────────────────────────────────────
    "max_positions":      6,           # Max concurrent open positions for this strategy
    "cooldown_bars":      5,           # Min bars between trades
}
