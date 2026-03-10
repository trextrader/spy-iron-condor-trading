"""
long_put_condor.py — Long Put Condor
=======================================

FAMILY: B) Long Vol / Convex — buy options, profit from large moves and volatility expansion
CLASS:  custom_multi_leg (index 8)
DIRECTION: Neutral

WHAT IT IS:
    Put-side rangebound debit. Mirror of long call condor.

LEG STRUCTURE:
#   Leg 1: LONG P rank3
#   Leg 2: SHORT P rank2
#   Leg 3: SHORT P rank1
#   Leg 4: LONG P rank0

RISK PROFILE:
    Risk:   Limited to net debit paid.
    Reward: Defined: max profit = inner width − debit.

WHEN IT FIRES (predicate atoms):
    ivr_low & consol_high & adx_weak & friction_ok

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
    "template_id":        "long_put_condor",
    "class_name":         "custom_multi_leg",
    "class_idx":          8,
    "family":             "LongVolConvex",

    # ── Leg Structure ─────────────────────────────────────────────────────
    "legs": [
        {"type": "P", "side": "LONG", "qty": 1, "rank": 3},
        {"type": "P", "side": "SHORT", "qty": 1, "rank": 2},
        {"type": "P", "side": "SHORT", "qty": 1, "rank": 1},
        {"type": "P", "side": "LONG", "qty": 1, "rank": 0}
    ],

    # ── Sizing ────────────────────────────────────────────────────────────
    "max_contracts":      10,          # Max contracts per trade
    "margin_pct":         0.50,        # Max % of buying power to deploy per trade
    "margin_type":        "width",   # "width" (spread) or "pct_spot" (naked)
    "margin_spot_pct":    0.5,      # If pct_spot: margin = spot * this * 100

    # ── Exit: Stop-Loss ───────────────────────────────────────────────────
    "stop_loss_mult":     1.5,   # Close when loss >= N * |credit| * qty * 100
    "stop_loss_dollar":   500,  # Hard dollar cap per trade (None = use multiplier only)

    # ── Exit: Profit Target ───────────────────────────────────────────────
    "profit_target":      1000,    # Dollar target (None = use 50% of credit)

    # ── Template Preference ───────────────────────────────────────────────
    "fallback_template":  None,        # If no template eligible, use this template_id

    # ── Entry Gate Overrides ──────────────────────────────────────────────
    "entry_threshold":    None,        # Override entry_logit threshold (default 0.55)
    "pop_threshold":      None,        # Override PoP threshold (default 0.50)

    # ── Position Limits ───────────────────────────────────────────────────
    "max_positions":      6,           # Max concurrent open positions for this strategy
    "cooldown_bars":      5,           # Min bars between trades

    # ── Entry: Structure (overrides neural model when not None) ──────────────
    "call_offset_pct":    None,  # % OTM for call strike (None = neural model)
    "put_offset_pct":     1.5,  # % OTM for put strike  (None = neural model)
    "spread_width":       5,  # Spread width points   (None = neural model / N/A)
    "target_dte":         21,  # Target DTE at entry   (None = neural model)

    # ── Entry: Quality Filter ─────────────────────────────────────────────────
    "max_leg_spread":     0.15,  # Max bid-ask ratio per leg

    # ── Entry: Delta Targets (template path) ────────────────────────────────
    "short_delta":        0.2,  # Target |delta| for short (selling) strikes
    "wing_delta":         0.1,  # Target |delta| for long wing strikes (None = N/A)

    # ── Exit: Time ────────────────────────────────────────────────────────────
    "max_dte_exit":       3,  # Close when DTE remaining <= N (0 = expire naturally)
    "hold_days":          21,  # Max calendar days held
}
