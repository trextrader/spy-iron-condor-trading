"""
guts.py — Guts
=================

FAMILY: B) Long Vol / Convex — buy options, profit from large moves and volatility expansion
CLASS:  custom_multi_leg (index 8)
DIRECTION: Neutral

WHAT IT IS:
    Buy ITM call + ITM put. Like long straddle but ITM. Needs VERY large move to profit.

LEG STRUCTURE:
#   Leg 1: LONG C rank0
#   Leg 2: LONG P rank1

RISK PROFILE:
    Risk:   Limited to total premium paid (expensive).
    Reward: Unlimited in both directions.

WHEN IT FIRES (predicate atoms):
    bw_expanding & adx_strong & ~ivr_high & friction_ok

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
    "template_id":        "guts",
    "class_name":         "custom_multi_leg",
    "class_idx":          8,
    "family":             "LongVolConvex",

    # ── Leg Structure ─────────────────────────────────────────────────────
    "legs": [
        {"type": "C", "side": "LONG", "qty": 1, "rank": 0},
        {"type": "P", "side": "LONG", "qty": 1, "rank": 1}
    ],

    # ── Sizing ────────────────────────────────────────────────────────────
    "max_contracts":      6,          # Max contracts per trade
    "margin_pct":         0.5804,        # Max % of buying power to deploy per trade
    "margin_type":        "pct_spot",   # "width" (spread) or "pct_spot" (naked)
    "margin_spot_pct":    0.1,      # If pct_spot: margin = spot * this * 100

    # ── Exit: Stop-Loss ───────────────────────────────────────────────────
    "stop_loss_mult":     0.0000,   # Close when loss >= N * |credit| * qty * 100
    "stop_loss_dollar":   1200.0000,  # Hard dollar cap per trade (None = use multiplier only)

    # ── Exit: Profit Target ───────────────────────────────────────────────
    "profit_target":      1750.0000,    # Dollar target (None = use 50% of credit)

    # ── Template Preference ───────────────────────────────────────────────
    "fallback_template":  None,        # If no template eligible, use this template_id

    # ── Entry Gate Overrides ──────────────────────────────────────────────
    "entry_threshold":    None,        # Override entry_logit threshold (default 0.55)
    "pop_threshold":      None,        # Override PoP threshold (default 0.50)

    # ── Position Limits ───────────────────────────────────────────────────
    "max_positions":      3,           # Max concurrent open positions for this strategy
    "cooldown_bars":      3,           # Min bars between trades

    # ── Entry: Structure (overrides neural model when not None) ──────────────
    "call_offset_pct":    1.0,  # % OTM for call strike (None = neural model)
    "put_offset_pct":     1.0,  # % OTM for put strike  (None = neural model)
    "spread_width":       18,  # Spread width points   (None = neural model / N/A)
    "target_dte":         7,  # Target DTE at entry   (None = neural model)

    # ── Entry: Quality Filter ─────────────────────────────────────────────────
    "max_leg_spread":     0.15,  # Max bid-ask ratio per leg

    # ── Entry: Delta Targets (template path) ────────────────────────────────
    "short_delta":        0.2723,  # Target |delta| for short (selling) strikes
    "wing_delta":         None,  # Target |delta| for long wing strikes (None = N/A)

    # ── Exit: Time ────────────────────────────────────────────────────────────
    "max_dte_exit":       2,  # Close when DTE remaining <= N (0 = expire naturally)
    "hold_days":          16,  # Max calendar days held
}
