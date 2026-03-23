"""
strap.py — Strap
===================

FAMILY: B) Long Vol / Convex — buy options, profit from large moves and volatility expansion
CLASS:  custom_multi_leg (index 8)
DIRECTION: Bullish

WHAT IT IS:
    Long vol with bullish bias — buy 2 calls + 1 put at same strike. Profits more from up moves.

LEG STRUCTURE:
#   Leg 1: LONG C rank0 (2x)
#   Leg 2: LONG P rank0 (1x)

RISK PROFILE:
    Risk:   Limited to total premium paid.
    Reward: Unlimited upside (2x calls), unlimited downside (1x put).

WHEN IT FIRES (predicate atoms):
    trend_bull & bw_expanding & ~ivr_high & friction_ok

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
    "template_id":        "strap",
    "class_name":         "custom_multi_leg",
    "class_idx":          8,
    "family":             "LongVolConvex",

    # ── Leg Structure ─────────────────────────────────────────────────────
    "legs": [
        {"type": "C", "side": "LONG", "qty": 2, "rank": 0},
        {"type": "P", "side": "LONG", "qty": 1, "rank": 0}
    ],

    # ── Sizing ────────────────────────────────────────────────────────────
    "max_contracts":      0,          # Max contracts per trade
    "margin_pct":         0.5641,        # Max % of buying power to deploy per trade
    "margin_type":        "pct_spot",   # "width" (spread) or "pct_spot" (naked)
    "margin_spot_pct":    0.0251,      # If pct_spot: margin = spot * this * 100

    # ── Exit: Stop-Loss ───────────────────────────────────────────────────
    "stop_loss_mult":     10.0000,   # Close when loss >= N * |credit| * qty * 100
    "stop_loss_dollar":   1300.0000,  # Hard dollar cap per trade (None = use multiplier only)

    # ── Exit: Profit Target ───────────────────────────────────────────────
    "profit_target":      900.0000,    # Dollar target (None = use 50% of credit)

    # ── Template Preference ───────────────────────────────────────────────
    "fallback_template":  None,        # If no template eligible, use this template_id

    # ── Entry Gate Overrides ──────────────────────────────────────────────
    "entry_threshold":    None,        # Override entry_logit threshold (default 0.55)
    "pop_threshold":      None,        # Override PoP threshold (default 0.50)

    # ── Position Limits ───────────────────────────────────────────────────
    "max_positions":      6,           # Max concurrent open positions for this strategy
    "cooldown_bars":      2,           # Min bars between trades

    # ── Entry: Structure (overrides neural model when not None) ──────────────
    "call_offset_pct":    -0.1000,  # % OTM for call strike (None = neural model)
    "put_offset_pct":     -0.1000,  # % OTM for put strike  (None = neural model)
    "spread_width":       8.5000,  # Spread width points   (None = neural model / N/A)
    "target_dte":         17,  # Target DTE at entry   (None = neural model)

    # ── Entry: Quality Filter ─────────────────────────────────────────────────
    "max_leg_spread":     0.0750,  # Max bid-ask ratio per leg

    # ── Entry: Delta Targets (template path) ────────────────────────────────
    "short_delta":        0.2481,  # Target |delta| for short (selling) strikes
    "wing_delta":         None,  # Target |delta| for long wing strikes (None = N/A)

    # ── Exit: Time ────────────────────────────────────────────────────────────
    "max_dte_exit":       1,  # Close when DTE remaining <= N (0 = expire naturally)
    "hold_days":          17,  # Max calendar days held
}
