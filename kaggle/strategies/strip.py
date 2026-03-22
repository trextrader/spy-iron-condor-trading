"""
strip.py — Strip
===================

FAMILY: B) Long Vol / Convex — buy options, profit from large moves and volatility expansion
CLASS:  custom_multi_leg (index 8)
DIRECTION: Bearish

WHAT IT IS:
    Long vol with bearish bias — buy 1 call + 2 puts at same strike. Profits more from down moves.

LEG STRUCTURE:
#   Leg 1: LONG C rank0 (1x)
#   Leg 2: LONG P rank0 (2x)

RISK PROFILE:
    Risk:   Limited to total premium paid.
    Reward: Unlimited downside (2x puts), unlimited upside (1x call).

WHEN IT FIRES (predicate atoms):
    trend_bear & bw_expanding & ~ivr_high & friction_ok

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
    "template_id":        "strip",
    "class_name":         "custom_multi_leg",
    "class_idx":          8,
    "family":             "LongVolConvex",

    # ── Leg Structure ─────────────────────────────────────────────────────
    "legs": [
        {"type": "C", "side": "LONG", "qty": 1, "rank": 0},
        {"type": "P", "side": "LONG", "qty": 2, "rank": 0}
    ],

    # ── Sizing ────────────────────────────────────────────────────────────
    "max_contracts":      8,          # Max contracts per trade
    "margin_pct":         0.5160,        # Max % of buying power to deploy per trade
    "margin_type":        "pct_spot",   # "width" (spread) or "pct_spot" (naked)
    "margin_spot_pct":    0.1103,      # If pct_spot: margin = spot * this * 100

    # ── Exit: Stop-Loss ───────────────────────────────────────────────────
    "stop_loss_mult":     0.0000,   # Close when loss >= N * |credit| * qty * 100
    "stop_loss_dollar":   400.0000,  # Hard dollar cap per trade (None = use multiplier only)

    # ── Exit: Profit Target ───────────────────────────────────────────────
    "profit_target":      3000.0000,    # Dollar target (None = use 50% of credit)

    # ── Template Preference ───────────────────────────────────────────────
    "fallback_template":  None,        # If no template eligible, use this template_id

    # ── Entry Gate Overrides ──────────────────────────────────────────────
    "entry_threshold":    None,        # Override entry_logit threshold (default 0.55)
    "pop_threshold":      None,        # Override PoP threshold (default 0.50)

    # ── Position Limits ───────────────────────────────────────────────────
    "max_positions":      15,           # Max concurrent open positions for this strategy
    "cooldown_bars":      14,           # Min bars between trades

    # ── Entry: Structure (overrides neural model when not None) ──────────────
    "call_offset_pct":    0.0363,  # % OTM for call strike (None = neural model)
    "put_offset_pct":     0.0723,  # % OTM for put strike  (None = neural model)
    "spread_width":       11.5000,  # Spread width points   (None = neural model / N/A)
    "target_dte":         28,  # Target DTE at entry   (None = neural model)

    # ── Entry: Quality Filter ─────────────────────────────────────────────────
    "max_leg_spread":     0.1486,  # Max bid-ask ratio per leg

    # ── Entry: Delta Targets (template path) ────────────────────────────────
    "short_delta":        0.4500,  # Target |delta| for short (selling) strikes
    "wing_delta":         None,  # Target |delta| for long wing strikes (None = N/A)

    # ── Exit: Time ────────────────────────────────────────────────────────────
    "max_dte_exit":       5,  # Close when DTE remaining <= N (0 = expire naturally)
    "hold_days":          5,  # Max calendar days held
}
