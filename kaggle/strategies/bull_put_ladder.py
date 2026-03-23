"""
bull_put_ladder.py — Bull Put Ladder
=======================================

FAMILY: C) Directional Defined Risk — directional bets with capped loss
CLASS:  custom_multi_leg (index 8)
DIRECTION: Very bearish/bullish

WHAT IT IS:
    Two-tailed play. Very bearish or very bullish if stock moves dramatically.

LEG STRUCTURE:
#   Leg 1: LONG P rank0
#   Leg 2: LONG P rank1
#   Leg 3: SHORT P rank2

RISK PROFILE:
    Risk:   Limited loss between strikes.
    Reward: Unlimited profit on extreme downside move.

WHEN IT FIRES (predicate atoms):
    adx_vstrong & bw_expanding & friction_ok

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
    "template_id":        "bull_put_ladder",
    "class_name":         "custom_multi_leg",
    "class_idx":          8,
    "family":             "DirectionalDefinedRisk",

    # ── Leg Structure ─────────────────────────────────────────────────────
    "legs": [
        {"type": "P", "side": "LONG", "qty": 1, "rank": 0},
        {"type": "P", "side": "LONG", "qty": 1, "rank": 1},
        {"type": "P", "side": "SHORT", "qty": 1, "rank": 2}
    ],

    # ── Sizing ────────────────────────────────────────────────────────────
    "max_contracts":      0,          # Max contracts per trade
    "margin_pct":         0.3896,        # Max % of buying power to deploy per trade
    "margin_type":        "pct_spot",   # "width" (spread) or "pct_spot" (naked)
    "margin_spot_pct":    0.1,      # If pct_spot: margin = spot * this * 100

    # ── Exit: Stop-Loss ───────────────────────────────────────────────────
    "stop_loss_mult":     10.0000,   # Close when loss >= N * |credit| * qty * 100
    "stop_loss_dollar":   950.0000,  # Hard dollar cap per trade (None = use multiplier only)

    # ── Exit: Profit Target ───────────────────────────────────────────────
    "profit_target":      2750.0000,    # Dollar target (None = use 50% of credit)

    # ── Template Preference ───────────────────────────────────────────────
    "fallback_template":  None,        # If no template eligible, use this template_id

    # ── Entry Gate Overrides ──────────────────────────────────────────────
    "entry_threshold":    None,        # Override entry_logit threshold (default 0.55)
    "pop_threshold":      None,        # Override PoP threshold (default 0.50)

    # ── Position Limits ───────────────────────────────────────────────────
    "max_positions":      5,           # Max concurrent open positions for this strategy
    "cooldown_bars":      10,           # Min bars between trades

    # ── Entry: Structure (overrides neural model when not None) ──────────────
    "call_offset_pct":    None,  # % OTM for call strike (None = neural model)
    "put_offset_pct":     1.5,  # % OTM for put strike  (None = neural model)
    "spread_width":       13.0000,  # Spread width points   (None = neural model / N/A)
    "target_dte":         8,  # Target DTE at entry   (None = neural model)

    # ── Entry: Quality Filter ─────────────────────────────────────────────────
    "max_leg_spread":     0.2250,  # Max bid-ask ratio per leg

    # ── Entry: Delta Targets (template path) ────────────────────────────────
    "short_delta":        0.1500,  # Target |delta| for short (selling) strikes
    "wing_delta":         0.1,  # Target |delta| for long wing strikes (None = N/A)

    # ── Exit: Time ────────────────────────────────────────────────────────────
    "max_dte_exit":       2,  # Close when DTE remaining <= N (0 = expire naturally)
    "hold_days":          12,  # Max calendar days held
}
