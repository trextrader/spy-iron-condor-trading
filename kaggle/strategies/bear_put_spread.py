"""
bear_put_spread.py — Bear Put Spread
=======================================

FAMILY: C) Directional Defined Risk — directional bets with capped loss
CLASS:  bear_put_spread (index 3)
DIRECTION: Bearish

WHAT IT IS:
    Buy higher put, sell lower put. Bearish debit spread. Capped profit and loss.

LEG STRUCTURE:
#   Leg 1: LONG P rank1
#   Leg 2: SHORT P rank0

RISK PROFILE:
    Risk:   Limited to net debit paid.
    Reward: Limited to spread width − debit.

WHEN IT FIRES (predicate atoms):
    trend_bear & adx_strong & friction_ok

HOW IT'S USED IN THE BACKTESTER:
    1. CondorNet v4.3 predicts strategy class 'bear_put_spread' (index 3)
    2. All templates mapped to this class are scored
    3. Eligibility predicate checks market atoms (above)
    4. If eligible, template is scored: 0.40*pop + 0.35*tanh(3*ev) - 0.15*tanh(2*ml) + 0.10*dte_aff
    5. Highest-scoring eligible template wins; legs are built from chain
    6. Fill engine executes the trade with qty, margin, and exit rules from CONFIG below
"""

CONFIG = {
    # ── Identity ──────────────────────────────────────────────────────────
    "template_id":        "bear_put_spread",
    "class_name":         "bear_put_spread",
    "class_idx":          3,
    "family":             "DirectionalDefinedRisk",

    # ── Leg Structure ─────────────────────────────────────────────────────
    "legs": [
        {"type": "P", "side": "LONG", "qty": 1, "rank": 1},
        {"type": "P", "side": "SHORT", "qty": 1, "rank": 0}
    ],

    # ── Sizing ────────────────────────────────────────────────────────────
    "max_contracts":      5,          # Max contracts per trade
    "margin_pct":         0.4437,        # Max % of buying power to deploy per trade
    "margin_type":        "width",   # "width" (spread) or "pct_spot" (naked)
    "margin_spot_pct":    0.6000,      # If pct_spot: margin = spot * this * 100

    # ── Exit: Stop-Loss ───────────────────────────────────────────────────
    "stop_loss_mult":     10.0000,   # Close when loss >= N * |credit| * qty * 100
    "stop_loss_dollar":   1105.0000,  # Hard dollar cap per trade (None = use multiplier only)

    # ── Exit: Profit Target ───────────────────────────────────────────────
    "profit_target":      2365.0000,    # Dollar target (None = use 50% of credit)

    # ── Template Preference ───────────────────────────────────────────────
    "fallback_template":  None,        # If no template eligible, use this template_id

    # ── Entry Gate Overrides ──────────────────────────────────────────────
    "entry_threshold":    None,        # Override entry_logit threshold (default 0.55)
    "pop_threshold":      None,        # Override PoP threshold (default 0.50)

    # ── Position Limits ───────────────────────────────────────────────────
    "max_positions":      12,           # Max concurrent open positions for this strategy
    "cooldown_bars":      1,           # Min bars between trades

    # ── Entry: Structure (overrides neural model when not None) ──────────────
    "call_offset_pct":    None,  # % OTM for call strike (None = neural model)
    "put_offset_pct":     1.4000,  # % OTM for put strike  (None = neural model)
    "spread_width":       5.0000,  # Spread width points   (None = neural model / N/A)
    "target_dte":         24,  # Target DTE at entry   (None = neural model)

    # ── Entry: Quality Filter ─────────────────────────────────────────────────
    "max_leg_spread":     0.0750,  # Max bid-ask ratio per leg

    # ── Entry: Delta Targets (template path) ────────────────────────────────
    "short_delta":        0.2218,  # Target |delta| for short (selling) strikes
    "wing_delta":         0.1879,  # Target |delta| for long wing strikes (None = N/A)

    # ── Exit: Time ────────────────────────────────────────────────────────────
    "max_dte_exit":       1,  # Close when DTE remaining <= N (0 = expire naturally)
    "hold_days":          14,  # Max calendar days held
}
