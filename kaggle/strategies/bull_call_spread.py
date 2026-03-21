"""
bull_call_spread.py — Bull Call Spread
=========================================

FAMILY: C) Directional Defined Risk — directional bets with capped loss
CLASS:  bull_call_spread (index 2)
DIRECTION: Bullish

WHAT IT IS:
    Buy lower call, sell higher call. Bullish debit spread. Capped profit and loss.

LEG STRUCTURE:
#   Leg 1: LONG C rank0
#   Leg 2: SHORT C rank1

RISK PROFILE:
    Risk:   Limited to net debit paid.
    Reward: Limited to spread width − debit.

WHEN IT FIRES (predicate atoms):
    trend_bull & adx_strong & gap_low & friction_ok

HOW IT'S USED IN THE BACKTESTER:
    1. CondorNet v4.3 predicts strategy class 'bull_call_spread' (index 2)
    2. All templates mapped to this class are scored
    3. Eligibility predicate checks market atoms (above)
    4. If eligible, template is scored: 0.40*pop + 0.35*tanh(3*ev) - 0.15*tanh(2*ml) + 0.10*dte_aff
    5. Highest-scoring eligible template wins; legs are built from chain
    6. Fill engine executes the trade with qty, margin, and exit rules from CONFIG below
"""

CONFIG = {
    # ── Identity ──────────────────────────────────────────────────────────
    "template_id":        "bull_call_spread",
    "class_name":         "bull_call_spread",
    "class_idx":          2,
    "family":             "DirectionalDefinedRisk",

    # ── Leg Structure ─────────────────────────────────────────────────────
    "legs": [
        {"type": "C", "side": "LONG", "qty": 1, "rank": 0},
        {"type": "C", "side": "SHORT", "qty": 1, "rank": 1}
    ],

    # ── Sizing ────────────────────────────────────────────────────────────
    "max_contracts":      2,          # Max contracts per trade
    "margin_pct":         0.2125,        # Max % of buying power to deploy per trade
    "margin_type":        "width",   # "width" (spread) or "pct_spot" (naked)
    "margin_spot_pct":    0.5,      # If pct_spot: margin = spot * this * 100

    # ── Exit: Stop-Loss ───────────────────────────────────────────────────
    "stop_loss_mult":     5.0000,   # Close when loss >= N * |credit| * qty * 100
    "stop_loss_dollar":   400.0000,  # Hard dollar cap per trade (None = use multiplier only)

    # ── Exit: Profit Target ───────────────────────────────────────────────
    "profit_target":      400.0000,    # Dollar target (None = use 50% of credit)

    # ── Template Preference ───────────────────────────────────────────────
    "fallback_template":  None,        # If no template eligible, use this template_id

    # ── Entry Gate Overrides ──────────────────────────────────────────────
    "entry_threshold":    None,        # Override entry_logit threshold (default 0.55)
    "pop_threshold":      None,        # Override PoP threshold (default 0.50)

    # ── Position Limits ───────────────────────────────────────────────────
    "max_positions":      11,           # Max concurrent open positions for this strategy
    "cooldown_bars":      6,           # Min bars between trades

    # ── Entry: Structure (overrides neural model when not None) ──────────────
    "call_offset_pct":    1.5,  # % OTM for call strike (None = neural model)
    "put_offset_pct":     None,  # % OTM for put strike  (None = neural model)
    "spread_width":       5.0000,  # Spread width points   (None = neural model / N/A)
    "target_dte":         13,  # Target DTE at entry   (None = neural model)

    # ── Entry: Quality Filter ─────────────────────────────────────────────────
    "max_leg_spread":     0.1125,  # Max bid-ask ratio per leg

    # ── Entry: Delta Targets (template path) ────────────────────────────────
    "short_delta":        0.1500,  # Target |delta| for short (selling) strikes
    "wing_delta":         0.1,  # Target |delta| for long wing strikes (None = N/A)

    # ── Exit: Time ────────────────────────────────────────────────────────────
    "max_dte_exit":       5,  # Close when DTE remaining <= N (0 = expire naturally)
    "hold_days":          14,  # Max calendar days held
}
