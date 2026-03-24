"""
synthetic_put.py — Synthetic Put
===================================

FAMILY: C) Directional Defined Risk — directional bets with capped loss
CLASS:  custom_multi_leg (index 8)
DIRECTION: Bearish/Hedge

WHAT IT IS:
    Replicates put protection on short stock position. Buy call to cap upside risk on short stock.

LEG STRUCTURE:
#   Leg 1: LONG C rank0

RISK PROFILE:
    Risk:   Limited to premium paid.
    Reward: Profits from downside on short stock.

WHEN IT FIRES (predicate atoms):
    trend_bear & adx_vstrong & gap_high & friction_ok

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
    "template_id":        "synthetic_put",
    "class_name":         "custom_multi_leg",
    "class_idx":          8,
    "family":             "DirectionalDefinedRisk",

    # ── Leg Structure ─────────────────────────────────────────────────────
    "legs": [
        {"type": "C", "side": "LONG", "qty": 1, "rank": 0}
    ],

    # ── Sizing ────────────────────────────────────────────────────────────
    "max_contracts":      7,          # Max contracts per trade
    "margin_pct":         0.5883,        # Max % of buying power to deploy per trade
    "margin_type":        "pct_spot",   # "width" (spread) or "pct_spot" (naked)
    "margin_spot_pct":    0.3048,      # If pct_spot: margin = spot * this * 100

    # ── Exit: Stop-Loss ───────────────────────────────────────────────────
    "stop_loss_mult":     0.0000,   # Close when loss >= N * |credit| * qty * 100
    "stop_loss_dollar":   1270.0000,  # Hard dollar cap per trade (None = use multiplier only)

    # ── Exit: Profit Target ───────────────────────────────────────────────
    "profit_target":      1215.0000,    # Dollar target (None = use 50% of credit)

    # ── Template Preference ───────────────────────────────────────────────
    "fallback_template":  None,        # If no template eligible, use this template_id

    # ── Entry Gate Overrides ──────────────────────────────────────────────
    "entry_threshold":    None,        # Override entry_logit threshold (default 0.55)
    "pop_threshold":      None,        # Override PoP threshold (default 0.50)

    # ── Position Limits ───────────────────────────────────────────────────
    "max_positions":      2,           # Max concurrent open positions for this strategy
    "cooldown_bars":      1,           # Min bars between trades

    # ── Entry: Structure (overrides neural model when not None) ──────────────
    "call_offset_pct":    1.4281,  # % OTM for call strike (None = neural model)
    "put_offset_pct":     1.4455,  # % OTM for put strike  (None = neural model)
    "spread_width":       10.5000,  # Spread width points   (None = neural model / N/A)
    "target_dte":         10,  # Target DTE at entry   (None = neural model)

    # ── Entry: Quality Filter ─────────────────────────────────────────────────
    "max_leg_spread":     0.1818,  # Max bid-ask ratio per leg

    # ── Entry: Delta Targets (template path) ────────────────────────────────
    "short_delta":        0.1527,  # Target |delta| for short (selling) strikes
    "wing_delta":         None,  # Target |delta| for long wing strikes (None = N/A)

    # ── Exit: Time ────────────────────────────────────────────────────────────
    "max_dte_exit":       5,  # Close when DTE remaining <= N (0 = expire naturally)
    "hold_days":          14,  # Max calendar days held
}
