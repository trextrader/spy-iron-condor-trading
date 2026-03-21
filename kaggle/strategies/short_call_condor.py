"""
short_call_condor.py — Short Call Condor
===========================================

FAMILY: A) Short Vol / Income — sell premium, profit from theta decay and range-bound markets
CLASS:  custom_multi_leg (index 8)
DIRECTION: Bearish

WHAT IT IS:
    Sell outer call wings, buy inner call body. Directional short vol play.

LEG STRUCTURE:
#   Leg 1: SHORT C rank0
#   Leg 2: LONG C rank1
#   Leg 3: LONG C rank2
#   Leg 4: SHORT C rank3

RISK PROFILE:
    Risk:   Defined: max loss = inner width − credit.
    Reward: Limited to net credit.

WHEN IT FIRES (predicate atoms):
    trend_bear & ivr_high & adx_strong & friction_ok

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
    "template_id":        "short_call_condor",
    "class_name":         "custom_multi_leg",
    "class_idx":          8,
    "family":             "ShortVolIncome",

    # ── Leg Structure ─────────────────────────────────────────────────────
    "legs": [
        {"type": "C", "side": "SHORT", "qty": 1, "rank": 0},
        {"type": "C", "side": "LONG", "qty": 1, "rank": 1},
        {"type": "C", "side": "LONG", "qty": 1, "rank": 2},
        {"type": "C", "side": "SHORT", "qty": 1, "rank": 3}
    ],

    # ── Sizing ────────────────────────────────────────────────────────────
    "max_contracts":      13,          # Max contracts per trade
    "margin_pct":         0.5545,        # Max % of buying power to deploy per trade
    "margin_type":        "width",   # "width" (spread) or "pct_spot" (naked)
    "margin_spot_pct":    0.4825,      # If pct_spot: margin = spot * this * 100

    # ── Exit: Stop-Loss ───────────────────────────────────────────────────
    "stop_loss_mult":     10.0000,   # Close when loss >= N * |credit| * qty * 100
    "stop_loss_dollar":   400.0000,  # Hard dollar cap per trade (None = use multiplier only)

    # ── Exit: Profit Target ───────────────────────────────────────────────
    "profit_target":      1165.0000,    # Dollar target (None = use 50% of credit)

    # ── Template Preference ───────────────────────────────────────────────
    "fallback_template":  None,        # If no template eligible, use this template_id

    # ── Entry Gate Overrides ──────────────────────────────────────────────
    "entry_threshold":    None,        # Override entry_logit threshold (default 0.55)
    "pop_threshold":      None,        # Override PoP threshold (default 0.50)

    # ── Position Limits ───────────────────────────────────────────────────
    "max_positions":      8,           # Max concurrent open positions for this strategy
    "cooldown_bars":      7,           # Min bars between trades

    # ── Entry: Structure (overrides neural model when not None) ──────────────
    "call_offset_pct":    1.4086,  # % OTM for call strike (None = neural model)
    "put_offset_pct":     None,  # % OTM for put strike  (None = neural model)
    "spread_width":       8.5000,  # Spread width points   (None = neural model / N/A)
    "target_dte":         14,  # Target DTE at entry   (None = neural model)

    # ── Entry: Quality Filter ─────────────────────────────────────────────────
    "max_leg_spread":     0.0921,  # Max bid-ask ratio per leg

    # ── Entry: Delta Targets (template path) ────────────────────────────────
    "short_delta":        0.2055,  # Target |delta| for short (selling) strikes
    "wing_delta":         0.0903,  # Target |delta| for long wing strikes (None = N/A)

    # ── Exit: Time ────────────────────────────────────────────────────────────
    "max_dte_exit":       1,  # Close when DTE remaining <= N (0 = expire naturally)
    "hold_days":          11,  # Max calendar days held
}
