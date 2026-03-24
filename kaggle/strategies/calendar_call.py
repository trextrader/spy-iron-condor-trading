"""
calendar_call.py — Calendar Call
===================================

FAMILY: D) Term Structure — exploit differential theta between near and far expirations
CLASS:  custom_multi_leg (index 8)
DIRECTION: Neutral

WHAT IT IS:
    Sell near-term call + buy same-strike far-dated call. Exploits faster theta decay of near option.

LEG STRUCTURE:
#   Leg 1: SHORT C rank0 [near DTE]
#   Leg 2: LONG C rank0 [far DTE]

RISK PROFILE:
    Risk:   Limited to net debit paid.
    Reward: Profits from near option decaying faster than far option.

WHEN IT FIRES (predicate atoms):
    adx_weak & consol_high & rsi_neutral & regime_stable & friction_ok

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
    "template_id":        "calendar_call",
    "class_name":         "custom_multi_leg",
    "class_idx":          8,
    "family":             "TermStructure",

    # ── Leg Structure ─────────────────────────────────────────────────────
    "legs": [
        {"type": "C", "side": "SHORT", "qty": 1, "rank": 0},
        {"type": "C", "side": "LONG", "qty": 1, "rank": 0}
    ],

    # ── Sizing ────────────────────────────────────────────────────────────
    "max_contracts":      1,          # Max contracts per trade
    "margin_pct":         0.1975,        # Max % of buying power to deploy per trade
    "margin_type":        "pct_spot",   # "width" (spread) or "pct_spot" (naked)
    "margin_spot_pct":    0.3225,      # If pct_spot: margin = spot * this * 100

    # ── Exit: Stop-Loss ───────────────────────────────────────────────────
    "stop_loss_mult":     5.0000,   # Close when loss >= N * |credit| * qty * 100
    "stop_loss_dollar":   1180.0000,  # Hard dollar cap per trade (None = use multiplier only)

    # ── Exit: Profit Target ───────────────────────────────────────────────
    "profit_target":      400.0000,    # Dollar target (None = use 50% of credit)

    # ── Template Preference ───────────────────────────────────────────────
    "fallback_template":  None,        # If no template eligible, use this template_id

    # ── Entry Gate Overrides ──────────────────────────────────────────────
    "entry_threshold":    None,        # Override entry_logit threshold (default 0.55)
    "pop_threshold":      None,        # Override PoP threshold (default 0.50)

    # ── Position Limits ───────────────────────────────────────────────────
    "max_positions":      1,           # Max concurrent open positions for this strategy
    "cooldown_bars":      16,           # Min bars between trades

    # ── Entry: Structure (overrides neural model when not None) ──────────────
    "call_offset_pct":    0.1000,  # % OTM for call strike (None = neural model)
    "put_offset_pct":     None,  # % OTM for put strike  (None = neural model)
    "spread_width":       18.0000,  # Spread width points   (None = neural model / N/A)
    "target_dte":         7,  # Target DTE at entry   (None = neural model)

    # ── Entry: Quality Filter ─────────────────────────────────────────────────
    "max_leg_spread":     0.1842,  # Max bid-ask ratio per leg

    # ── Entry: Delta Targets (template path) ────────────────────────────────
    "short_delta":        0.2550,  # Target |delta| for short (selling) strikes
    "wing_delta":         None,  # Target |delta| for long wing strikes (None = N/A)

    # ── Exit: Time ────────────────────────────────────────────────────────────
    "max_dte_exit":       1,  # Close when DTE remaining <= N (0 = expire naturally)
    "hold_days":          21,  # Max calendar days held
}
