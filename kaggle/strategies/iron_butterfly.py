"""
iron_butterfly.py — Iron Butterfly
=====================================

FAMILY: A) Short Vol / Income — sell premium, profit from theta decay and range-bound markets
CLASS:  custom_multi_leg (index 8)
DIRECTION: Neutral

WHAT IT IS:
    Tighter than IC; short P and short C at SAME strike; higher max profit but narrower range.

LEG STRUCTURE:
#   Leg 1: LONG P rank0
#   Leg 2: SHORT P rank1
#   Leg 3: SHORT C rank1
#   Leg 4: LONG C rank2

RISK PROFILE:
    Risk:   Defined: max loss = wing width − credit.
    Reward: Limited to net credit (higher than IC due to ATM shorts).

WHEN IT FIRES (predicate atoms):
    ivr_high & consol_vhigh & rsi_neutral & regime_stable & friction_ok

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
    "template_id":        "iron_butterfly",
    "class_name":         "custom_multi_leg",
    "class_idx":          8,
    "family":             "ShortVolIncome",

    # ── Leg Structure ─────────────────────────────────────────────────────
    "legs": [
        {"type": "P", "side": "LONG", "qty": 1, "rank": 0},
        {"type": "P", "side": "SHORT", "qty": 1, "rank": 1},
        {"type": "C", "side": "SHORT", "qty": 1, "rank": 1},
        {"type": "C", "side": "LONG", "qty": 1, "rank": 2}
    ],

    # ── Sizing ────────────────────────────────────────────────────────────
    "max_contracts":      5,          # Max contracts per trade
    "margin_pct":         0.50,        # Max % of buying power to deploy per trade
    "margin_type":        "width",   # "width" (spread) or "pct_spot" (naked)
    "margin_spot_pct":    0.5,      # If pct_spot: margin = spot * this * 100

    # ── Exit: Stop-Loss ───────────────────────────────────────────────────
    "stop_loss_mult":     2.0,   # Close when loss >= N * |credit| * qty * 100
    "stop_loss_dollar":   1025.0000,  # Hard dollar cap per trade (BO range 500–1500, step 250)

    # ── Exit: Profit Target ───────────────────────────────────────────────
    "profit_target":      700.0000,   # Dollar target (BO range 500–2000, step 250)

    # ── Template Preference ───────────────────────────────────────────────
    "fallback_template":  None,        # If no template eligible, use this template_id

    # ── Entry Gate Overrides ──────────────────────────────────────────────
    "entry_threshold":    None,        # Override entry_logit threshold (default 0.55)
    "pop_threshold":      None,        # Override PoP threshold (default 0.50)

    # ── Position Limits ───────────────────────────────────────────────────
    "max_positions":      9,           # Max concurrent open positions for this strategy
    "cooldown_bars":      2,           # Min bars between trades

    # ── Entry: Structure (overrides neural model when not None) ──────────────
    "call_offset_pct":    0.0,  # % OTM for call strike (None = neural model)
    "put_offset_pct":     0.0,  # % OTM for put strike  (None = neural model)
    "spread_width":       11.0000,  # Spread width points   (BO range 5–20, step 5)
    "target_dte":         7,  # Target DTE at entry   (BO range 7–21, step 7)

    # ── Entry: Quality Filter ─────────────────────────────────────────────────
    "max_leg_spread":     0.15,  # Max bid-ask ratio per leg

    # ── Entry: Delta Targets (template path) ────────────────────────────────
    "short_delta":        0.4833,  # Target |delta| for short (selling) strikes
    "wing_delta":         0.2754,  # Target |delta| for long wing strikes (None = N/A)

    # ── Exit: Time ────────────────────────────────────────────────────────────
    "max_dte_exit":       7,  # Close when DTE remaining <= N (0 = expire naturally)
    "hold_days":          5,  # Max calendar days held
}
