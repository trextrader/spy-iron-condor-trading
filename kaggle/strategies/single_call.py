"""
single_call.py — Per-strategy config for single_call (v43 class idx 0).
========================================================================
Templates in this class: long_call, short_call, covered_call.

Tuning notes (run18 backtest, 500 bars, 2025 bullish market):
  - short_call fallback (10 qty, 6 pos): 15 trades, +27.5% PnL, -26% DD
  - short_call (10 qty, 2 pos, 1.5× stop): 7 trades, +0.1% PnL, -15% DD
  - long_call fallback (2 qty, $3k stop): 12 trades, -8.9% PnL, -21% DD (deep ITM too expensive)
  - Current: short_call (1 qty, $3k hard stop) — profit-generating template with capped risk
"""

from strategies._defaults import DEFAULT_CONFIG

CONFIG = {
    **DEFAULT_CONFIG,

    # ── Identity ─────────────────────────────────────────────────────────
    "class_name":           "single_call",
    "class_idx":            0,

    # ── Sizing ───────────────────────────────────────────────────────────
    "max_contracts":        1,          # Minimal exposure: 1 contract only
    "margin_pct":           0.25,       # 25% of max_deploy for qty calc
    "margin_type":          "pct_spot", # Single-leg: % of spot, not width-based
    "margin_spot_pct":      0.02,       # Long call: ~2% of spot (premium est.)
    #   Note: for short_call templates this gets overridden to 0.15 at runtime

    # ── Exit: Stop-Loss ──────────────────────────────────────────────────
    "stop_loss_mult":       1.5,        # 1.5× credit for naked shorts
    "stop_loss_dollar":     500,       # Hard $ cap: $500 max loss/trade (1:5 SL:PT ratio)

    # ── Exit: Profit Target ──────────────────────────────────────────────
    "profit_target":        2500,       # Take profit at $2,500

    # ── Template Preference ──────────────────────────────────────────────
    "preferred_templates":  ["short_call", "covered_call", "long_call"],
    "fallback_template":    "short_call",  # Profitable template + $3k hard stop for DD control

    # ── Entry Gate Overrides ─────────────────────────────────────────────
    "entry_threshold":      None,       # Use global (0.55)
    "pop_threshold":        None,       # Use global (0.50)

    # ── Position Limits ──────────────────────────────────────────────────
    "max_positions":        2,          # Max 2 single_call positions at once
    "cooldown_bars":        10,         # 10 bars (~50 min) between entries
}
