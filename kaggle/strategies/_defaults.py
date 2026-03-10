"""
_defaults.py — Default configuration inherited by all strategies
================================================================

These defaults are merged with per-strategy CONFIG overrides.
Any key not specified in a strategy's CONFIG inherits from here.
"""

DEFAULT_CONFIG = {
    # ── Sizing ────────────────────────────────────────────────────────────
    "max_contracts":      10,          # Max contracts per trade
    "margin_pct":         0.50,        # Max % of buying power to deploy per trade
    "margin_type":        "width",     # "width" (spread) or "pct_spot" (naked)
    "margin_spot_pct":    0.15,        # If pct_spot: margin = spot * this * 100

    # ── Exit: Stop-Loss ───────────────────────────────────────────────────
    "stop_loss_mult":     2.0,         # Close when loss >= N × |credit| × qty × 100
    "stop_loss_dollar":   None,        # Hard dollar cap per trade (None = use multiplier only)

    # ── Exit: Profit Target ───────────────────────────────────────────────
    "profit_target":      None,        # Dollar target (None = use 50% of credit)

    # ── Exit: Time ────────────────────────────────────────────────────────
    "max_dte_exit":       0,           # Close when DTE remaining <= this (0 = let expire naturally)
    "hold_days":          30,          # Max calendar days to hold regardless of DTE

    # ── Entry: Structure (overrides neural model pol[0..3] when not None) ─
    "call_offset_pct":    None,        # Call strike offset % from spot (None = use neural model)
    "put_offset_pct":     None,        # Put strike offset % from spot (None = use neural model)
    "spread_width":       None,        # Spread width in points (None = use neural model; N/A single-leg)
    "target_dte":         None,        # Target DTE for entries (None = use neural model)

    # ── Entry: Quality Filter ─────────────────────────────────────────────
    "max_leg_spread":     0.15,        # Max bid-ask spread ratio per leg (chain quality gate)

    # ── Template Preference ───────────────────────────────────────────────
    "fallback_template":  None,        # If no template eligible, use this template_id

    # ── Entry Gate Overrides ──────────────────────────────────────────────
    "entry_threshold":    None,        # Override entry_logit threshold (default 0.55)
    "pop_threshold":      None,        # Override PoP threshold (default 0.50)

    # ── Position Limits ───────────────────────────────────────────────────
    "max_positions":      6,           # Max concurrent open positions for this strategy
    "cooldown_bars":      5,           # Min bars between trades
}
