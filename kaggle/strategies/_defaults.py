"""
_defaults.py — Default strategy config inherited by all strategies.
=================================================================
Any key not overridden in a per-strategy config file uses these values.
"""

DEFAULT_CONFIG = {
    # ── Identity ────────────────────────────────────────────────────────────
    "class_name":           None,       # e.g. "single_call" (set per-strategy)
    "class_idx":            None,       # e.g. 0 (set per-strategy)

    # ── Sizing ──────────────────────────────────────────────────────────────
    "max_contracts":        10,         # Hard cap on contracts per trade
    "margin_pct":           0.50,       # % of max_deploy used for qty calculation
    "margin_type":          "width",    # "pct_spot" | "width" | "premium"
    "margin_spot_pct":      0.15,       # For margin_type="pct_spot": fraction of spot
    #   - Short single-leg: 0.15 (15% of spot, Reg-T approx)
    #   - Long single-leg:  0.02 (2% of spot, premium estimate)

    # ── Exit: Stop-Loss ─────────────────────────────────────────────────────
    "stop_loss_mult":       2.0,        # ×credit stop-loss multiplier
    "stop_loss_dollar":     None,       # Fixed $ stop override (None = use multiplier)

    # ── Exit: Profit Target ─────────────────────────────────────────────────
    "profit_target":        None,       # $ profit target (None = 50% of credit default)

    # ── Template Preference ─────────────────────────────────────────────────
    "preferred_templates":  None,       # Ordered list of template_ids to prefer
    "fallback_template":    None,       # Template to use when all predicates fail

    # ── Entry Gate Overrides ────────────────────────────────────────────────
    "entry_threshold":      None,       # Override V43_ENTRY_THRESHOLD (None = global)
    "pop_threshold":        None,       # Override V43_POP_THRESHOLD (None = global)
    "abstain_threshold":    None,       # Override ABSTAIN_CONFIDENCE_THRESHOLD

    # ── Position Limits ─────────────────────────────────────────────────────
    "max_positions":        None,       # Per-strategy position limit (None = global)
    "cooldown_bars":        None,       # Override MIN_BARS_BETWEEN_TRADES
}
