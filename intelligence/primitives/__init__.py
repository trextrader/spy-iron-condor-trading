# intelligence/primitives/__init__.py
"""
Canonical primitives for CondorBrain rule execution.
All 14 primitives with schema-locked signatures.
"""

from .bands import (
    compute_dynamic_bollinger_bands,
    compute_bandwidth_percentile_and_expansion,
    compute_volume_ratio,
    compute_spread_friction_ratio,
    compute_gap_risk_score,
    compute_iv_confidence,
    compute_mtf_consensus,
)

from .momentum import (
    compute_vol_normalized_macd,
    compute_vol_normalized_adx,
    compute_adx_trend_metrics,
    compute_dynamic_rsi,
    compute_psar_flip_membership,
)

from .topology import (
    compute_beta1_regime_score,
    compute_chaos_membership,
    compute_curvature_proxy,
)

from .fuzzy import (
    compute_fuzzy_reversion_score_11,
)

from .gates_extended import (
    compute_trend_strength_gate,
    compute_reversion_fuzzy_gate,
    compute_chaos_risk_gate,
    compute_regime_score_gate,
    compute_liquidity_gate,
    compute_spread_liquidity_combo_gate,
    compute_gap_override_gate,
    compute_position_size_gate,
)

from .signals import (
    compute_macd_trend_signal,
    compute_bb_breakout_signal,
    compute_bb_reversion_signal,
    compute_band_squeeze_breakout_signal,
    compute_rsi_reversion_signal,
    compute_rsi_divergence_signal,
    compute_mtf_alignment_signal,
    compute_fuzzy_reversion_signal,
    compute_gap_event_signal,
    compute_chaos_dampening_signal,
    compute_chaos_detection_signal,
    compute_regime_shift_signal,
    compute_bb_squeeze_signal,
    compute_volume_spike_signal,
    compute_liquidity_exec_signal,
    compute_trend_follow_entry_signal,
    compute_reversion_vs_trend_conflict_signal,
    compute_spread_block_signal,
    compute_gap_exit_signal,
    compute_swing_high_low_signal,
    compute_size_adjustment_signal,
    compute_final_execution_signal,
    compute_bandwidth_expansion_signal,
)

__all__ = [
    # Bands / microstructure (P001-P007)
    "compute_dynamic_bollinger_bands",
    "compute_bandwidth_percentile_and_expansion",
    "compute_volume_ratio",
    "compute_spread_friction_ratio",
    "compute_gap_risk_score",
    "compute_iv_confidence",
    "compute_mtf_consensus",
    # Momentum (M001-M004)
    "compute_vol_normalized_macd",
    "compute_vol_normalized_adx",
    "compute_adx_trend_metrics",
    "compute_dynamic_rsi",
    "compute_psar_flip_membership",
    # Topology (T001-T002)
    "compute_beta1_regime_score",
    "compute_chaos_membership",
    "compute_curvature_proxy",
    # Fuzzy (F001)
    "compute_fuzzy_reversion_score_11",
    # Gates (G003-G010)
    "compute_trend_strength_gate",
    "compute_reversion_fuzzy_gate",
    "compute_chaos_risk_gate",
    "compute_regime_score_gate",
    "compute_liquidity_gate",
    "compute_spread_liquidity_combo_gate",
    "compute_gap_override_gate",
    "compute_position_size_gate",
    # Signals (S001-S015)
    "compute_macd_trend_signal",
    "compute_bb_breakout_signal",
    "compute_bb_reversion_signal",
    "compute_band_squeeze_breakout_signal",
    "compute_rsi_reversion_signal",
    "compute_rsi_divergence_signal",
    "compute_mtf_alignment_signal",
    "compute_fuzzy_reversion_signal",
    "compute_gap_event_signal",
    "compute_chaos_dampening_signal",
    "compute_chaos_detection_signal",
    "compute_regime_shift_signal",
    "compute_bb_squeeze_signal",
    "compute_volume_spike_signal",
    "compute_liquidity_exec_signal",
    "compute_trend_follow_entry_signal",
    "compute_reversion_vs_trend_conflict_signal",
    "compute_spread_block_signal",
    "compute_gap_exit_signal",
    "compute_swing_high_low_signal",
    "compute_size_adjustment_signal",
    "compute_final_execution_signal",
]
