"""
Extended Gate primitives G003–G010
Thin, deterministic wrappers around core primitives.
Exact canonical signatures – DO NOT MODIFY.
"""

from __future__ import annotations
import pandas as pd


def compute_trend_strength_gate(
    adx_norm: pd.Series,
    threshold: float,
) -> dict:
    """
    G003 – Trend Strength Gate (A3 / trend-only rules)

    Returns:
        {
            "trend_ok": bool Series
        }
    """
    adx_norm = adx_norm.fillna(0.0)
    trend_ok = (adx_norm >= threshold)
    return {"trend_ok": trend_ok}


def compute_reversion_fuzzy_gate(
    fuzzy_reversion_11: pd.Series,
    threshold: float,
) -> dict:
    """
    G004 – Fuzzy Reversion Gate (B1)

    Returns:
        {
            "reversion_ok": bool Series
        }
    """
    mu = fuzzy_reversion_11.fillna(0.5)
    reversion_ok = (mu >= threshold)
    return {"reversion_ok": reversion_ok}


def compute_chaos_risk_gate(
    chaos_membership: pd.Series,
    max_chaos: float,
) -> dict:
    """
    G005 – Chaos Risk Gate (E3)

    Returns:
        {
            "chaos_block": bool Series
        }
    """
    cm = chaos_membership.fillna(0.0)
    chaos_block = (cm >= max_chaos)
    return {"chaos_block": chaos_block}


def compute_regime_score_gate(
    regime_score: pd.Series,
    min_score: float,
) -> dict:
    """
    G006 – Regime Confidence Gate (C2)

    Returns:
        {
            "regime_ok": bool Series
        }
    """
    rs = regime_score.fillna(0.0)
    regime_ok = (rs >= min_score)
    return {"regime_ok": regime_ok}


def compute_liquidity_gate(
    volume_ratio: pd.Series,
    min_ratio: float = 1.0,
    threshold: float = None,
) -> dict:
    """
    G007 – Liquidity Gate (volume-based)
    
    Returns:
        {
            "liquidity_ok": bool Series
        }
    """
    if threshold is not None:
        min_ratio = threshold
        
    vr = volume_ratio.fillna(1.0)
    liquidity_ok = (vr >= min_ratio)
    return {"liquidity_ok": liquidity_ok}


def compute_spread_liquidity_combo_gate(
    exec_allow: pd.Series,
    liquidity_ok: pd.Series,
) -> dict:
    """
    G008 – Combined Execution/Liquidity Gate

    Returns:
        {
            "exec_liquidity_ok": bool Series
        }
    """
    ea = exec_allow.fillna(True)
    lo = liquidity_ok.fillna(True)
    return {"exec_liquidity_ok": (ea & lo)}


def compute_gap_override_gate(
    risk_override: pd.Series,
) -> dict:
    """
    G009 – Gap Override Exit Gate

    Returns:
        {
            "force_exit": bool Series
        }
    """
    ro = risk_override.fillna(False)
    return {"force_exit": ro.astype(bool)}


def compute_position_size_gate(
    position_size_multiplier: pd.Series,
    min_mult: float,
) -> dict:
    """
    G010 – Position Size Gate (Chaos dampening)

    Returns:
        {
            "size_mult": float Series
        }
    """
    psm = position_size_multiplier.fillna(1.0)
    psm = psm.clip(min_mult, 1.0)
    return {"size_mult": psm}
