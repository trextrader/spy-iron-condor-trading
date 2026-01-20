# intelligence/primitives/fuzzy.py
"""
Fuzzy logic primitives (F001)
Exact canonical signatures - DO NOT MODIFY
"""

import pandas as pd


def compute_fuzzy_reversion_score_11(
    mu_mtf: pd.Series,
    mu_ivr: pd.Series,
    mu_vix: pd.Series,
    mu_rsi: pd.Series,
    mu_stoch: pd.Series,
    mu_adx: pd.Series,
    mu_sma: pd.Series,
    mu_psar: pd.Series,
    mu_bb: pd.Series,
    mu_bbsqueeze: pd.Series,
    mu_vol: pd.Series,
    w_mtf: float = 0.25,
    w_ivr: float = 0.15,
    w_vix: float = 0.10,
    w_rsi: float = 0.15,
    w_stoch: float = 0.05,
    w_adx: float = 0.05,
    w_sma: float = 0.05,
    w_psar: float = 0.10,
    w_bb: float = 0.05,
    w_bbsqueeze: float = 0.03,
    w_vol: float = 0.02,

) -> pd.Series:
    """
    F001 - 11-Factor Fuzzy Reversion Score (Rule B1 v2.0)

    Returns Series: fuzzy_score in [0, 1]

    Weights (default):
        MTF=0.25, IVR=0.15, VIX=0.10, RSI=0.15, Stoch=0.05,
        ADX=0.05, SMA=0.05, PSAR=0.10, BB=0.05, BBsqueeze=0.03, Vol=0.02
    """
    weights_sum = (
        w_mtf
        + w_ivr
        + w_vix
        + w_rsi
        + w_stoch
        + w_adx
        + w_sma
        + w_psar
        + w_bb
        + w_bbsqueeze
        + w_vol
    )

    if abs(weights_sum - 1.0) > 1e-6:
        # normalize defensively
        w_mtf /= weights_sum
        w_ivr /= weights_sum
        w_vix /= weights_sum
        w_rsi /= weights_sum
        w_stoch /= weights_sum
        w_adx /= weights_sum
        w_sma /= weights_sum
        w_psar /= weights_sum
        w_bb /= weights_sum
        w_bbsqueeze /= weights_sum
        w_vol /= weights_sum

    fs = (
        w_mtf * mu_mtf.fillna(0.0)
        + w_ivr * mu_ivr.fillna(0.0)
        + w_vix * mu_vix.fillna(0.0)
        + w_rsi * mu_rsi.fillna(0.0)
        + w_stoch * mu_stoch.fillna(0.0)
        + w_adx * mu_adx.fillna(0.0)
        + w_sma * mu_sma.fillna(0.0)
        + w_psar * mu_psar.fillna(0.0)
        + w_bb * mu_bb.fillna(0.0)
        + w_bbsqueeze * mu_bbsqueeze.fillna(0.0)
        + w_vol * mu_vol.fillna(0.0)
    )

    return fs.clip(0.0, 1.0)
