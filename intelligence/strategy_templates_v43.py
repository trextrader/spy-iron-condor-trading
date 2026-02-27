"""
strategy_templates_v43.py — CondorNet v4.3 Strategy Catalog
============================================================
Full catalog: 50+ strategies mapped to v4.3 STRATEGY_TYPES classes,
with vectorized predicate eligibility masks, leg templates, and risk
economics multipliers.

Single source of truth for:
  - Leg structure (opt_type, side, qty, strike_rank) per strategy
  - Strategy family classification (ShortVolIncome | LongVolConvex |
    DirectionalDefinedRisk | TermStructure)
  - Vectorized eligibility predicates: compute_predicate_atoms(df) → dict,
    then template.get_eligibility_mask(atoms) → np.ndarray[bool, N]
  - Per-strategy economics multipliers applied to the IC-baseline pop/ev/max_loss:
      pop_out  = clip(base_pop  * pop_scale  + pop_offset,  0.01, 0.99)
      ev_out   = base_ev  * ev_scale          (may be negative for debit strategies)
      ml_out   = base_ml  * max_loss_scale

Predicate design philosophy:
  - Atoms are precomputed once across the entire df (vectorized numpy).
  - Each template eligibility is a boolean composition of atoms.
  - Atoms use only columns confirmed present in v4.3 TF CSVs (schema_v43).
  - Mutual exclusion by IV regime: LOW → long structures; HIGH → short vol;
    NEUTRAL → IC and directional credit.

Author: CondorNet v4.3 Strategy Scaffold
Date: 2026-02-26
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple


# =============================================================================
# FAMILY CONSTANTS
# =============================================================================

FAMILY_SHORT_VOL_INCOME    = "ShortVolIncome"
FAMILY_LONG_VOL_CONVEX     = "LongVolConvex"
FAMILY_DIRECTIONAL_DEFINED = "DirectionalDefinedRisk"
FAMILY_TERM_STRUCTURE      = "TermStructure"


# =============================================================================
# LEG TEMPLATE
# =============================================================================

class LegTemplate:
    """Describes one leg of a multi-leg options strategy."""
    __slots__ = ("opt_type", "side", "qty", "strike_rank", "expiry_mode")

    def __init__(
        self,
        opt_type: str,      # "C" or "P"
        side: str,          # "LONG" or "SHORT"
        qty: int,           # 1 or 2 for ratio spreads
        strike_rank: int,   # 0=A (lowest) … 3=D (highest)
        expiry_mode: str = "same",  # "same" | "near" | "far" | "diag_long"
    ):
        self.opt_type    = opt_type
        self.side        = side
        self.qty         = qty
        self.strike_rank = strike_rank
        self.expiry_mode = expiry_mode

    def qty_sign(self) -> int:
        return self.qty if self.side == "LONG" else -self.qty

    def __repr__(self):
        s = "+" if self.side == "LONG" else "-"
        return f"{s}{self.qty}{self.opt_type}({chr(65 + self.strike_rank)})"


# =============================================================================
# STRATEGY TEMPLATE
# =============================================================================

class StrategyTemplate:
    """
    Complete descriptor for one options strategy, including:
      - v4.3 class mapping (10-class universe)
      - Leg structure
      - Vectorized eligibility predicate
      - Economics multipliers (pop, ev, max_loss relative to IC baseline)
    """

    def __init__(
        self,
        template_id: str,
        v43_class: str,
        family: str,
        legs: Tuple[LegTemplate, ...],
        notes: str = "",
        pop_scale: float = 1.0,
        pop_offset: float = 0.0,
        ev_scale: float = 1.0,
        max_loss_scale: float = 1.0,
        predicate: Optional[Callable[[Dict[str, np.ndarray]], np.ndarray]] = None,
    ):
        self.template_id     = template_id
        self.v43_class       = v43_class
        self.family          = family
        self.legs            = legs
        self.notes           = notes
        self.pop_scale       = pop_scale
        self.pop_offset      = pop_offset
        self.ev_scale        = ev_scale
        self.max_loss_scale  = max_loss_scale
        self._predicate      = predicate

    def get_eligibility_mask(self, atoms: Dict[str, np.ndarray]) -> np.ndarray:
        """Return boolean array [N] — True where this strategy is eligible."""
        if self._predicate is None:
            N = next(iter(atoms.values())).shape[0]
            return np.zeros(N, dtype=bool)
        return self._predicate(atoms).astype(bool)

    def __repr__(self):
        return (f"StrategyTemplate({self.template_id!r}, "
                f"class={self.v43_class!r}, family={self.family!r})")


# =============================================================================
# PREDICATE ATOM COMPUTATION  (vectorized, O(N), called once per dataset)
# =============================================================================

def compute_predicate_atoms(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Precompute all predicate atom arrays from the M5 ETL-enriched DataFrame.
    All atoms are bool np.ndarrays of shape [N].
    Missing columns → safe defaults (conservative: atom = False).
    """
    N = len(df)

    def get(col: str, default: float = 0.0) -> np.ndarray:
        if col in df.columns:
            return df[col].fillna(default).values.astype(np.float64)
        return np.full(N, default, dtype=np.float64)

    atoms: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # IVR / Volatility regime
    # ivr_zone ∈ {-1, 0, +1}: computed by data_pipeline_v43
    # ------------------------------------------------------------------
    ivr_zone = get("ivr_zone", 0.0)
    atoms["ivr_high"]    = ivr_zone >= 1.0    # IVR > 70th pct
    atoms["ivr_low"]     = ivr_zone <= -1.0   # IVR < 30th pct
    atoms["ivr_neutral"] = ivr_zone == 0.0    # 30th–70th pct
    atoms["ivr_any"]     = ivr_zone != 0.0    # any extreme

    # ------------------------------------------------------------------
    # Trend signals (multi-source consensus)
    # ------------------------------------------------------------------
    trend_up   = get("TrendSeekerUp",   0.0)
    trend_dn   = get("TrendSeekerDown", 0.0)
    psar_trend = get("psar_trend",      0.0)
    breakout   = get("breakout_score",  0.0)
    slope      = get("Slope",           0.0)
    pressure_u = get("pressure_up",     0.0)
    pressure_d = get("pressure_down",   0.0)

    atoms["trend_bull"] = (trend_up > 0) | (psar_trend > 0) | (pressure_u > 0.6)
    atoms["trend_bear"] = (trend_dn > 0) | (psar_trend < 0) | (pressure_d > 0.6)
    atoms["trend_weak"] = ~atoms["trend_bull"] & ~atoms["trend_bear"]
    atoms["psar_bull"]  = psar_trend > 0
    atoms["psar_bear"]  = psar_trend < 0

    atoms["breakout_up"]   = breakout > 0
    atoms["breakout_down"] = breakout < 0
    atoms["breakout_any"]  = breakout != 0
    atoms["no_breakout"]   = breakout == 0

    atoms["slope_positive"] = slope > 0
    atoms["slope_negative"] = slope < 0

    # ------------------------------------------------------------------
    # Consolidation / range-bound
    # ------------------------------------------------------------------
    consol = get("consolidation_score", 0.0)
    atoms["consol_high"]  = consol >= 0.65
    atoms["consol_vhigh"] = consol >= 0.80

    # ------------------------------------------------------------------
    # Momentum / RSI
    # ------------------------------------------------------------------
    rsi = get("rsi_dyn", 50.0)
    atoms["rsi_neutral"]    = (rsi >= 35.0) & (rsi <= 65.0)
    atoms["rsi_overbought"] = rsi > 70.0
    atoms["rsi_oversold"]   = rsi < 30.0
    atoms["rsi_bull"]       = rsi > 55.0
    atoms["rsi_bear"]       = rsi < 45.0

    # ------------------------------------------------------------------
    # ADX (trend strength)
    # ------------------------------------------------------------------
    adx = get("adx_adaptive", 50.0)
    atoms["adx_weak"]    = adx < 25.0
    atoms["adx_strong"]  = adx >= 25.0
    atoms["adx_vstrong"] = adx >= 40.0

    # ------------------------------------------------------------------
    # Bollinger Bands
    # ------------------------------------------------------------------
    bb_pct  = get("bb_percentile", 50.0)
    bw_rate = get("bw_expansion_rate", 0.0)
    atoms["bb_neutral"]     = (bb_pct >= 20.0) & (bb_pct <= 80.0)
    atoms["bb_high"]        = bb_pct > 80.0
    atoms["bb_low"]         = bb_pct < 20.0
    atoms["bw_expanding"]   = bw_rate > 0.0
    atoms["bw_compressing"] = bw_rate <= 0.0

    # ------------------------------------------------------------------
    # Reversal signals (IVR Reversal Engine features)
    # ------------------------------------------------------------------
    stretch_zone = get("stretch_zone", 0.0)
    rev_score    = get("reversal_score", 0.0)
    # Bullish reversal: low IV (cheap premium) + oversold (price_stretch < -2 ATR)
    atoms["reversal_bull"]   = (ivr_zone <= -1.0) & (stretch_zone <= -1.0)
    # Bearish reversal: high IV (expensive) + overbought
    atoms["reversal_bear"]   = (ivr_zone >= 1.0)  & (stretch_zone >= 1.0)
    atoms["reversal_strong"] = np.abs(rev_score) >= 2.0

    # ------------------------------------------------------------------
    # Gap / chaos
    # ------------------------------------------------------------------
    gap   = get("gap_risk_score",   0.0)
    chaos = get("chaos_membership", 0.0)
    atoms["gap_low"]    = gap <= 0.40
    atoms["gap_high"]   = gap > 0.60
    atoms["chaos_low"]  = chaos < 0.30
    atoms["chaos_high"] = chaos > 0.70

    # ------------------------------------------------------------------
    # Regime persistence (bars in same vol×trend bucket)
    # ------------------------------------------------------------------
    regime_pers = get("regime_persistence", 0.0)
    atoms["regime_stable"] = regime_pers >= 5

    # ------------------------------------------------------------------
    # Friction / liquidity (any of the 5 friction windows passes)
    # ------------------------------------------------------------------
    friction_ok = np.zeros(N, dtype=bool)
    for n in [5, 10, 20, 40, 60]:
        col = f"friction_ok_{n}"
        if col in df.columns:
            friction_ok |= df[col].fillna(0).values > 0
    atoms["friction_ok"] = friction_ok

    return atoms


# =============================================================================
# STRATEGY CATALOG BUILDER
# =============================================================================

def _S(template_id, v43_class, family, legs, notes="",
        pop_scale=1.0, pop_offset=0.0, ev_scale=1.0, max_loss_scale=1.0,
        pred=None) -> StrategyTemplate:
    """Compact StrategyTemplate constructor for catalog definition."""
    return StrategyTemplate(
        template_id=template_id, v43_class=v43_class, family=family,
        legs=legs, notes=notes,
        pop_scale=pop_scale, pop_offset=pop_offset,
        ev_scale=ev_scale, max_loss_scale=max_loss_scale,
        predicate=pred,
    )


def _L(t, side, qty, rank, exp="same") -> LegTemplate:
    return LegTemplate(opt_type=t, side=side, qty=qty, strike_rank=rank, expiry_mode=exp)


def build_strategy_catalog() -> List[StrategyTemplate]:
    """
    Build and return the full strategy catalog.
    Call once at pipeline startup; cache the result.
    """
    SVI = FAMILY_SHORT_VOL_INCOME
    LVC = FAMILY_LONG_VOL_CONVEX
    DDR = FAMILY_DIRECTIONAL_DEFINED
    TS  = FAMILY_TERM_STRUCTURE

    catalog: List[StrategyTemplate] = []

    # ==========================================================================
    # A) NEUTRAL INCOME — Short Vol / Credit (ShortVolIncome)
    # ==========================================================================

    # ---- Iron Condor (native class) -------------------------------------------
    catalog.append(_S(
        "iron_condor", "iron_condor", SVI,
        (_L("P","LONG",1,0), _L("P","SHORT",1,1), _L("C","SHORT",1,2), _L("C","LONG",1,3)),
        notes="Neutral credit; wider profit zone.",
        pop_scale=1.00, ev_scale=1.00, max_loss_scale=1.00,
        pred=lambda a: (
            a["ivr_neutral"]                        # IC owns neutral-IV zone only
            & (a["consol_high"] | a["adx_weak"])    # range-bound by either measure
            & a["rsi_neutral"] & a["bb_neutral"]
            & a["no_breakout"]  & a["friction_ok"]
        ),
    ))

    # ---- Iron Butterfly -------------------------------------------------------
    catalog.append(_S(
        "iron_butterfly", "custom_multi_leg", SVI,
        (_L("P","LONG",1,0), _L("P","SHORT",1,1), _L("C","SHORT",1,1), _L("C","LONG",1,2)),
        notes="Tighter than IC; higher max profit but narrower range.",
        pop_scale=0.85, ev_scale=0.95, max_loss_scale=1.10,
        pred=lambda a: (
            a["ivr_high"] & a["consol_vhigh"]
            & a["rsi_neutral"] & a["regime_stable"] & a["friction_ok"]
        ),
    ))

    # ---- Short Straddle -------------------------------------------------------
    catalog.append(_S(
        "short_straddle", "straddle", SVI,
        (_L("P","SHORT",1,0), _L("C","SHORT",1,0)),
        notes="ATM short vol; max premium; uncapped tails.",
        pop_scale=0.78, ev_scale=1.12, max_loss_scale=1.20,
        pred=lambda a: (
            a["ivr_high"] & (a["consol_high"] | a["consol_vhigh"])
            & a["gap_low"] & a["friction_ok"]
        ),
    ))

    # ---- Short Strangle -------------------------------------------------------
    catalog.append(_S(
        "short_strangle", "strangle", SVI,
        (_L("P","SHORT",1,0), _L("C","SHORT",1,1)),
        notes="OTM short vol; wider profit zone than straddle.",
        pop_scale=0.85, ev_scale=1.05, max_loss_scale=1.15,
        pred=lambda a: (
            a["ivr_high"] & a["consol_high"]
            & a["no_breakout"] & a["friction_ok"]
        ),
    ))

    # ---- Bull Put Spread (credit) ---------------------------------------------
    catalog.append(_S(
        "bull_put_spread_credit", "custom_multi_leg", SVI,
        (_L("P","LONG",1,0), _L("P","SHORT",1,1)),
        notes="Bullish credit spread; defined risk.",
        pop_scale=0.90, ev_scale=0.90, max_loss_scale=0.70,
        pred=lambda a: (
            a["trend_bull"] & a["ivr_high"] & a["gap_low"]
            & a["friction_ok"] & (a["rsi_bull"] | a["reversal_bull"])
        ),
    ))

    # ---- Bear Call Spread (credit) -------------------------------------------
    catalog.append(_S(
        "bear_call_spread_credit", "custom_multi_leg", SVI,
        (_L("C","SHORT",1,0), _L("C","LONG",1,1)),
        notes="Bearish credit spread; defined risk.",
        pop_scale=0.90, ev_scale=0.90, max_loss_scale=0.70,
        pred=lambda a: (
            a["trend_bear"] & a["ivr_high"] & a["friction_ok"]
            & (a["rsi_bear"] | a["reversal_bear"])
        ),
    ))

    # ---- Short Put (naked) ---------------------------------------------------
    catalog.append(_S(
        "short_put", "single_put", SVI,
        (_L("P","SHORT",1,0),),
        notes="Bullish naked put; limited profit, large downside.",
        pop_scale=1.08, ev_scale=1.15, max_loss_scale=1.40,
        pred=lambda a: (
            a["trend_bull"] & a["ivr_high"]
            & a["rsi_oversold"] & a["gap_low"] & a["friction_ok"]
        ),
    ))

    # ---- Short Call (naked) --------------------------------------------------
    catalog.append(_S(
        "short_call", "single_call", SVI,
        (_L("C","SHORT",1,0),),
        notes="Bearish naked call; limited profit, unlimited upside risk.",
        pop_scale=1.05, ev_scale=1.10, max_loss_scale=1.60,
        pred=lambda a: (
            a["trend_bear"] & a["ivr_high"]
            & a["rsi_overbought"] & a["friction_ok"]
        ),
    ))

    # ---- Jade Lizard ---------------------------------------------------------
    catalog.append(_S(
        "jade_lizard", "custom_multi_leg", SVI,
        (_L("P","SHORT",1,0), _L("C","SHORT",1,1), _L("C","LONG",1,2)),
        notes="Slight bull; no upside risk when credit >= call spread width.",
        pop_scale=0.95, ev_scale=1.05, max_loss_scale=0.90,
        pred=lambda a: (
            a["reversal_bull"] & a["ivr_high"]
            & a["rsi_oversold"] & a["friction_ok"]
        ),
    ))

    # ---- Reverse Jade Lizard -------------------------------------------------
    catalog.append(_S(
        "reverse_jade_lizard", "custom_multi_leg", SVI,
        (_L("C","SHORT",1,2), _L("P","LONG",1,1), _L("P","SHORT",1,0)),
        notes="Slight bear; mirror Jade Lizard.",
        pop_scale=0.95, ev_scale=1.05, max_loss_scale=0.90,
        pred=lambda a: (
            a["reversal_bear"] & a["ivr_high"]
            & a["rsi_overbought"] & a["friction_ok"]
        ),
    ))

    # ---- Short Call Condor ---------------------------------------------------
    catalog.append(_S(
        "short_call_condor", "custom_multi_leg", SVI,
        (_L("C","SHORT",1,0), _L("C","LONG",1,1), _L("C","LONG",1,2), _L("C","SHORT",1,3)),
        notes="Directional short vol; sell wings, buy body calls.",
        pop_scale=0.80, ev_scale=0.85, max_loss_scale=0.80,
        pred=lambda a: (
            a["trend_bear"] & a["ivr_high"] & a["adx_strong"] & a["friction_ok"]
        ),
    ))

    # ---- Short Put Condor ----------------------------------------------------
    catalog.append(_S(
        "short_put_condor", "custom_multi_leg", SVI,
        (_L("P","SHORT",1,3), _L("P","LONG",1,2), _L("P","LONG",1,1), _L("P","SHORT",1,0)),
        notes="Directional short vol; put-side analog of short call condor.",
        pop_scale=0.80, ev_scale=0.85, max_loss_scale=0.80,
        pred=lambda a: (
            a["trend_bull"] & a["ivr_high"] & a["adx_strong"] & a["friction_ok"]
        ),
    ))

    # ---- Short Guts ----------------------------------------------------------
    catalog.append(_S(
        "short_guts", "custom_multi_leg", SVI,
        (_L("C","SHORT",1,0), _L("P","SHORT",1,1)),
        notes="ITM straddle-like; high premium, uncapped risk.",
        pop_scale=0.72, ev_scale=1.25, max_loss_scale=1.30,
        pred=lambda a: (
            a["consol_high"] & a["ivr_high"] & a["friction_ok"]
        ),
    ))

    # ---- Call Ratio Spread (slight bull, falling vol) ------------------------
    catalog.append(_S(
        "call_ratio_spread", "custom_multi_leg", SVI,
        (_L("C","LONG",1,0), _L("C","SHORT",2,1)),
        notes="Slight bull/neutral; net credit; uncapped upside risk.",
        pop_scale=0.88, ev_scale=0.80, max_loss_scale=0.85,
        pred=lambda a: (
            a["trend_weak"] & a["bw_compressing"]
            & ~a["ivr_high"] & a["friction_ok"]
        ),
    ))

    # ---- Put Ratio Spread (slight bear, falling vol) -------------------------
    catalog.append(_S(
        "put_ratio_spread", "custom_multi_leg", SVI,
        (_L("P","SHORT",2,0), _L("P","LONG",1,1)),
        notes="Slight bear/neutral; net credit; uncapped downside risk.",
        pop_scale=0.88, ev_scale=0.80, max_loss_scale=0.85,
        pred=lambda a: (
            a["trend_weak"] & a["bw_compressing"]
            & ~a["ivr_high"] & a["friction_ok"]
        ),
    ))

    # ---- Covered Call --------------------------------------------------------
    catalog.append(_S(
        "covered_call", "single_call", SVI,
        (_L("C","SHORT",1,1),),
        notes="Stock + short call; income; capped upside.",
        pop_scale=0.85, ev_scale=0.88, max_loss_scale=0.60,
        pred=lambda a: (
            a["trend_bull"] & a["ivr_high"]
            & a["rsi_bull"] & a["friction_ok"]
        ),
    ))

    # ---- Cash-Secured Put ----------------------------------------------------
    catalog.append(_S(
        "cash_secured_put", "single_put", SVI,
        (_L("P","SHORT",1,0),),
        notes="Put selling with cash reserve; bullish income entry.",
        pop_scale=1.05, ev_scale=1.00, max_loss_scale=1.00,
        pred=lambda a: (
            a["reversal_bull"] & a["ivr_high"]
            & a["gap_low"] & a["friction_ok"]
        ),
    ))

    # ---- Covered Short Straddle ----------------------------------------------
    catalog.append(_S(
        "covered_short_straddle", "custom_multi_leg", SVI,
        (_L("C","SHORT",1,0), _L("P","SHORT",1,0)),
        notes="Stock + short ATM straddle; income with underlying protection.",
        pop_scale=0.80, ev_scale=1.10, max_loss_scale=1.00,
        pred=lambda a: (
            a["ivr_high"] & a["consol_vhigh"] & a["trend_bull"]
            & a["regime_stable"] & a["gap_low"] & a["friction_ok"]
        ),
    ))

    # ---- Covered Short Strangle ----------------------------------------------
    catalog.append(_S(
        "covered_short_strangle", "custom_multi_leg", SVI,
        (_L("P","SHORT",1,0), _L("C","SHORT",1,1)),
        notes="Stock + short OTM strangle; less risk than straddle.",
        pop_scale=0.85, ev_scale=1.05, max_loss_scale=1.00,
        pred=lambda a: (
            a["ivr_high"] & a["consol_high"]
            & (a["trend_bull"] | a["trend_weak"]) & a["friction_ok"]
        ),
    ))

    # ==========================================================================
    # B) LONG VOL / CONVEX  (LongVolConvex)
    # ==========================================================================

    # ---- Long Straddle -------------------------------------------------------
    catalog.append(_S(
        "straddle_long", "straddle", LVC,
        (_L("P","LONG",1,0), _L("C","LONG",1,0)),
        notes="Long ATM vol; needs large move; pays theta.",
        pop_scale=-1.0, pop_offset=1.0,
        ev_scale=-0.60, max_loss_scale=0.40,
        pred=lambda a: (
            # Long straddle: vol expanding from cheap IV levels
            a["bw_expanding"] & a["ivr_low"]
            & a["friction_ok"]
        ),
    ))

    # ---- Long Strangle -------------------------------------------------------
    catalog.append(_S(
        "strangle_long", "strangle", LVC,
        (_L("P","LONG",1,0), _L("C","LONG",1,1)),
        notes="Long OTM vol; cheaper than straddle; needs larger move.",
        pop_scale=-1.0, pop_offset=1.0,
        ev_scale=-0.55, max_loss_scale=0.30,
        pred=lambda a: (
            # Long strangle: vol expanding + breakout, IV not already expensive
            a["bw_expanding"] & ~a["ivr_high"]
            & a["breakout_any"] & a["friction_ok"]
        ),
    ))

    # ---- Call Ratio Backspread -----------------------------------------------
    catalog.append(_S(
        "call_ratio_backspread", "custom_multi_leg", LVC,
        (_L("C","SHORT",1,0), _L("C","LONG",2,1)),
        notes="Convex upside; sell 1 lower call, buy 2 higher; needs bullish breakout.",
        pop_scale=-0.50, pop_offset=0.80,
        ev_scale=0.60, max_loss_scale=0.30,
        pred=lambda a: (
            a["breakout_up"] & a["adx_strong"]
            & a["bw_expanding"] & a["friction_ok"]
        ),
    ))

    # ---- Put Ratio Backspread ------------------------------------------------
    catalog.append(_S(
        "put_ratio_backspread", "custom_multi_leg", LVC,
        (_L("P","LONG",2,0), _L("P","SHORT",1,1)),
        notes="Convex downside; buy 2 lower puts, sell 1 higher; needs bearish breakout.",
        pop_scale=-0.50, pop_offset=0.80,
        ev_scale=0.60, max_loss_scale=0.30,
        pred=lambda a: (
            a["breakout_down"] & a["adx_strong"]
            & a["bw_expanding"] & a["friction_ok"]
        ),
    ))

    # ---- Inverse Iron Butterfly ----------------------------------------------
    catalog.append(_S(
        "inverse_iron_butterfly", "custom_multi_leg", LVC,
        (_L("P","SHORT",1,0), _L("P","LONG",1,1), _L("C","LONG",1,1), _L("C","SHORT",1,2)),
        notes="Long vol; small debit; profits from breakout either side.",
        pop_scale=-0.80, pop_offset=1.0,
        ev_scale=-0.50, max_loss_scale=0.50,
        pred=lambda a: (
            a["bw_expanding"] & a["adx_strong"]
            & ~a["ivr_high"] & a["friction_ok"]
        ),
    ))

    # ---- Inverse Iron Condor -------------------------------------------------
    catalog.append(_S(
        "inverse_iron_condor", "custom_multi_leg", LVC,
        (_L("P","SHORT",1,0), _L("P","LONG",1,1), _L("C","LONG",1,2), _L("C","SHORT",1,3)),
        notes="Long vol; wider than inv. butterfly; benefits from large move.",
        pop_scale=-0.70, pop_offset=1.0,
        ev_scale=-0.50, max_loss_scale=0.50,
        pred=lambda a: (
            a["bw_expanding"] & a["adx_strong"]
            & ~a["ivr_high"] & a["friction_ok"]
        ),
    ))

    # ---- Short Call Butterfly ------------------------------------------------
    catalog.append(_S(
        "short_call_butterfly", "butterfly_call", LVC,
        (_L("C","SHORT",1,0), _L("C","LONG",2,1), _L("C","SHORT",1,2)),
        notes="Long vol around ATM; benefits from expanding bandwidth.",
        pop_scale=-0.60, pop_offset=0.90,
        ev_scale=-0.40, max_loss_scale=0.40,
        pred=lambda a: (
            a["bw_expanding"] & a["ivr_high"]
            & a["no_breakout"] & a["friction_ok"]
        ),
    ))

    # ---- Short Put Butterfly -------------------------------------------------
    catalog.append(_S(
        "short_put_butterfly", "butterfly_call", LVC,  # reuse butterfly_call class
        (_L("P","SHORT",1,0), _L("P","LONG",2,1), _L("P","SHORT",1,2)),
        notes="Put-side long vol butterfly.",
        pop_scale=-0.60, pop_offset=0.90,
        ev_scale=-0.40, max_loss_scale=0.40,
        pred=lambda a: (
            a["bw_expanding"] & a["ivr_high"]
            & a["no_breakout"] & a["friction_ok"]
        ),
    ))

    # ---- Long Call Condor ----------------------------------------------------
    catalog.append(_S(
        "long_call_condor", "custom_multi_leg", LVC,
        (_L("C","LONG",1,0), _L("C","SHORT",1,1), _L("C","SHORT",1,2), _L("C","LONG",1,3)),
        notes="Rangebound; low IV; expects mild move into body of condor.",
        pop_scale=0.60, pop_offset=0.0,
        ev_scale=-0.30, max_loss_scale=0.30,
        pred=lambda a: (
            a["ivr_low"] & a["consol_high"] & a["adx_weak"] & a["friction_ok"]
        ),
    ))

    # ---- Long Put Condor -----------------------------------------------------
    catalog.append(_S(
        "long_put_condor", "custom_multi_leg", LVC,
        (_L("P","LONG",1,3), _L("P","SHORT",1,2), _L("P","SHORT",1,1), _L("P","LONG",1,0)),
        notes="Put-side rangebound; low IV.",
        pop_scale=0.60, pop_offset=0.0,
        ev_scale=-0.30, max_loss_scale=0.30,
        pred=lambda a: (
            a["ivr_low"] & a["consol_high"] & a["adx_weak"] & a["friction_ok"]
        ),
    ))

    # ---- Strip (bearish directional, large move expected) --------------------
    catalog.append(_S(
        "strip", "custom_multi_leg", LVC,
        (_L("C","LONG",1,0), _L("P","LONG",2,0)),
        notes="Long vol, bearish bias (2×puts + 1×call at same strike).",
        pop_scale=-0.50, pop_offset=0.90,
        ev_scale=0.50, max_loss_scale=0.40,
        pred=lambda a: (
            a["trend_bear"] & a["bw_expanding"]
            & ~a["ivr_high"] & a["friction_ok"]
        ),
    ))

    # ---- Strap (bullish directional, large move expected) --------------------
    catalog.append(_S(
        "strap", "custom_multi_leg", LVC,
        (_L("C","LONG",2,0), _L("P","LONG",1,0)),
        notes="Long vol, bullish bias (2×calls + 1×put at same strike).",
        pop_scale=-0.50, pop_offset=0.90,
        ev_scale=0.50, max_loss_scale=0.40,
        pred=lambda a: (
            a["trend_bull"] & a["bw_expanding"]
            & ~a["ivr_high"] & a["friction_ok"]
        ),
    ))

    # ---- Guts (long; ITM, strike-separated) ----------------------------------
    catalog.append(_S(
        "guts", "custom_multi_leg", LVC,
        (_L("C","LONG",1,0), _L("P","LONG",1,1)),
        notes="Long ITM call + ITM put; needs very large move.",
        pop_scale=-0.60, pop_offset=0.90,
        ev_scale=0.40, max_loss_scale=0.40,
        pred=lambda a: (
            a["bw_expanding"] & a["adx_strong"]
            & ~a["ivr_high"] & a["friction_ok"]
        ),
    ))

    # ==========================================================================
    # C) DIRECTIONAL DEFINED RISK  (DirectionalDefinedRisk)
    # ==========================================================================

    # ---- Long Call -----------------------------------------------------------
    catalog.append(_S(
        "long_call", "single_call", DDR,
        (_L("C","LONG",1,0),),
        notes="Bullish convex; limited loss; needs breakout up.",
        pop_scale=0.60, ev_scale=0.70, max_loss_scale=0.40,
        pred=lambda a: (
            a["trend_bull"] & a["breakout_up"]
            & ~a["ivr_high"] & a["adx_strong"] & a["friction_ok"]
        ),
    ))

    # ---- Long Put ------------------------------------------------------------
    catalog.append(_S(
        "long_put", "single_put", DDR,
        (_L("P","LONG",1,0),),
        notes="Bearish convex; limited loss; needs breakdown.",
        pop_scale=0.60, ev_scale=0.70, max_loss_scale=0.40,
        pred=lambda a: (
            a["trend_bear"] & a["breakout_down"]
            & ~a["ivr_high"] & a["adx_strong"] & a["friction_ok"]
        ),
    ))

    # ---- Bull Call Spread (debit; native class) ------------------------------
    catalog.append(_S(
        "bull_call_spread", "bull_call_spread", DDR,
        (_L("C","LONG",1,0), _L("C","SHORT",1,1)),
        notes="Bullish debit spread; limited P/L.",
        pop_scale=0.75, ev_scale=0.80, max_loss_scale=0.65,
        pred=lambda a: (
            a["trend_bull"] & a["adx_strong"]
            & a["gap_low"] & a["friction_ok"]
        ),
    ))

    # ---- Bear Put Spread (debit; native class) --------------------------------
    catalog.append(_S(
        "bear_put_spread", "bear_put_spread", DDR,
        (_L("P","LONG",1,1), _L("P","SHORT",1,0)),
        notes="Bearish debit spread; limited P/L.",
        pop_scale=0.75, ev_scale=0.80, max_loss_scale=0.65,
        pred=lambda a: (
            a["trend_bear"] & a["adx_strong"] & a["friction_ok"]
        ),
    ))

    # ---- Long Call Butterfly (cheap; pinning expected; native class) ----------
    catalog.append(_S(
        "long_call_butterfly", "butterfly_call", DDR,
        (_L("C","LONG",1,0), _L("C","SHORT",2,1), _L("C","LONG",1,2)),
        notes="Neutral cheap debit; profits from pinning at body strike.",
        pop_scale=0.72,  # > 0.60 of long_call_condor → wins on consol_vhigh bars
        ev_scale=-0.20, max_loss_scale=0.20,
        pred=lambda a: (
            ~a["ivr_high"] & a["adx_weak"] & a["consol_vhigh"]
            & a["rsi_neutral"] & a["friction_ok"]
        ),
    ))

    # ---- Long Put Butterfly --------------------------------------------------
    catalog.append(_S(
        "long_put_butterfly", "custom_multi_leg", DDR,
        (_L("P","LONG",1,0), _L("P","SHORT",2,1), _L("P","LONG",1,2)),
        notes="Put-side long butterfly; cheap pinning play.",
        pop_scale=0.60, ev_scale=-0.20, max_loss_scale=0.20,
        pred=lambda a: (
            ~a["ivr_high"] & a["adx_weak"] & a["consol_vhigh"]
            & a["rsi_neutral"] & a["friction_ok"]
        ),
    ))

    # ---- Protective Put (hedge) ----------------------------------------------
    catalog.append(_S(
        "protective_put", "single_put", DDR,
        (_L("P","LONG",1,0),),
        notes="Portfolio hedge; pays premium; protects downside.",
        pop_scale=0.40, ev_scale=-0.50, max_loss_scale=0.30,
        pred=lambda a: (
            a["gap_high"] | (a["chaos_high"] & a["trend_bull"])
        ),
    ))

    # ---- Long Synthetic Future -----------------------------------------------
    catalog.append(_S(
        "long_synthetic_future", "custom_multi_leg", DDR,
        (_L("C","LONG",1,0), _L("P","SHORT",1,0)),
        notes="Buy call + sell put at same strike; replicates long stock.",
        pop_scale=0.65, ev_scale=0.75, max_loss_scale=0.80,
        pred=lambda a: (
            a["trend_bull"] & a["adx_vstrong"]
            & a["gap_low"] & a["friction_ok"]
        ),
    ))

    # ---- Short Synthetic Future ----------------------------------------------
    catalog.append(_S(
        "short_synthetic_future", "custom_multi_leg", DDR,
        (_L("P","LONG",1,0), _L("C","SHORT",1,0)),
        notes="Buy put + sell call at same strike; replicates short stock.",
        pop_scale=0.65, ev_scale=0.75, max_loss_scale=0.80,
        pred=lambda a: (
            a["trend_bear"] & a["adx_vstrong"] & a["friction_ok"]
        ),
    ))

    # ---- Long Combo (bullish, two-strike deadzone) ---------------------------
    catalog.append(_S(
        "long_combo", "custom_multi_leg", DDR,
        (_L("P","SHORT",1,0), _L("C","LONG",1,1)),
        notes="Sell lower put + buy higher call; bullish with deadzone.",
        pop_scale=0.70, ev_scale=0.70, max_loss_scale=0.70,
        pred=lambda a: (
            a["trend_bull"] & a["breakout_up"] & a["adx_strong"] & a["friction_ok"]
        ),
    ))

    # ---- Short Combo (bearish) -----------------------------------------------
    catalog.append(_S(
        "short_combo", "custom_multi_leg", DDR,
        (_L("P","LONG",1,0), _L("C","SHORT",1,1)),
        notes="Buy lower put + sell higher call; bearish with deadzone.",
        pop_scale=0.70, ev_scale=0.70, max_loss_scale=0.70,
        pred=lambda a: (
            a["trend_bear"] & a["breakout_down"] & a["adx_strong"] & a["friction_ok"]
        ),
    ))

    # ---- Collar (protective overlay; bullish with hedge) ---------------------
    catalog.append(_S(
        "collar", "custom_multi_leg", DDR,
        (_L("P","LONG",1,0), _L("C","SHORT",1,1)),
        notes="Stock + long put + short call; bullish with cap protection.",
        pop_scale=0.70, ev_scale=0.50, max_loss_scale=0.30,
        pred=lambda a: (
            a["trend_bull"] & (a["gap_high"] | a["chaos_high"]) & a["friction_ok"]
        ),
    ))

    # ---- Diagonal Call (PMCC — Poor Man's Covered Call) ---------------------
    catalog.append(_S(
        "diagonal_call", "bull_call_spread", DDR,
        (_L("C","LONG",1,0,"diag_long"), _L("C","SHORT",1,1,"near")),
        notes="Buy far-dated ITM call + sell near OTM call; bullish theta harvest.",
        pop_scale=0.80, ev_scale=0.85, max_loss_scale=0.50,
        pred=lambda a: (
            a["trend_bull"] & a["ivr_high"] & a["regime_stable"] & a["friction_ok"]
        ),
    ))

    # ---- Diagonal Put --------------------------------------------------------
    catalog.append(_S(
        "diagonal_put", "bear_put_spread", DDR,
        (_L("P","LONG",1,1,"diag_long"), _L("P","SHORT",1,0,"near")),
        notes="Buy far-dated ITM put + sell near OTM put; bearish theta harvest.",
        pop_scale=0.80, ev_scale=0.85, max_loss_scale=0.50,
        pred=lambda a: (
            a["trend_bear"] & a["ivr_high"] & a["regime_stable"] & a["friction_ok"]
        ),
    ))

    # ---- Put Broken Wing (asymmetric; slight bullish bias) -------------------
    catalog.append(_S(
        "put_broken_wing", "custom_multi_leg", DDR,
        (_L("P","LONG",1,0), _L("P","SHORT",2,1), _L("P","LONG",1,2)),
        notes="Skip-strike butterfly; skewed to reduce downside risk.",
        pop_scale=0.85, ev_scale=0.85, max_loss_scale=0.50,
        pred=lambda a: (
            (a["trend_bull"] | a["reversal_bull"])
            & ~a["ivr_high"] & a["consol_high"] & a["friction_ok"]
        ),
    ))

    # ---- Call Broken Wing (asymmetric; slight bearish bias) ------------------
    catalog.append(_S(
        "call_broken_wing", "butterfly_call", DDR,
        (_L("C","LONG",1,0), _L("C","SHORT",2,1), _L("C","LONG",1,2)),
        notes="Skip-strike call butterfly; skewed to reduce upside risk.",
        pop_scale=0.85, ev_scale=0.85, max_loss_scale=0.50,
        pred=lambda a: (
            (a["trend_bear"] | a["reversal_bear"])
            & ~a["ivr_high"] & a["consol_high"] & a["friction_ok"]
        ),
    ))

    # ---- Inverse Call Broken Wing (bullish; higher max profit) ---------------
    catalog.append(_S(
        "inverse_call_broken_wing", "custom_multi_leg", DDR,
        (_L("C","SHORT",1,0), _L("C","LONG",2,1), _L("C","SHORT",1,2)),
        notes="Inverted broken wing; bullish impulse + vol expansion.",
        pop_scale=0.70, ev_scale=0.75, max_loss_scale=0.60,
        pred=lambda a: (
            a["breakout_up"] & a["ivr_high"] & a["friction_ok"]
        ),
    ))

    # ---- Inverse Put Broken Wing (bearish) -----------------------------------
    catalog.append(_S(
        "inverse_put_broken_wing", "custom_multi_leg", DDR,
        (_L("P","SHORT",1,2), _L("P","LONG",2,1), _L("P","SHORT",1,0)),
        notes="Inverted broken wing; bearish impulse + vol expansion.",
        pop_scale=0.70, ev_scale=0.75, max_loss_scale=0.60,
        pred=lambda a: (
            a["breakout_down"] & a["ivr_high"] & a["friction_ok"]
        ),
    ))

    # ---- Bull Call Ladder (Long Call Ladder) ---------------------------------
    catalog.append(_S(
        "bull_call_ladder", "custom_multi_leg", DDR,
        (_L("C","LONG",1,0), _L("C","SHORT",1,1), _L("C","SHORT",1,2)),
        notes="Slight bullish; expects capped upside; avoids extreme tail.",
        pop_scale=0.75, ev_scale=0.65, max_loss_scale=0.70,
        pred=lambda a: (
            a["trend_bull"] & a["adx_weak"] & a["bw_compressing"] & a["friction_ok"]
        ),
    ))

    # ---- Bear Call Ladder (Short Call Ladder) --------------------------------
    catalog.append(_S(
        "bear_call_ladder", "custom_multi_leg", DDR,
        (_L("C","SHORT",1,0), _L("C","LONG",1,1), _L("C","LONG",1,2)),
        notes="Bearish or very bullish; unusual; high vol expected.",
        pop_scale=0.50, ev_scale=0.60, max_loss_scale=0.80,
        pred=lambda a: (
            a["adx_vstrong"] & a["bw_expanding"] & a["friction_ok"]
        ),
    ))

    # ---- Bull Put Ladder (Short Put Ladder) ----------------------------------
    catalog.append(_S(
        "bull_put_ladder", "custom_multi_leg", DDR,
        (_L("P","LONG",1,0), _L("P","LONG",1,1), _L("P","SHORT",1,2)),
        notes="Very bearish or very bullish; high vol; two-tailed.",
        pop_scale=0.50, ev_scale=0.60, max_loss_scale=0.80,
        pred=lambda a: (
            a["adx_vstrong"] & a["bw_expanding"] & a["friction_ok"]
        ),
    ))

    # ---- Bear Put Ladder (Long Put Ladder) -----------------------------------
    catalog.append(_S(
        "bear_put_ladder", "custom_multi_leg", DDR,
        (_L("P","SHORT",1,0), _L("P","SHORT",1,1), _L("P","LONG",1,2)),
        notes="Neutral to slightly bearish; falling vol; careful downside.",
        pop_scale=0.75, ev_scale=0.65, max_loss_scale=0.70,
        pred=lambda a: (
            a["trend_bear"] & a["adx_weak"] & a["bw_compressing"] & a["friction_ok"]
        ),
    ))

    # ---- Synthetic Put (Protective Call + short stock) ----------------------
    catalog.append(_S(
        "synthetic_put", "custom_multi_leg", DDR,
        (_L("C","LONG",1,0),),
        notes="Replicates put protection on short stock position.",
        pop_scale=0.60, ev_scale=0.50, max_loss_scale=0.40,
        pred=lambda a: (
            a["trend_bear"] & a["adx_vstrong"]
            & a["gap_high"] & a["friction_ok"]
        ),
    ))

    # ==========================================================================
    # D) TERM STRUCTURE  (TermStructure)
    # ==========================================================================

    # ---- Calendar Call -------------------------------------------------------
    catalog.append(_S(
        "calendar_call", "custom_multi_leg", TS,
        (_L("C","SHORT",1,0,"near"), _L("C","LONG",1,0,"far")),
        notes="Sell near-term call + buy same-strike far call; theta capture.",
        pop_scale=0.75, ev_scale=0.70, max_loss_scale=0.30,
        pred=lambda a: (
            a["adx_weak"] & a["consol_high"] & a["rsi_neutral"]
            & a["regime_stable"] & a["friction_ok"]
        ),
    ))

    # ---- Calendar Put --------------------------------------------------------
    catalog.append(_S(
        "calendar_put", "custom_multi_leg", TS,
        (_L("P","SHORT",1,0,"near"), _L("P","LONG",1,0,"far")),
        notes="Put-side calendar; theta capture with neutral directional bias.",
        pop_scale=0.75, ev_scale=0.70, max_loss_scale=0.30,
        pred=lambda a: (
            a["adx_weak"] & a["consol_high"] & a["rsi_neutral"]
            & a["regime_stable"] & a["friction_ok"]
        ),
    ))

    # ---- Double Diagonal -----------------------------------------------------
    catalog.append(_S(
        "double_diagonal", "custom_multi_leg", TS,
        (
            _L("P","LONG",1,0,"far"), _L("P","SHORT",1,1,"near"),
            _L("C","SHORT",1,2,"near"), _L("C","LONG",1,3,"far"),
        ),
        notes="Calendar + strangle; neutral; exploits term structure IV edge.",
        pop_scale=0.80, ev_scale=0.75, max_loss_scale=0.40,
        pred=lambda a: (
            a["adx_weak"] & a["consol_high"] & a["ivr_high"]
            & a["regime_stable"] & a["friction_ok"]
        ),
    ))

    return catalog


# =============================================================================
# MODULE-LEVEL SINGLETON CATALOG
# =============================================================================

STRATEGY_CATALOG: List[StrategyTemplate] = build_strategy_catalog()
"""Full catalog — built once at import time. ~50 strategies."""


def get_catalog() -> List[StrategyTemplate]:
    """Return the singleton catalog (re-instantiate if needed)."""
    return STRATEGY_CATALOG


# =============================================================================
# VECTORIZED ELIGIBILITY MATRIX
# =============================================================================

def build_eligibility_matrix(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, List[StrategyTemplate]]:
    """
    Compute the [N, K] eligibility matrix for all K catalog templates.

    Args:
        df: M5 ETL-enriched DataFrame (with IVR reversal features present).

    Returns:
        eligible: np.ndarray[bool, (N, K)] — True where bar i is eligible
                  for template k.
        catalog:  The ordered list of StrategyTemplate objects (K entries).
    """
    atoms   = compute_predicate_atoms(df)
    catalog = STRATEGY_CATALOG
    N       = len(df)
    K       = len(catalog)

    eligible = np.zeros((N, K), dtype=bool)
    for k, template in enumerate(catalog):
        eligible[:, k] = template.get_eligibility_mask(atoms)

    return eligible, catalog


# =============================================================================
# QUICK SELF-CHECK
# =============================================================================

if __name__ == "__main__":
    catalog = get_catalog()
    print(f"Strategy catalog: {len(catalog)} templates")

    from collections import Counter
    by_class  = Counter(t.v43_class  for t in catalog)
    by_family = Counter(t.family     for t in catalog)

    print("\nBy v4.3 class:")
    for cls, cnt in sorted(by_class.items()):
        print(f"  {cls:25s}: {cnt}")

    print("\nBy family:")
    for fam, cnt in sorted(by_family.items()):
        print(f"  {fam:30s}: {cnt}")

    print("\nAll templates:")
    for t in catalog:
        print(f"  {t.template_id:35s} -> {t.v43_class:20s} [{t.family[:3]}]"
              f"  pop×{t.pop_scale:+.2f}+{t.pop_offset:.2f}"
              f"  ev×{t.ev_scale:+.2f}"
              f"  ml×{t.max_loss_scale:.2f}")
