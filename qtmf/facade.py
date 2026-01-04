from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .models import TradeIntent, SizingPlan

# Reuse QTMF's existing fuzzy sizing helpers
from intelligence.fuzzy_engine import (
    compute_base_quantity,
    compute_fuzzy_confidence,
    normalize_volatility,
    compute_scaling_factor,
    # Core
    calculate_mtf_membership,
    calculate_iv_membership,
    calculate_regime_membership,
    # Advanced
    calculate_rsi_membership,
    calculate_adx_membership,
    calculate_bbands_membership,
    calculate_stoch_membership,
    calculate_volume_membership,
    calculate_sma_distance_membership,
)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _derive_put_call_weights(
    *,
    direction_probs: Optional[Any],
    gaussian_confidence: float,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Derive put/call wing weights from directional probabilities.

    For Iron Condors, default is 50/50 balanced wings.
    Can be adjusted based on neural direction_probs if available.

    Returns:
        (put_weight, call_weight, diagnostics)
    """
    # Default: Balanced 50/50 for Iron Condor
    put_weight = 0.5
    call_weight = 0.5

    # If neural forecast provides directional bias, adjust weights
    if direction_probs is not None and len(direction_probs) >= 3:
        # direction_probs format: [down, neutral, up]
        prob_down = direction_probs[0]
        prob_up = direction_probs[2]

        # Slight bias: if market leans bullish, reduce call weight slightly
        # (less risk on call side if expecting upward move)
        if prob_up > 0.5:
            call_weight = 0.45
            put_weight = 0.55
        elif prob_down > 0.5:
            put_weight = 0.45
            call_weight = 0.55

    diagnostics = {
        "put_weight": put_weight,
        "call_weight": call_weight,
        "direction_probs": direction_probs,
        "confidence": gaussian_confidence
    }

    return put_weight, call_weight, diagnostics

# ... (Let's target benchmark_and_size logic specifically)

def benchmark_and_size(
    trade_intent: TradeIntent | Dict[str, Any],
) -> SizingPlan:
    """Primary QTMF callable for Gaussian.
    
    This function is *pure*: no broker calls, no state mutations.
    
    Inputs:
      - A TradeIntent dataclass (preferred) or a compatible dict.
    
    Outputs:
      - SizingPlan with wing weights and quantities.
    """

    if isinstance(trade_intent, dict):
        ti = TradeIntent(
            symbol=str(trade_intent.get("symbol", "SPY")),
            action=str(trade_intent.get("action", "HOLD")),
            gaussian_confidence=float(trade_intent.get("gaussian_confidence", 0.0)),
            current_price=float(trade_intent.get("current_price", 0.0)),
            direction_probs=trade_intent.get("direction_probs"),
            vix=trade_intent.get("vix"),
            ivr=trade_intent.get("ivr"),
            realized_vol=trade_intent.get("realized_vol"),
            mtf_snapshot=trade_intent.get("mtf_snapshot"),
            # New Fields
            rsi=trade_intent.get("rsi"),
            adx=trade_intent.get("adx"),
            bb_position=trade_intent.get("bb_position"),
            bb_width=trade_intent.get("bb_width"),
            stoch_k=trade_intent.get("stoch_k"),
            volume_ratio=trade_intent.get("volume_ratio"),
            sma_distance=trade_intent.get("sma_distance"),
            neural_forecast=trade_intent.get("neural_forecast"),
            suggested_total_qty=trade_intent.get("suggested_total_qty"),
            extras=trade_intent.get("extras"),
        )
    else:
        ti = trade_intent

    extras = ti.extras or {}
    diagnostics: Dict[str, Any] = {
        "symbol": ti.symbol,
        "action": ti.action,
        "gaussian_confidence": ti.gaussian_confidence,
    }

    # Hard approval gates (caller can tune via extras)
    min_gaussian_conf = float(extras.get("min_gaussian_confidence", 0.55))
    if ti.gaussian_confidence < min_gaussian_conf:
        return SizingPlan(
            approved=False,
            reason=f"gaussian_confidence_below_{min_gaussian_conf:.2f}",
            total_qty=0,
            put_qty=0,
            call_qty=0,
            diagnostics=diagnostics,
        )

    # --- 1) Determine total quantity ---
    # Prefer a risk-based calculation if equity/max_loss_per_contract are provided.
    equity = extras.get("equity")
    max_loss_per_contract = extras.get("max_loss_per_contract")
    risk_fraction = float(extras.get("risk_fraction", 0.02))

    total_qty: int = 0
    total_qty_source = "none"

    if equity is not None and max_loss_per_contract is not None:
        try:
            total_qty = compute_base_quantity(
                equity=float(equity),
                max_loss_per_contract=float(max_loss_per_contract),
                risk_fraction=risk_fraction,
            )
            total_qty_source = "risk_ceiling"
        except Exception:
            total_qty = 0
            total_qty_source = "risk_ceiling_error"

    # If caller provided a suggested qty (e.g., RL), clamp by risk ceiling if available.
    if ti.suggested_total_qty is not None:
        try:
            suggested = int(ti.suggested_total_qty)
            if total_qty > 0:
                total_qty = min(total_qty, suggested)
                total_qty_source = f"min({total_qty_source},suggested)"
            else:
                total_qty = suggested
                total_qty_source = "suggested"
        except Exception:
            pass

    # If still unset, fall back to 1.
    if total_qty <= 0:
        total_qty = int(extras.get("fallback_total_qty", 1))
        total_qty_source = "fallback"

    # --- 2) Apply 9-Factor Fuzzy Scaling ---
    
    # A. Core Indicators
    mtf_mu = calculate_mtf_membership(ti.mtf_snapshot)
    iv_mu = calculate_iv_membership(float(ti.ivr)) if ti.ivr is not None else 0.5
    regime_mu = calculate_regime_membership(float(ti.vix)) if ti.vix is not None else 0.5
    
    # B. Advanced Indicators (with defaults)
    rsi_mu = calculate_rsi_membership(ti.rsi, 
        float(extras.get('rsi_neutral_min', 40.0)), 
        float(extras.get('rsi_neutral_max', 60.0))) if ti.rsi is not None else 0.5
        
    adx_mu = calculate_adx_membership(ti.adx,
        float(extras.get('adx_threshold_low', 25.0)),
        float(extras.get('adx_threshold_high', 40.0))) if ti.adx is not None else 0.5
        
    bb_mu = calculate_bbands_membership(ti.bb_position, ti.bb_width,
        float(extras.get('bb_squeeze_threshold', 0.02))) if ti.bb_position is not None else 0.5
        
    stoch_mu = calculate_stoch_membership(ti.stoch_k,
        float(extras.get('stoch_neutral_min', 30.0)),
        float(extras.get('stoch_neutral_max', 70.0))) if ti.stoch_k is not None else 0.5
        
    vol_mu = calculate_volume_membership(ti.volume_ratio,
        float(extras.get('volume_min_ratio', 0.8))) if ti.volume_ratio is not None else 0.5
        
    sma_mu = calculate_sma_distance_membership(ti.sma_distance,
        float(extras.get('sma_max_distance', 0.02))) if ti.sma_distance is not None else 0.5

    memberships = {
        "mtf": float(mtf_mu),
        "iv": float(iv_mu),
        "regime": float(regime_mu),
        "rsi": float(rsi_mu),
        "adx": float(adx_mu),
        "bbands": float(bb_mu),
        "stoch": float(stoch_mu),
        "volume": float(vol_mu),
        "sma": float(sma_mu),
    }
    
    weights = {
        "mtf": float(extras.get("w_mtf", 0.25)),
        "iv": float(extras.get("w_iv", 0.20)),
        "regime": float(extras.get("w_regime", 0.15)),
        "rsi": float(extras.get("w_rsi", 0.10)),
        "adx": float(extras.get("w_adx", 0.10)),
        "bbands": float(extras.get("w_bbands", 0.10)),
        "volume": float(extras.get("w_volume", 0.05)),
        "sma": float(extras.get("w_sma", 0.05)),
    }

    Ft = compute_fuzzy_confidence(memberships, weights)

    realized_vol = float(ti.realized_vol) if ti.realized_vol is not None else float(extras.get("realized_vol", 0.0))
    low_vol = float(extras.get("low_vol", 10.0))
    high_vol = float(extras.get("high_vol", 35.0))
    sigma_star = normalize_volatility(realized_vol, low_vol, high_vol)

    # Scale: combine Gaussian confidence and fuzzy Ft to avoid double-counting.
    fused_conf = _clamp((ti.gaussian_confidence * 0.60) + (Ft * 0.40), 0.0, 1.0)
    g = compute_scaling_factor(fused_conf, sigma_star, min_scale=float(extras.get("min_scale", 0.10)))

    # Check minimum requirements before applying scaling floor
    require_two_wings = bool(extras.get("require_two_wings", True))
    min_total_for_two = int(extras.get("min_total_qty_for_two_wings", 2))
    min_floor = min_total_for_two if require_two_wings else 1

    scaled_qty = int(total_qty * g)
    # Unconditionally enforce minimum floor for strategy type
    scaled_qty = max(min_floor, scaled_qty)

    diagnostics.update(
        {
            "total_qty_source": total_qty_source,
            "total_qty_pre_scale": total_qty,
            "memberships": memberships,
            "membership_weights": weights,
            "fuzzy_Ft": Ft,
            "realized_vol": realized_vol,
            "sigma_star": sigma_star,
            "fused_conf": fused_conf,
            "scaling_g": g,
            "total_qty_post_scale": scaled_qty,
        }
    )

    total_qty = scaled_qty

    # --- 3) Derive put/call weights and split quantities ---
    put_w, call_w, wdiag = _derive_put_call_weights(
        direction_probs=ti.direction_probs,
        gaussian_confidence=ti.gaussian_confidence,
    )
    diagnostics.update({"weighting": wdiag})

    if require_two_wings and total_qty < min_total_for_two:
        return SizingPlan(
            approved=False,
            reason=f"insufficient_total_qty_for_two_wings(min={min_total_for_two},got={total_qty})",
            total_qty=total_qty,
            put_qty=0,
            call_qty=0,
            put_weight=put_w,
            call_weight=call_w,
            diagnostics=diagnostics,
        )

    # Split with rounding and guardrails
    put_qty = int(round(total_qty * put_w))
    call_qty = int(total_qty - put_qty)

    if require_two_wings:
        # Ensure both are at least 1 by shifting if needed.
        if put_qty == 0 and total_qty >= 2:
            put_qty = 1
            call_qty = total_qty - put_qty
        if call_qty == 0 and total_qty >= 2:
            call_qty = 1
            put_qty = total_qty - call_qty

    # Ratio cap guardrail (default 3:1)
    max_ratio = float(extras.get("max_wing_ratio", 3.0))
    lo = min(put_qty, call_qty)
    hi = max(put_qty, call_qty)
    if require_two_wings and lo > 0 and (hi / lo) > max_ratio:
        # Compress toward the cap while preserving total_qty
        if put_qty > call_qty:
            call_qty = max(1, int(round(put_qty / max_ratio)))
            put_qty = total_qty - call_qty
        else:
            put_qty = max(1, int(round(call_qty / max_ratio)))
            call_qty = total_qty - put_qty

        diagnostics["ratio_capped"] = True
        diagnostics["max_wing_ratio"] = max_ratio

    diagnostics.update({"put_qty": put_qty, "call_qty": call_qty, "put_weight": put_w, "call_weight": call_w})

    return SizingPlan(
        approved=True,
        reason="approved",
        total_qty=total_qty,
        put_qty=put_qty,
        call_qty=call_qty,
        put_weight=put_w,
        call_weight=call_w,
        diagnostics=diagnostics,
    )
