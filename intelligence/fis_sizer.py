"""
intelligence/fis_sizer.py

Orchestrator for the Quantor-MTFuzz™ Sizing Logic.
Integrates Fuzzifier, InferenceEngine, and Defuzzifier to produce
risk-adjusted position sizes based on capital constraints and market regime.
"""
import math
from config import StrategyConfig
from core.dto import MarketSnapshot, TradeDecision, SizedDecision
from intelligence.fuzzifier import Fuzzifier
from intelligence.inference_engine import InferenceEngine
from intelligence.defuzzifier import Defuzzifier
import intelligence.fuzzy_engine as fe

class FISSizer:
    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg
        self.fuzzifier = Fuzzifier(cfg)
        self.inference = InferenceEngine() # Uses default weights
        self.defuzzifier = Defuzzifier(
            low_vol_threshold=10.0, 
            high_vol_threshold=getattr(cfg, 'regime_vix_high', 30.0)
        )

    def size_trade(self, decision: TradeDecision, snapshot: MarketSnapshot, account_equity: float, model_outputs: dict = None) -> SizedDecision:
        """
        Execute the full sizing pipeline.
        
        Phase 1: Hard Capital Constraints (Max Loss Ceiling)
        Phase 2: Feature Extraction & Fuzzification
        Phase 3: Inference (Confidence Score)
        Phase 4: Defuzzification (Volatility Scaling)
        Phase 5: Final Sizing
        """
        if not getattr(self.cfg, "allow_fis_legacy", False):
            return self._reject(decision, "fis_legacy_disabled")
        
        # --- Stage 1: Hard Constraints ---
        # "Never bet more than X% of equity on max loss"
        max_risk_dollars = account_equity * self.cfg.risk_per_trade_pct
        
        # We need the leg structure to know max loss.
        # But wait - TradeDecision doesn't have legs yet! 
        # The legs are generated temporarily in 'generate_trade_signal' but passed separately.
        # Ideally, Sizer should know about the intended legs to calculate Max Loss.
        # However, for pure signal logic, we might estimate max loss or require the legs be passed.
        
        # CRITICAL Refactor Note:
        # In the new slim architecture, 'generate_trade_signal' returns (Decision, Legs, Credit).
        # We should pass 'Legs' to this function too, or extract Max Loss from Decision metadata if available.
        # For now, let's assume we calculate 'Max Quantity' based on a naive estimate 
        # or that 'decision.rationale' might contain 'width'.
        
        wing_width = decision.rationale.get('width', 5.00)
        credit = decision.rationale.get('credit', 1.00)
        max_loss_per_share = wing_width - credit
        max_loss_per_contract = max_loss_per_share * 100.0
        
        q0 = 0
        if max_loss_per_contract > 0:
            q0 = math.floor(max_risk_dollars / max_loss_per_contract)
        
        # Cap at global max
        q0 = min(q0, self.cfg.max_contracts_per_trade)
        
        if q0 <= 0:
             return self._reject(decision, "Insufficient Capital or Zero Width")

        # --- Stage 2: Fuzzification ---
        features_df = self.fuzzifier.extract_features(snapshot)
        memberships = self.fuzzifier.fuzzify(features_df)
        
        # Inject Neural CDE Factor (11th Factor)
        if model_outputs:
            model_confidence = model_outputs.get("confidence", 0.5)
            prob_profit = model_outputs.get("prob_profit", 0.5)
            memberships["neural_cde"] = {
                "favorable": fe.calculate_model_membership(model_confidence, prob_profit)
            }
        
        # --- Stage 3: Inference ---
        confidence = self.inference.evaluate(memberships)
        
        # --- Stage 4: Defuzzification (Vol Scaling) ---
        # Get Realized Vol from snapshot bars if available, or approximate with VIX/IV
        # Fuzzifier extracts IV Rank, but Defuzzifier wants Realized Vol (sigma).
        # We can use VIX as a proxy for now if realized vol calc is expensive to re-do.
        vix = snapshot.vix or 15.0
        realized_vol_proxy = vix # Simple proxy
        
        scaling_factor = self.defuzzifier.defuzzify(confidence, realized_vol_proxy)
        
        # --- Stage 5: Final Sizing ---
        final_qty = int(q0 * scaling_factor)
        
        return SizedDecision(
            decision=decision,
            contracts=final_qty,
            confidence=confidence,
            risk_budget=final_qty * max_loss_per_contract,
            legs=[] # We don't modify legs here, but the DTO expects them. Use empty or pass through if provided.
        )
        
    def _reject(self, decision, reason):
        return SizedDecision(decision, 0, 0.0, 0.0, [])
