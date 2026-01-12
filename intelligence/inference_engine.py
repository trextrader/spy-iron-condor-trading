"""
intelligence/inference_engine.py

Core Inference Engine for the FIS Pipeline.
Currently implements an Additive Fuzzy System (Weighted Sum) approach 
to aggregate membership degrees into a single confidence scalar.

Future upgrades:
- Mamdani Rule Matrix (If-Then)
- Neuro-Fuzzy adaption
"""
from typing import Dict

class InferenceEngine:
    def __init__(self, weights: Dict[str, float] = None):
        # Default Weights for Iron Condor Suitability
        # These prioritize Mean Reversion (RSI/ADX) + IV Rank (Premium)
        self.weights = weights or {
            "adx": 0.20,       # Trend Strength (Low is good)
            "rsi": 0.20,       # Overbought/Sold (Neutral is good)
            "iv_rank": 0.30,   # Implied Volatility (High is good)
            "mtf": 0.15,       # Multi-Timeframe Consensus
            "vol_regime": 0.15 # VIX/Bollinger Stability
        }

    def evaluate(self, memberships: Dict[str, Dict[str, float]]) -> float:
        """
        Aggregate fuzzy memberships into a single confidence score [0.0, 1.0].
        
        Input:
            memberships: nested dict, e.g. {'adx': {'ranging': 0.8, 'trending': 0.2}}
        
        Strategy:
            - We extract the 'favorable' membership from each indicator.
            - We compute the weighted sum of these favorable scores.
        """
        score = 0.0
        total_weight = 0.0

        # Mapping of indicator -> key for "Favorable Condition"
        # e.g. for ADX, we want "ranging"
        favorable_keys = {
            "adx": "ranging",
            "rsi": "neutral",
            "iv_rank": "high",   # We want High IV to sell premium
            "mtf": "neutral",    # or 'aligned' depending on definition. 
            # Note: MTF logic in fuzzy_engine might return different keys.
            # We need to align Fuzzifier output keys with Inference expectations.
        }

        for indicator, w in self.weights.items():
            if indicator not in memberships:
                continue

            # Extract the specific membership value we care about
            # For "Additive" logic, we usually pick the "Good" state.
            target_state = favorable_keys.get(indicator)
            
            # Handling generic/dynamic keys (fallback)
            val = 0.0
            ind_groups = memberships[indicator]
            
            if target_state and target_state in ind_groups:
                val = ind_groups[target_state]
            elif "score" in ind_groups: # Direct score support
                 val = ind_groups["score"]
            elif len(ind_groups) == 1: # Single value case
                val = list(ind_groups.values())[0]
            
            score += val * w
            total_weight += w

        if total_weight == 0:
            return 0.5 # Neutral fallback

        final_score = score / total_weight
        return max(0.0, min(1.0, final_score))
