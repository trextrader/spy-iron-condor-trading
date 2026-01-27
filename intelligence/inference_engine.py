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
        # 11-Factor Hybrid Weights (Neural CDE + 10 Technical Factors)
        # Neural CDE is the primary driver (~30%), technically confirmed by audit.
        self.weights = weights or {
            "neural_cde": 0.30,  # Continuous-time model confidence
            "mtf": 0.10,         # Multi-timeframe consensus
            "iv_rank": 0.10,     # Vol percentile
            "vix": 0.08,         # Market regime
            "rsi": 0.08,         # Momentum
            "adx": 0.08,         # Trend strength
            "bbands": 0.07,      # Volatility bands
            "stoch": 0.06,       # Oscillator
            "volume": 0.05,      # Liquidity
            "sma_dist": 0.04,    # Equilibrium
            "psar": 0.04         # Reversal
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
            "neural_cde": "favorable",
            "adx": "ranging",
            "rsi": "neutral",
            "iv_rank": "high",   
            "mtf": "neutral",    
            "vix": "stable",
            "bbands": "neutral",
            "stoch": "neutral",
            "volume": "high",
            "sma_dist": "neutral",
            "psar": "favorable"
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
