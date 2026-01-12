"""
intelligence/defuzzifier.py

Defuzzification & Risk Scaling Module.
Converts the fuzzy confidence score into a concrete sizing scalar (0.0 to 1.0)
adjusting for Realized Volatility and other risk factors.
"""

class Defuzzifier:
    def __init__(self, low_vol_threshold=10.0, high_vol_threshold=30.0):
        self.low_vol = low_vol_threshold
        self.high_vol = high_vol_threshold

    def normalize_volatility(self, realized_vol: float) -> float:
        """
        Normalize volatility into sigma_star [0, 1].
        Higher sigma_star = Higher Risk.
        """
        if self.high_vol <= self.low_vol:
            return 1.0
        
        sigma_star = (realized_vol - self.low_vol) / (self.high_vol - self.low_vol)
        return max(0.0, min(1.0, sigma_star))

    def defuzzify(self, confidence: float, realized_vol: float, risk_params: dict = None) -> float:
        """
        Calculate final scaling factor 'g'.
        
        Formula: g = Confidence * (1 - SigmaStar)
        
        This penalizes high confidence if the market is extremely volatile.
        """
        sigma_star = self.normalize_volatility(realized_vol)
        
        # Base scaling
        g = confidence * (1.0 - sigma_star)
        
        # Optional: Apply valid range
        # Min scale 0.1 to avoid trivial positions if confidence > 0
        min_scale = risk_params.get("min_scale", 0.0) if risk_params else 0.0
        
        if g < min_scale and g > 0.05: # Hysteresis
            g = min_scale
            
        return max(0.0, min(1.0, g))
