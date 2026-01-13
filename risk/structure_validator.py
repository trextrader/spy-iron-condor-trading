"""
risk/structure_validator.py

Structure Validator Module.
Ensures that complex option structures (like Iron Condors) adhere to 
logical and rigid invariants (e.g., Strike ordering, Width consistency).
"""
from core.dto import IronCondorLegs

class StructureValidator:
    def validate_iron_condor(self, legs: IronCondorLegs) -> bool:
        """
        Check Iron Condor Invariants:
        1. Long Put Strike < Short Put Strike
        2. Short Put Strike < Short Call Strike
        3. Short Call Strike < Long Call Strike
        4. Expirations Match (for standard IC)
        """
        lp = legs.long_put.strike
        sp = legs.short_put.strike
        sc = legs.short_call.strike
        lc = legs.long_call.strike
        
        # 1. Strike Ordering
        if not (lp < sp < sc < lc):
            return False, f"Invalid Strike Order: {lp} < {sp} < {sc} < {lc} failed"
            
        # 2. Expirations Match
        # Assuming legs have 'expiration' attribute
        # (DTO defines it as Any=None but typically date)
        e1 = legs.long_put.expiration
        e2 = legs.short_put.expiration
        e3 = legs.short_call.expiration
        e4 = legs.long_call.expiration
        
        if not (e1 == e2 == e3 == e4):
             return False, "Mismatched Expirations (Calendar spreads not supported in this validator)"
             
        # 3. Credit Positive
        if legs.net_credit <= 0:
            return False, f"Negative or Zero Credit: {legs.net_credit}"
            
        return True, "OK"
