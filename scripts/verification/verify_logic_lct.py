
import pandas as pd
import numpy as np
from intelligence.rule_engine.executor import LogicEvaluator

def run_lct_audit():
    print("=== Sub-Phase 3.1: Logic Consistency Test (LCT) ===")
    
    # Setup context
    idx = pd.date_range("2025-01-01", periods=10, freq="min")
    context = {
        "a": pd.Series([True, True, False, False, True, True, False, False, True, True], index=idx),
        "b": pd.Series([True, False, True, False, True, False, True, False, True, False], index=idx),
        "c": pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], index=idx),
        "d": {"score": pd.Series([0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.1, 0.1, 0.5, 0.5], index=idx)}
    }
    
    evaluator = LogicEvaluator(context)
    
    # Test 1: Simple Comparison
    print("\n[1] Simple Comparison: 'c > 0.5'")
    res1 = evaluator.evaluate("c > 0.5")
    expected1 = context["c"] > 0.5
    print(f"  Match: {res1.equals(expected1)}")
    
    # Test 2: Simple AND
    print("\n[2] Simple AND: 'AND(a, b)'")
    res2 = evaluator.evaluate("AND(a, b)")
    expected2 = context["a"] & context["b"]
    print(f"  Match: {res2.equals(expected2)}")
    
    # Test 3: Nested Logic (Vulnerability Check)
    # The current string split logic will likely fail on 'AND(a, OR(b, c > 0.5))'
    # because it splits by ',' and finds 'a', ' OR(b', ' c > 0.5))'
    print("\n[3] Nested Logic: 'AND(a, OR(b, c > 0.5))'")
    try:
        res3 = evaluator.evaluate("AND(a, OR(b, c > 0.5))")
        expected3 = context["a"] & (context["b"] | (context["c"] > 0.5))
        print(f"  Match: {res3.equals(expected3)}")
    except Exception as e:
        print(f"  CRASH: {e}")

    # Test 4: Attribute Access
    print("\n[4] Attribute Access: 'd.score > 0.8'")
    res4 = evaluator.evaluate("d.score > 0.8")
    expected4 = context["d"]["score"] > 0.8
    print(f"  Match: {res4.equals(expected4)}")

if __name__ == "__main__":
    run_lct_audit()
