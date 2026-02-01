import torch
import sys
import os
import numpy as np

# Adjust path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from intelligence.predicate_discovery import PredicateSelector, PredicateCombiner, TemplateType

def test_nested_logic_export():
    print("Testing Nested Logical Rule Extraction...")
    
    n_preds = 20
    n_chains = 5
    max_depth = 4
    
    combiner = PredicateCombiner(
        n_predicates=n_preds, 
        n_chains=n_chains, 
        max_chain_depth=max_depth
    )
    
    # Mock predicate names
    pred_names = [f"Rule_{i}" for i in range(n_preds)]
    
    # Mock attention: make each chain focus on specific predicates
    with torch.no_grad():
        combiner.chain_attention.zero_()
        # Chain 0: Rule 1 AND Rule 5
        combiner.chain_attention[0, 0, 1] = 10.0
        combiner.chain_attention[0, 1, 5] = 10.0
        combiner.logic_gates[0, 0] = -10.0 # AND
        
        # Chain 1: Rule 2 OR Rule 3
        combiner.chain_attention[1, 0, 2] = 10.0
        combiner.chain_attention[1, 1, 3] = 10.0
        combiner.logic_gates[1, 0] = 10.0 # OR
    
    logic_sets = combiner.get_logic_sets(pred_names)
    
    print("\nDiscovered Logical Sets (Nested):")
    for i, s in enumerate(logic_sets):
        print(f"  Set {i}: {s}")
    
    if len(logic_sets) >= 2:
        print("\nExport Logic: PASS")
    else:
        print("\nExport Logic: FAIL")

if __name__ == '__main__':
    try:
        test_nested_logic_export()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
