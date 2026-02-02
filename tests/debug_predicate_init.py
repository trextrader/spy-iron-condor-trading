
import torch
import sys
import os
import numpy as np

# Add project root
sys.path.append(os.getcwd())

from intelligence.predicate_discovery import PredicateSelector, CompareOp, COMPARE_SYMBOLS

def check_initialization_bias():
    print("Initializing PredicateSelector with default params...")
    selector = PredicateSelector(n_slots=1000)
    
    print("Running forward pass (return_params=True) in EVAL mode...")
    selector.eval()
    with torch.no_grad():
        importance, params = selector.forward(return_params=True)
        
    print(f"Params shape: {params.shape}")
    
    # Extract Compare Op (index 5)
    cmp_ops = params[:, 5].numpy().astype(int)
    
    print("\nCompare Operator Distribution:")
    unique, counts = np.unique(cmp_ops, return_counts=True)
    for val, count in zip(unique, counts):
        sym = COMPARE_SYMBOLS[val] if 0 <= val < len(COMPARE_SYMBOLS) else "UNK"
        print(f"  Op {val} ('{sym}'): {count} ({count/len(cmp_ops)*100:.1f}%)")
        
    # Extract Templates (index 11)
    if params.shape[1] > 11:
        templates = params[:, 11].numpy().astype(int)
        print("\nTemplate Distribution:")
        unique_t, counts_t = np.unique(templates, return_counts=True)
        for val, count in zip(unique_t, counts_t):
            print(f"  Template {val}: {count} ({count/len(templates)*100:.1f}%)")

if __name__ == "__main__":
    check_initialization_bias()
