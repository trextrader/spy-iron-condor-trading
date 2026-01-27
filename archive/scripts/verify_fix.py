import sys
import os
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from io import StringIO

# Add repo root to path
repo_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, repo_root)

# Import the module
try:
    from audit.export_learned_conditions import model_forward
except ImportError:
    # Handle case where audit is not a package or path issue
    sys.path.append(os.path.join(repo_root, "audit"))
    from export_learned_conditions import model_forward

def test_model_forward_no_print():
    # Mock model
    model = MagicMock(spec=nn.Module)
    # Output shape: [Batch=1, D=10]
    # We need 2 outputs? No, model(x) returns a tuple/list usually in this codebase?
    # export_learned_conditions.py:312: outputs = model(x); pol = outputs[0]
    
    dummy_output = torch.randn(1, 10)
    model.return_value = (dummy_output,)
    
    x = torch.randn(1, 52) # input dim doesn't matter much for mock
    
    # Capture stdout
    captured_output = StringIO()
    sys.stdout = captured_output
    
    try:
        model_forward(model, x)
    finally:
        sys.stdout = sys.__stdout__
        
    output = captured_output.getvalue()
    print(f"Captured Output: '{output}'")
    
    if "policy_head_dim" in output:
        print("FAIL: Debug print found!")
        sys.exit(1)
    else:
        print("PASS: No debug print found.")
        sys.exit(0)

if __name__ == "__main__":
    test_model_forward_no_print()
