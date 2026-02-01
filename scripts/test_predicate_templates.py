import torch
import sys
import os

# Adjust path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from intelligence.predicate_discovery import PredicateSelector, TemplateType
from intelligence.condor_brain import CondorBrain

def test_predicate_templates():
    print("Testing PredicateSelector with Templates...")
    selector = PredicateSelector(n_slots=100, max_active=10, n_fields=54)
    
    # Force some template types for testing
    # 0=GEN, 1=MOM, 2=REL, 3=CROSS, 4=THRESH
    with torch.no_grad():
        selector.template_head.weight.zero_()
        selector.template_head.bias.fill_(-10) # Default to GENERAL
        selector.template_head.bias[1] = 10     # Force MOMENTUM on first slots
        
    importance, params = selector(return_params=True)
    
    # Check if first rule is MOMENTUM
    t_idx = params[0, 11].item()
    print(f"Slot 0 template: {TemplateType(t_idx).name}")
    
    # Check constraints: for MOMENTUM, l_f1 should equal r_f1
    if t_idx == TemplateType.MOMENTUM:
        l_f1 = params[0, 0].item()
        r_f1 = params[0, 6].item()
        print(f"MOMENTUM check: l_f1={l_f1}, r_f1={r_f1} -> {'PASS' if l_f1 == r_f1 else 'FAIL'}")

    # Check active predicates formatting
    preds, imps, names = selector.get_active_predicates(threshold=-1, max_return=5)
    print("\nSample Formatted Rules:")
    for n in names:
        print(f"  {n}")

def test_condorbrain_integration():
    print("\nTesting CondorBrain with Predicate Discovery...")
    model = CondorBrain(
        input_dim=54,
        use_predicate_discovery=True,
        max_active_predicates=64,
        use_cde=False # Faster for smoke test
    )
    
    batch = 2
    seq_len = 32
    x = torch.randn(batch, seq_len, 54)
    
    outputs = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {outputs[0].shape}")
    print(f"Backbone input dim (expected 54+64=118): {model.input_proj.in_features if not model.use_cde else 'N/A'}")
    
    if outputs[0].shape == (batch, 10):
        print("Integration: PASS")
    else:
        print("Integration: FAIL")

if __name__ == '__main__':
    try:
        test_predicate_templates()
        test_condorbrain_integration()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
