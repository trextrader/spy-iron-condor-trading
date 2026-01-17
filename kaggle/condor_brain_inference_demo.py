# ============================================================
# CONDORBRAIN INFERENCE DEMO (Run after training)
# ============================================================
import sys
sys.path.insert(0, '/kaggle/working/spy-iron-condor-trading')
import torch
import pandas as pd
import numpy as np
from intelligence.condor_brain import CondorBrain

print("🚀 Loading Trained Model...")
device = torch.device('cuda')

# 1. Initialize Architecture (Must match training config!)
model = CondorBrain(
    d_model=512,       # Match FAST config
    n_layers=12,       
    input_dim=24,
    use_vol_gated_attn=True,
    use_topk_moe=True,
    moe_n_experts=3, moe_k=1
).to(device)

# 2. Load Weights (Adjust filename to best epoch)
MODEL_PATH = "condor_brain_fast_e1.pth" 
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"✅ Loaded {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ '{MODEL_PATH}' not found yet. Run training first!")
    sys.exit()

# 3. Create Sample Input (Last 10 sequences from data)
# In production, use real recent data
DATA_PATH = "/kaggle/input/spy-options-data/mamba_institutional_1m.csv"
df = pd.read_csv(DATA_PATH).iloc[-500:] 
FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume', 'delta', 'gamma', 
                'vega', 'theta', 'iv', 'ivr', 'spread_ratio', 'te', 'rsi', 
                'atr', 'adx', 'bb_lower', 'bb_upper', 'stoch_k', 'sma', 
                'psar', 'strike', 'target_spot', 'max_dd_60m']

X = df[FEATURE_COLS].values.astype(np.float32)
X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)

# Preprocessing (Same as training)
median = np.median(X, axis=0, keepdims=True)
mad = np.median(np.abs(X - median), axis=0, keepdims=True) + 1e-8
X_norm = np.clip((X - median) / (1.4826 * mad), -10, 10)

# 4. Predict
print("\n🔮 Generating Predictions...")
input_seq = torch.tensor(X_norm[-240:], device=device).unsqueeze(0) # [1, 240, 24]

with torch.no_grad():
    # Toggle 'return_experts=True' to see which expert creates the condor!
    outputs, regime_logits, _, experts = model(input_seq, return_regime=True, return_experts=True)

# 5. Display Results
preds = outputs.cpu().numpy()[0]
regime_probs = torch.softmax(regime_logits, dim=-1).cpu().numpy()[0]

print(f"\n📊 CONDORBRAIN SIGNAL:")
print(f"   Call Offset : ${preds[0]:.2f}")
print(f"   Put Offset  : ${preds[1]:.2f}")
print(f"   Wing Width  : ${preds[2]:.2f}")
print(f"   DTE Model   : {preds[3]:.1f} days")
print(f"   Prob Profit : {preds[4]*100:.1f}%")
print(f"   Conviction  : {preds[7]*100:.1f}%")

print(f"\n🧠 INTERNAL STATE:")
print(f"   Regime: Low={regime_probs[0]:.2f}, Normal={regime_probs[1]:.2f}, High={regime_probs[2]:.2f}")
if 'routing_weights' in experts:
    print(f"   Expert Choice: {experts['routing_weights'].argmax().item()} (Weight: {experts['routing_weights'].max().item():.2f})")
