# ============================================================
# CONDORBRAIN 2.2 INFERENCE DEMO (Sequence + Forecasting)
# ============================================================
import sys
sys.path.insert(0, '/kaggle/working/spy-iron-condor-trading')
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from intelligence.condor_brain import CondorBrain

print("🚀 Loading Trained Model (Sequence Mode)...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Configuration (MATCHING TRAINING)
D_MODEL = 512
N_LAYERS = 12
SEQ_LEN = 256

POSSIBLE_PATHS = [
    "condor_brain_retrain_e1.pth",                    # NEW CORRECT NAME (Priority 1)
    "models/condor_brain_retrain_e1.pth",             # User custom path
    "/kaggle/working/condor_brain_retrain_e1.pth",    # Kaggle Output
    "condor_brain_seq_e1.pth",                        # Legacy name
    "/kaggle/working/condor_brain_seq_e1.pth",        # Kaggle Output
    "/kaggle/input/condor-brain-seq-e1/condor_brain_seq_e1.pth", # Kaggle Input
]
SEQ_MODEL_PATH = "condor_brain_seq_e1.pth" # Default
import os
for p in POSSIBLE_PATHS:
    if os.path.exists(p):
        SEQ_MODEL_PATH = p
        print(f"   Found model at: {SEQ_MODEL_PATH}")
        break

# 2. Load Checkpoint (supports both raw state_dict and dict checkpoint)
try:
    print(f"   Loading {SEQ_MODEL_PATH}...")
    ckpt = torch.load(SEQ_MODEL_PATH, map_location="cpu")
except FileNotFoundError:
    print(f"❌ '{SEQ_MODEL_PATH}' not found. Did training finish?")
    sys.exit()

# Normalize checkpoint format
if isinstance(ckpt, dict) and "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
    ckpt_feature_cols = ckpt.get("feature_cols", None)
    ckpt_median = ckpt.get("median", None)
    ckpt_mad = ckpt.get("mad", None)
    ckpt_seq_len = int(ckpt.get("seq_len", SEQ_LEN))
    ckpt_input_dim = int(ckpt.get("input_dim", 24))
    ckpt_use_diffusion = bool(ckpt.get("use_diffusion", False))
    ckpt_diffusion_steps = int(ckpt.get("diffusion_steps", 0))
else:
    state_dict = ckpt
    ckpt_feature_cols = None
    ckpt_median = None
    ckpt_mad = None
    ckpt_seq_len = SEQ_LEN
    ckpt_input_dim = 24
    ckpt_use_diffusion = False
    ckpt_diffusion_steps = 0

SEQ_LEN = ckpt_seq_len

# 3. Initialize Model
model = CondorBrain(
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    input_dim=ckpt_input_dim,
    use_vol_gated_attn=True,
    use_topk_moe=True,
    moe_n_experts=3, moe_k=1,
    use_diffusion=ckpt_use_diffusion,
    diffusion_steps=ckpt_diffusion_steps if ckpt_use_diffusion else 0,
).to(device)

print("   Loading weights into model...")
model.load_state_dict(state_dict, strict=True)
model.eval()
print("✅ Model Loaded Successfully")

# 4. Prepare Data (Mock or Real)
DATA_PATH = "/kaggle/input/spy-options-data/mamba_institutional_1m.csv"
try:
    df = pd.read_csv(DATA_PATH).iloc[-1000:]
    print(f"   Loaded {len(df)} rows from dataset.")
except:
    print("   ⚠️ No CSV found, generating synthetic noise for demo...")
    df = pd.DataFrame(np.random.randn(1000, 24), columns=['col'+str(i) for i in range(24)])

# Features
FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume', 'delta', 'gamma', 
                'vega', 'theta', 'iv', 'ivr', 'spread_ratio', 'te', 'rsi', 
                'atr', 'adx', 'bb_lower', 'bb_upper', 'stoch_k', 'sma', 
                'psar', 'strike', 'target_spot', 'max_dd_60m']

# If checkpoint provided the exact training feature columns, prefer them
if ckpt_feature_cols is not None:
    FEATURE_COLS = list(ckpt_feature_cols)

# Robust Norm (prefer training scalers if available)
X = df[FEATURE_COLS].values.astype(np.float32)
X = np.nan_to_num(X, nan=0.0)
if ckpt_median is not None and ckpt_mad is not None:
    median = np.asarray(ckpt_median, dtype=np.float32).reshape(1, -1)
    mad = np.asarray(ckpt_mad, dtype=np.float32).reshape(1, -1)
else:
    median = np.median(X, axis=0, keepdims=True).astype(np.float32)
    mad = (np.median(np.abs(X - median), axis=0, keepdims=True) + 1e-8).astype(np.float32)

X_norm = np.clip((X - median) / (1.4826 * mad), -10.0, 10.0).astype(np.float32)

# Take last Sequence
input_seq = X_norm[-SEQ_LEN:] # (256, 24)
input_tensor = torch.tensor(input_seq, device=device).unsqueeze(0) # (1, 256, 24)

# 5. Run Full Inference
print("\n🔮 Running CondorBrain 2.2 Inference...")

with torch.no_grad():
    # A) Policy & Internal State
    # Returns tuple: (outputs, regime, horizon, features, experts) or just outputs depending on flags
    # We call with ALL flags to see everything
    res = model(input_tensor, return_regime=True, return_experts=True, return_features=True)
    
    # ---- Safe unpacking (handles diffusion-enabled tuples) ----
    policy_out = res[0]       # (1, 8)
    regime_logits = res[1]    # (1, 3) or None
    horizon_dummy = res[2]    # (None or dict)
    next_feat = res[3] if len(res) > 3 else None  # (1, 4)

    diffusion_out = None
    experts = None
    if getattr(model, "use_diffusion", False):
        diffusion_out = res[4] if len(res) > 4 else None
        experts = res[5] if len(res) > 5 else None
    else:
        experts = res[4] if len(res) > 4 else None

    # B) 32-Bar Auto-Regressive Forecast (Meta-Forecaster Mode)
    # We need to reshape input to (T, 4) for the helper, 
    # BUT the helper expects raw features if it does transformations.
    # Our helper `predict_next_state` assumes 'x' matches model input dim for autoregression.
    # Since model input is 24-dim and we predict 4-dim, strict autoregression requires a projector.
    # For this demo, we can just run the simple loop if supported or show single-step.
    
    # Simplification: We predict 32 steps of the 4 features assuming constant background for others (Zero-Order Hold)
    # This is handled by a specialized loop if inside model, but here we just show Single Step Feature Forecast.
    print("   Generating Multi-Step Feature Forecast (32 bars)...")
    # For visual demo, we simulate 32 steps by feeding prediction back
    # NOTE: This is naive because we only predict 4 dims out of 24.
    # Real MetaForecaster implementation manages the full state.
    # Here we just show the immediate next-step vector.
    
    forecast_32 = []
    curr_seq = input_tensor
    for _ in range(32):
        # Predict next 4 features
        f = model(curr_seq, return_regime=False, return_features=True)[3] # (1, 4)
        forecast_32.append(f.cpu().numpy()[0])
        
        # Project 4 dims back to 24 for valid input (Naive padding)
        # In reality, you'd update OHLCV cols and recompute indicators.
        # This is just a tensor shape demo.
        f_expanded = torch.zeros((1, 1, input_tensor.shape[2]), device=device)
        f_expanded[:,:,:4] = f.unsqueeze(1) # Fill first 4 cols
        
        curr_seq = torch.cat([curr_seq, f_expanded], dim=1)[:, -SEQ_LEN:, :]

# 6. Visualize Outputs
policy = policy_out.cpu().numpy()[0]
regime_probs = torch.softmax(regime_logits, dim=-1).cpu().numpy()[0]
forecast_32 = np.array(forecast_32)

print("\n" + "="*50)
print("🦅 CONDORBRAIN POLICY OUTPUT (The Trade)")
print("="*50)
print(f"Call Offset : ${policy[0]:.2f}")
print(f"Put Offset  : ${policy[1]:.2f}")
print(f"Wing Width  : ${policy[2]:.2f}")
print(f"Target DTE  : {policy[3]:.1f} Days")
print(f"Prob Profit : {policy[4]*100:.1f}%")
print(f"Confidence  : {policy[7]*100:.1f}%")

print("\n" + "="*50)
print("🧠 BRAIN STATE (The Reasoning)")
print("="*50)
print(f"Regime      : Low({regime_probs[0]:.2f}) | Normal({regime_probs[1]:.2f}) | High({regime_probs[2]:.2f})")
if experts is not None and 'routing_weights' in experts:
    print(f"Active Exp  : {experts['routing_weights'].argmax().item()} (Strength: {experts['routing_weights'].max().item():.2f})")
else:
    print("Active Exp  : [Data Unavailable - Restart Kernel to Refresh Code]")

print("\n" + "="*50)
print("📈 META-FORECAST (Next 32 Bars)")
print("="*50)
print(f"Next Close Return : {forecast_32[0,0]:.6f} (Exp: {np.exp(forecast_32[0,0]):.4f}x)")
print(f"Volatility Proxy  : {forecast_32[0,1]:.6f}")
print("Trajectory Head   :")
# Simple text plot of return trajectory
for i, val in enumerate(forecast_32[:, 0]):
    bar = "+" * int(val * 1000) if val > 0 else "-" * int(abs(val) * 1000)
    print(f"   T+{i:02d}: {val:+.4f} | {bar}")

print("\n✅ Inference Complete.")
