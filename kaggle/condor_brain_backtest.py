"""
CondorBrain GPU Backtest - Kaggle Notebook
==========================================
Upload this to Kaggle along with:
1. condor_brain_e10_d1024_L24_lr1e04.pth (model weights)
2. mamba_institutional_1m.csv (test data)

Enable GPU: Settings -> Accelerator -> GPU T4 x2
"""

# =============================================================================
# CELL 1: Install Dependencies
# =============================================================================
# !pip install mamba-ssm causal-conv1d pandas numpy torch --quiet

# =============================================================================
# CELL 2: Setup & Imports
# =============================================================================
import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Clone repo
# !git clone https://github.com/trextrader/spy-iron-condor-trading.git repo
# %cd repo

# Add to path
sys.path.insert(0, '/kaggle/working/spy-iron-condor-trading')

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# =============================================================================
# CELL 3: Load Trained Model
# =============================================================================
from intelligence.condor_brain import CondorBrain

# Model configuration (must match training)
# Model configuration (must match training)
MODEL_CONFIG = {
    'd_model': 512,    # UPDATED: Matches condor_brain_seq_e1
    'n_layers': 12,    # UPDATED: Matches condor_brain_seq_e1
    'input_dim': 24,   # Renamed from n_features to match __init__
    'use_vol_gated_attn': True,
    'use_topk_moe': True,
    'moe_n_experts': 3,
    'moe_k': 1,
}

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CondorBrain(**MODEL_CONFIG).to(device)

# Enable Multi-GPU (DataParallel)
if torch.cuda.device_count() > 1:
    print(f"🚀 Multi-GPU Detected: {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)

# Load weights - update path for Kaggle
MODEL_PATH = "condor_brain_seq_e1.pth"
# Alternative: MODEL_PATH = "/kaggle/working/condor_brain_seq_e1.pth"
# Alternative: MODEL_PATH = "models/condor_brain_e10_d1024_L24_lr1e04.pth"

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint, strict=False)
model.eval()

print(f"✅ Model loaded on {device}")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

# =============================================================================
# CELL 4: Load Test Data
# =============================================================================
DATA_PATH = "/kaggle/input/spy-options-data/mamba_institutional_1m.csv"
# Alternative: DATA_PATH = "data/processed/mamba_institutional_1m.csv"

print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df):,} rows")

def add_technical_indicators(df):
    """Generate basic technical indicators using pandas (No TA-Lib dependency)."""
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    
    # SMA & Bollinger Bands
    df['sma'] = df['close'].rolling(window=20).mean()
    std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['sma'] + 2 * std
    df['bb_lower'] = df['sma'] - 2 * std
    
    # Stochastic K
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14 + 1e-8))
    
    # Fill remaining NaNs from rolling windows
    df.fillna(0, inplace=True)
    return df

print("   Generating Technical Indicators...")
df = add_technical_indicators(df)

# Feature columns (MUST MATCH TRAINING EXACTLY)
FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'volume', 'delta', 'gamma', 
    'vega', 'theta', 'iv', 'ivr', 'spread_ratio', 'te', 'rsi', 
    'atr', 'adx', 'bb_lower', 'bb_upper', 'stoch_k', 'sma', 
    'psar', 'strike', 'target_spot', 'max_dd_60m'
]

# Ensure columns exist
available_cols = [c for c in FEATURE_COLS if c in df.columns]
missing_cols = set(FEATURE_COLS) - set(available_cols)
if missing_cols:
    print(f"⚠️ Warning: Missing columns: {missing_cols}")
    print("   Attempting to generate basic indicators (Mocking Greeks)...")
    # Basic Feature Engineering for Backtest (if columns missing)
    # Greeks/IV might be missing in raw OHLCV, so we mock them for structural compliance
    for col in missing_cols:
        df[col] = 0.0 # Default pad
    
    available_cols = FEATURE_COLS # Now all exist (mocked or real)

print(f"Using {len(available_cols)} features (Target: {len(FEATURE_COLS)})")

# Slice Data to avoid OOM (Last 1M rows)
MAX_ROWS = 1_000_000
if len(df) > MAX_ROWS:
    print(f"✂️ Slicing data to last {MAX_ROWS:,} rows (User Requested Limit)...")
    df = df.iloc[-MAX_ROWS:].reset_index(drop=True)

# =============================================================================
# CELL 5: Prepare Sequences for Inference
# =============================================================================
LOOKBACK = 256  # UPDATED: Matches SEQ_LEN

def prepare_sequences(df, feature_cols, lookback=256):
    """Prepare feature sequences for model inference."""
    # Handle NaNs: Forward fill then fill 0
    df_clean = df[feature_cols].ffill().fillna(0.0)
    X = df_clean.values.astype(np.float32)
    
    # Normalize (same as training)
    # Use nanmedian to be robust against any artifacts
    median = np.nanmedian(X, axis=0, keepdims=True)
    mad = np.nanmedian(np.abs(X - median), axis=0, keepdims=True) + 1e-8
    X_norm = np.clip((X - median) / (1.4826 * mad), -10, 10)
    
    # Create sequences (Float16 saves 50% RAM)
    n_samples = len(X_norm) - lookback + 1
    sequences = np.zeros((n_samples, lookback, len(feature_cols)), dtype=np.float16)
    
    for i in range(n_samples):
        sequences[i] = X_norm[i:i+lookback]
    
    return sequences

print("Preparing sequences...")
sequences = prepare_sequences(df, available_cols, LOOKBACK)
print(f"Created {len(sequences):,} sequences of shape {sequences.shape}")

# =============================================================================
# CELL 6: Run GPU Inference
# =============================================================================
BATCH_SIZE = 512  # Kaggle T4 can handle this

@torch.no_grad()
def run_inference(model, sequences, batch_size=512, device='cuda'):
    """Generate predictions on GPU with verbose progress."""
    try:
        from tqdm.auto import tqdm
    except ImportError:
        def tqdm(x, **kwargs): return x

    model.eval()
    all_preds = []
    all_regimes = []
    
    n_batches = (len(sequences) + batch_size - 1) // batch_size
    
    print(f"🚀 Starting Inference on {n_batches} batches ({len(sequences):,} samples)...")
    
    for i in tqdm(range(n_batches), desc="Inference", unit="batch"):
        start = i * batch_size
        end = min(start + batch_size, len(sequences))
        
        batch = torch.tensor(sequences[start:end], device=device, dtype=torch.float32)
        
        # DataParallel returns tuple, standard returns tuple
        outputs, regime_probs, h_T, *_ = model(batch)
        
        all_preds.append(outputs.cpu().numpy())
        all_regimes.append(regime_probs.argmax(dim=-1).cpu().numpy())
        
    return np.concatenate(all_preds), np.concatenate(all_regimes)

print("Running GPU inference...")
import time
start_time = time.time()

predictions, regimes = run_inference(model, sequences, BATCH_SIZE, device)

elapsed = time.time() - start_time
throughput = len(sequences) / elapsed

print(f"✅ Inference complete!")
print(f"   Time: {elapsed:.1f}s")
print(f"   Throughput: {throughput:,.0f} samples/sec")
print(f"   Predictions shape: {predictions.shape}")

# =============================================================================
# CELL 7: Analyze Predictions
# =============================================================================
# Output columns
OUTPUT_COLS = [
    'call_offset_pct', 'put_offset_pct', 'wing_width',
    'dte', 'prob_of_profit', 'expected_roi', 'max_loss_pct', 'confidence'
]

# Create predictions DataFrame
pred_df = pd.DataFrame(predictions, columns=OUTPUT_COLS)

# Apply Activations (Model returns raw logits)
# 0-3: Offsets/DTE -> Softplus
for col in ['call_offset_pct', 'put_offset_pct', 'wing_width', 'dte']:
    pred_df[col] = np.log1p(np.exp(pred_df[col])) # Softplus

# 4: Prob Profit -> Sigmoid
pred_df['prob_of_profit'] = 1 / (1 + np.exp(-pred_df['prob_of_profit']))

# 5: Expected ROI -> Tanh
pred_df['expected_roi'] = np.tanh(pred_df['expected_roi'])

# 6: Max Loss -> Sigmoid
pred_df['max_loss_pct'] = 1 / (1 + np.exp(-pred_df['max_loss_pct']))

# 7: Confidence -> Sigmoid
pred_df['confidence'] = 1 / (1 + np.exp(-pred_df['confidence']))

pred_df['regime'] = regimes.astype(int)
pred_df['regime_label'] = pred_df['regime'].map({0: 'Low', 1: 'Normal', 2: 'High'})

print("\n📊 Prediction Statistics:")
print(pred_df.describe())

print("\n📈 Regime Distribution:")
print(pred_df['regime_label'].value_counts())

print("\n🎯 Confidence Statistics:")
print(f"   Mean confidence: {pred_df['confidence'].mean():.4f}")
print(f"   High confidence (>0.7): {(pred_df['confidence'] > 0.7).sum():,} signals")

# =============================================================================
# CELL 8: Simulate Trading Strategy
# =============================================================================
def simulate_backtest(predictions, df, lookback=240):
    """Simple backtest simulation using neural predictions."""
    
    # Align predictions with data
    aligned_df = df.iloc[lookback-1:].reset_index(drop=True)
    aligned_df = aligned_df.iloc[:len(predictions)].copy()
    
    # Add predictions
    for i, col in enumerate(OUTPUT_COLS):
        aligned_df[f'pred_{col}'] = predictions[:len(aligned_df), i]
    
    # Trading signals: enter when confidence > threshold
    CONFIDENCE_THRESHOLD = 0.5
    aligned_df['signal'] = aligned_df['pred_confidence'] > CONFIDENCE_THRESHOLD
    
    # Simulate returns (simplified)
    # In reality, you'd need full options data for actual P&L
    aligned_df['strategy_return'] = 0.0
    
    # For signals, use predicted expected_roi scaled by confidence
    mask = aligned_df['signal']
    aligned_df.loc[mask, 'strategy_return'] = (
        aligned_df.loc[mask, 'pred_expected_roi'] * 
        aligned_df.loc[mask, 'pred_confidence'] * 0.01  # Scale factor
    )
    
    # Calculate cumulative returns
    aligned_df['cumulative_return'] = (1 + aligned_df['strategy_return']).cumprod()
    
    # Calculate potential strikes (Approximation)
    aligned_df['put_strike'] = aligned_df['close'] * (1 - aligned_df['pred_put_offset_pct'])
    aligned_df['call_strike'] = aligned_df['close'] * (1 + aligned_df['pred_call_offset_pct'])
    
    # Executed trades log
    trades = aligned_df[aligned_df['signal']].copy()
    if not trades.empty:
        print(f"\n📝 Executed Trade Log (First 10 of {len(trades):,}):")
        trade_log = trades[[
            'close', 'put_strike', 'call_strike', 
            'pred_expected_roi', 'pred_confidence', 'pred_prob_of_profit'
        ]].head(10)
        print(trade_log.to_string())
    
    return aligned_df

print("\nRunning backtest simulation...")
results_df = simulate_backtest(predictions, pred_df, LOOKBACK)  # Pass pred_df fully processed

# Performance metrics
total_return = results_df['cumulative_return'].iloc[-1] - 1
n_trades = results_df['signal'].sum()
sharpe = results_df['strategy_return'].mean() / (results_df['strategy_return'].std() + 1e-8) * np.sqrt(252 * 390)

print(f"\n📈 Backtest Results:")
print(f"   Total Return: {total_return*100:.2f}%")
print(f"   Sharpe Ratio: {sharpe:.2f}")
print(f"   Number of Signals: {n_trades:,}")
if n_trades > 0:
    avg_roi = results_df.loc[results_df['signal'], 'pred_expected_roi'].mean()
    print(f"   Avg Predicted ROI: {avg_roi*100:.2f}%")
else:
    print(f"   Avg Predicted ROI: 0.00%")

# =============================================================================
# CELL 9: Save Results
# =============================================================================
OUTPUT_PATH = "condor_brain_predictions.csv"
results_df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Saved predictions to {OUTPUT_PATH}")

# Quick visualization (optional)
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Cumulative returns
axes[0, 0].plot(results_df['cumulative_return'])
axes[0, 0].set_title('Cumulative Returns')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Return')

# Confidence distribution
axes[0, 1].hist(results_df['pred_confidence'], bins=50, edgecolor='black')
axes[0, 1].set_title('Confidence Distribution')
axes[0, 1].set_xlabel('Confidence')

# Regime over time
axes[1, 0].scatter(range(len(results_df)), results_df['signal'].astype(int), alpha=0.1, s=1)
axes[1, 0].set_title('Trading Signals')
axes[1, 0].set_xlabel('Time')

# Predicted ROI distribution
axes[1, 1].hist(results_df['pred_expected_roi'], bins=50, edgecolor='black')
axes[1, 1].set_title('Predicted ROI Distribution')
axes[1, 1].set_xlabel('Expected ROI')

plt.tight_layout()
plt.savefig('backtest_analysis.png', dpi=150)
plt.show()

print("\n🎉 GPU Backtest Complete!")
