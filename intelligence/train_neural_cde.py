import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

# Add project root to path
sys.path.insert(0, os.getcwd())

from intelligence.models.neural_cde import NeuralCDE
from intelligence.canonical_feature_registry import FEATURE_COLS_V22, apply_semantic_nan_fill

# ============================================================
# CONFIG
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN = 256
PREDICT_HORIZON = 45 # Same as CondorBrain targets

def load_and_prep_data(data_path):
    print(f"Loading {data_path}...")
    df = pd.read_csv(data_path)
    
    # Simple feature prep
    feature_cols = FEATURE_COLS_V22
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
            
    X = df[feature_cols].values.astype(np.float32)
    X = apply_semantic_nan_fill(X, feature_cols)
    
    # Robust Normalization
    median = np.median(X, axis=0)
    mad = np.median(np.abs(X - median), axis=0)
    mad = np.maximum(mad, 1e-6)
    X = (X - median) / (1.4826 * mad)
    X = np.clip(X, -10.0, 10.0)
    
    # Targets: [Future Return, Future Vol] simplification for prototype
    # Just fitting the "Next Step" or "Horizon" to prove CDE learning
    Y = np.zeros((len(X), 2), dtype=np.float32)
    
    # 1. 5-day return
    future_close = df['close'].shift(-5).ffill()
    Y[:, 0] = (future_close / df['close'] - 1.0) * 100.0
    
    # 2. 5-day volatility change
    future_vol = df['iv'].shift(-5).ffill()
    Y[:, 1] = future_vol - df['iv']
    
    return X, Y, feature_cols

def make_sequences(X, Y, seq_len):
    # Simple sliding window
    # For speed, just random sample indices for training
    return X, Y

class CDETrainer:
    def __init__(self, model, lr=1e-3):
        self.model = model.to(DEVICE)
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
    def train_epoch(self, X, Y, batch_size=512, steps=1000):
        self.model.train()
        total_loss = 0
        
        # Random sampling loader
        max_idx = len(X) - SEQ_LEN - 1
        
        pbar = tqdm(range(steps), desc="Training CDE")
        for _ in pbar:
            idxs = np.random.randint(0, max_idx, size=batch_size)
            
            # Construct batch: (B, T, D)
            batch_x = []
            batch_y = []
            
            for idx in idxs:
                batch_x.append(X[idx : idx+SEQ_LEN])
                batch_y.append(Y[idx+SEQ_LEN]) # Target at T+1
                
            x_tensor = torch.tensor(np.array(batch_x), device=DEVICE)
            y_tensor = torch.tensor(np.array(batch_y), device=DEVICE)
            
            self.optimizer.zero_grad()
            
            # CDE Forward: Expects (B, T, D)
            # Returns final state z_T: (B, hidden)
            z_final = self.model(x_tensor)
            
            # Decode to output
            preds = self.model.decoder(z_final)
            
            loss = self.loss_fn(preds, y_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})
            
        return total_loss / steps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    
    print(f"🚀 Starting Neural CDE Prototype Training on {DEVICE}")
    
    # 1. Data
    X, Y, feats = load_and_prep_data(args.data)
    input_dim = X.shape[1]
    output_dim = 2 # Return, Vol
    
    print(f"Data shape: {X.shape}. Input dim: {input_dim}")
    
    # 2. Model
    # Hidden state z evolves. Decoder maps z_T -> Prediction
    model = NeuralCDE(input_dim, args.hidden, output_dim, n_layers=args.layers)
    trainer = CDETrainer(model)
    
    # 3. Train
    for epoch in range(args.epochs):
        avg_loss = trainer.train_epoch(X, Y)
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.6f}")
        
        # Save Checkpoint
        ckpt_path = f"models/neural_cde_proto_e{epoch+1}.pth"
        torch.save({
            'state_dict': model.state_dict(),
            'config': vars(args),
            'epoch': epoch + 1,
            'loss': avg_loss
        }, ckpt_path)
        print(f"💾 Saved {ckpt_path}")
        
    # 4. Save
    save_path = "models/neural_cde_proto.pth"
    torch.save({
        'state_dict': model.state_dict(),
        'config': vars(args),
        'timestamp': time.time()
    }, save_path)
    print(f"✅ Model saved to {save_path}")

if __name__ == "__main__":
    main()
