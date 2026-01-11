"""
CondorBrain Sweep Runner (Fast Multi-Config)

Loads preprocessed tensors from prepare_tensors.py and runs multiple
training configurations sequentially without reloading data.
"""
import sys
import os
import time
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from intelligence.condor_brain import CondorBrain, CondorLoss, HAS_MAMBA

def train_one_config(args, config, data_loaders, device):
    """Train a single configuration."""
    d_model = config['d_model']
    layers = config['layers']
    lr = config['lr']
    batch_size = config['batch_size']
    
    print(f"\n{'='*60}")
    print(f"TRAINING CONFIG: {d_model}d x {layers} layers | LR: {lr} | Batch: {batch_size}")
    print(f"{'='*60}")
    
    # Model
    model = CondorBrain(
        d_model=d_model,
        n_layers=layers,
        input_dim=24  # Standard feature count
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")
    
    criterion = CondorLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    
    train_loader, val_loader = data_loaders
    best_loss = float('inf')
    
    output_path = f"models/sweep/condor_d{d_model}_L{layers}_lr{lr:.0e}.pth"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        for batch_x, batch_y, batch_r in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_r = batch_r.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda'):
                outputs, regime_probs, _ = model(batch_x, return_regime=True, forecast_days=0)
                loss = criterion(outputs, batch_y, regime_probs, batch_r)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            n_batches += 1
            
        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y, batch_r in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                batch_r = batch_r.to(device, non_blocking=True)
                
                with autocast('cuda'):
                    outputs, regime_probs, _ = model(batch_x, return_regime=True, forecast_days=0)
                    loss = criterion(outputs, batch_y, regime_probs, batch_r)
                val_loss += loss.item()
                n_val_batches += 1
        
        train_loss /= max(n_batches, 1)
        val_loss /= max(n_val_batches, 1)
        scheduler.step()
        
        print(f"Epoch {epoch+1:2d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        if val_loss < best_loss and not np.isnan(val_loss):
            best_loss = val_loss
            torch.save(model.state_dict(), output_path)
            
    print(f"✓ Best Loss: {best_loss:.4f} -> {output_path}")
    return best_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensors", default="data/processed/condor_tensors.pt")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    
    if not os.path.exists(args.tensors):
        print(f"Tensors not found at {args.tensors}")
        return
        
    print(f"Loading {args.tensors}...")
    data = torch.load(args.tensors)
    
    # Create datasets ONCE
    train_ds = TensorDataset(data['X_train'], data['y_train'], data['r_train'])
    val_ds = TensorDataset(data['X_val'], data['y_val'], data['r_val'])
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=256, pin_memory=True, num_workers=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # SWEEP CONFIGURATIONS
    configs = [
        {'d_model': 256, 'layers': 12, 'lr': 1e-4, 'batch_size': 256},
        {'d_model': 512, 'layers': 16, 'lr': 5e-5, 'batch_size': 256},
        {'d_model': 512, 'layers': 24, 'lr': 5e-5, 'batch_size': 256},
        {'d_model': 1024, 'layers': 32, 'lr': 1e-5, 'batch_size': 256},
    ]
    
    results = []
    for config in configs:
        loss = train_one_config(args, config, (train_loader, val_loader), device)
        results.append({**config, 'loss': loss})
        
    print("\nSWEEP RESULTS:")
    for res in results:
        print(f"{res['d_model']}d/{res['layers']}L: {res['loss']:.4f}")

if __name__ == "__main__":
    import numpy as np
    main()
