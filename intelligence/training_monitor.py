"""
CondorBrain Training Monitor: Per-Head Metrics & In-Memory Checkpointing
(Production-Hardened Version)

Performance Features:
- NO GPU sync points in training loop
- Throttled plotting (every N epochs, not every epoch)
- File-based PNG output (non-blocking, Colab-stable)
- Per-head tracking for analysis (8 outputs + regime)
- Best-global checkpoint for model restore
- Best-per-head for reporting only
"""
import copy
import time
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# Output head names - MUST match CondorBrain output[:, i] order exactly
# Model outputs (B, 8): [call_offset, put_offset, wing_width, dte, pop, roi, max_loss, confidence]
# Training targets match same order in prepare_features()
MAIN_HEADS = [
    'call_offset', 'put_offset', 'wing_width', 'dte',
    'pop', 'roi', 'max_loss', 'confidence'
]
# Extra tracked metrics (not direct model outputs)
EXTRA_HEADS = ['regime_accuracy']
HEAD_NAMES = MAIN_HEADS + EXTRA_HEADS


@dataclass
class HeadMetrics:
    """Metrics for a single output head."""
    name: str
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    best_val_loss: float = float('inf')
    best_epoch: int = 0


@dataclass
class TrainingState:
    """Complete training state for checkpointing."""
    epoch: int
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]
    scheduler_state_dict: Optional[Dict[str, Any]]
    train_loss: float
    val_loss: float
    head_val_losses: Dict[str, float]
    

class TrainingMonitor:
    """
    Production-hardened training monitor with per-head tracking.
    
    ARCHITECTURE DECISION:
    - Best-global checkpoint: Used for early stopping + final model save
    - Best-per-head tracking: For analysis/reporting ONLY (can't restore 8 different heads)
    
    PERFORMANCE:
    - No .item() calls in training loop
    - Plotting throttled to every N epochs
    - Uses file-based PNG output (non-blocking)
    """
    
    def __init__(self, checkpoint_capacity: int = 3, plot_dir: str = "monitor_plots"):
        """
        Args:
            checkpoint_capacity: How many recent checkpoints to keep in memory
            plot_dir: Directory for plot PNG files (non-blocking output)
        """
        self.capacity = checkpoint_capacity
        self.plot_dir = plot_dir
        
        # Per-head metrics
        self.heads: Dict[str, HeadMetrics] = {
            name: HeadMetrics(name=name) for name in HEAD_NAMES
        }
        
        # Global metrics
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_val_loss: float = float('inf')
        self.best_epoch: int = 0
        
        # In-memory checkpoints (circular buffer of recent states)
        self._checkpoints: List[TrainingState] = []
        self._best_checkpoint: Optional[TrainingState] = None
        
        # Best per-head tracking (FOR REPORTING ONLY - cannot restore these separately)
        self._best_per_head: Dict[str, int] = {}  # head_name -> best_epoch
        
        # Visualization state
        self._fig = None
        self._axes = None
        self._last_plot_time = 0
        
    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        head_val_losses: Dict[str, float],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Update monitor with epoch results and checkpoint if improved.
        
        CRITICAL: This is called ONCE per epoch, AFTER validation.
        All values passed here should already be Python floats (no GPU tensors).
        
        Returns:
            Dict with status info including which heads improved
        """
        # Update global metrics
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        # Update per-head metrics (FOR REPORTING)
        improved_heads = []
        for name, loss in head_val_losses.items():
            if name not in self.heads:
                continue
            head = self.heads[name]
            head.val_losses.append(loss)
            
            if loss < head.best_val_loss:
                head.best_val_loss = loss
                head.best_epoch = epoch
                self._best_per_head[name] = epoch
                improved_heads.append(name)
        
        # Only checkpoint when GLOBAL loss improves (this is what we restore)
        improved_global = val_loss < self.best_val_loss
        
        if improved_global:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            
            # Create checkpoint (deep copy of state dicts)
            state = TrainingState(
                epoch=epoch,
                model_state_dict=copy.deepcopy(model.state_dict()),
                optimizer_state_dict=copy.deepcopy(optimizer.state_dict()),
                scheduler_state_dict=copy.deepcopy(scheduler.state_dict()) if scheduler else None,
                train_loss=train_loss,
                val_loss=val_loss,
                head_val_losses=head_val_losses.copy()
            )
            
            # Update best checkpoint
            self._best_checkpoint = state
            
            # Also keep in circular buffer
            self._checkpoints.append(state)
            if len(self._checkpoints) > self.capacity:
                self._checkpoints.pop(0)
        
        return {
            'improved_global': improved_global,
            'improved_heads': improved_heads,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
        }
    
    def get_best_checkpoint(self) -> Optional[TrainingState]:
        """Get the globally best checkpoint (for restoring model)."""
        return self._best_checkpoint
    
    def restore_best(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None) -> bool:
        """
        Restore model to globally best checkpoint.
        
        NOTE: This restores the GLOBAL best, not per-head best.
        Per-head best epochs are for analysis only.
        """
        if self._best_checkpoint is None:
            print("[Monitor] No checkpoint available!")
            return False
        
        model.load_state_dict(self._best_checkpoint.model_state_dict)
        if optimizer and self._best_checkpoint.optimizer_state_dict:
            optimizer.load_state_dict(self._best_checkpoint.optimizer_state_dict)
        print(f"[Monitor] ✓ Restored to epoch {self._best_checkpoint.epoch} (val_loss={self._best_checkpoint.val_loss:.4f})")
        return True
    
    def print_summary(self):
        """Print summary of best epochs per head (for analysis)."""
        print("\n" + "="*70)
        print("TRAINING MONITOR SUMMARY")
        print("="*70)
        print(f"{'Head':<17} {'Best Epoch':>10} {'Best Val Loss':>14} {'Note':>15}")
        print("-"*70)
        for name, head in self.heads.items():
            note = "← RESTORED" if name == "GLOBAL" and head.best_epoch == self.best_epoch else ""
            print(f"{name:<17} {head.best_epoch:>10} {head.best_val_loss:>14.6f} {note:>15}")
        print("-"*70)
        print(f"{'GLOBAL (restore)':<17} {self.best_epoch:>10} {self.best_val_loss:>14.6f} {'← MODEL SAVED':>15}")
        print("="*70)
        print("\n⚠️  Note: Per-head best epochs are for ANALYSIS only.")
        print("    Model is saved/restored from GLOBAL best epoch.")
    
    def save_analytics_to_file(self, output_dir: str = "training_analytics"):
        """
        Save all training analytics to files for post-analysis.
        
        Creates:
        - analytics.json: Per-head best epochs, losses, and all history
        - epoch_snapshots/: PNG plots for each epoch
        """
        import json
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/epoch_snapshots", exist_ok=True)
        
        # Build analytics dict
        analytics = {
            'global': {
                'best_epoch': self.best_epoch,
                'best_val_loss': self.best_val_loss,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
            },
            'per_head': {}
        }
        
        for name, head in self.heads.items():
            analytics['per_head'][name] = {
                'best_epoch': head.best_epoch,
                'best_val_loss': head.best_val_loss,
                'val_losses': head.val_losses,
            }
        
        # Save JSON
        json_path = f"{output_dir}/analytics.json"
        with open(json_path, 'w') as f:
            json.dump(analytics, f, indent=2)
        
        print(f"[Monitor] ✓ Analytics saved to {json_path}")
        return json_path
    
    def save_epoch_snapshot(self, epoch: int, total_epochs: int, output_dir: str = "training_analytics"):
        """Save epoch snapshot plot to file."""
        import os
        os.makedirs(f"{output_dir}/epoch_snapshots", exist_ok=True)
        output_path = f"{output_dir}/epoch_snapshots/epoch_{epoch:03d}.png"
        return self.save_plot_to_file(epoch, total_epochs, output_path)
    
    def save_plot_to_file(self, epoch: int, total_epochs: int, output_path: str = None):
        """
        Save plot to PNG file (NON-BLOCKING).
        
        This is preferred over inline display for:
        - No matplotlib blocking
        - Works in all Colab cells
        - Faster than display()
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(3, 4, figsize=(16, 10))
            axes = axes.flatten()
            
            epochs_x = list(range(1, len(self.train_losses) + 1))
            
            # Plot 0: Global loss
            ax = axes[0]
            ax.plot(epochs_x, self.train_losses, 'b-', label='Train', linewidth=2)
            ax.plot(epochs_x, self.val_losses, 'r-', label='Val', linewidth=2)
            if self.best_epoch > 0:
                ax.axvline(x=self.best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best E{self.best_epoch}')
            ax.set_title(f'GLOBAL Loss\nBest: E{self.best_epoch} ({self.best_val_loss:.4f})', fontweight='bold')
            ax.legend(loc='upper right', fontsize=7)
            ax.grid(True, alpha=0.3)
            
            # Plots 1-9: Per-head losses
            for i, (name, head) in enumerate(self.heads.items()):
                ax = axes[i + 1]
                
                if len(head.val_losses) > 0:
                    ax.plot(epochs_x[:len(head.val_losses)], head.val_losses, 'r-', linewidth=1.5)
                    if head.best_epoch > 0:
                        ax.axvline(x=head.best_epoch, color='g', linestyle='--', alpha=0.7)
                        ax.scatter([head.best_epoch], [head.best_val_loss], color='g', s=40, zorder=5)
                
                ax.set_title(f'{name}\nBest: E{head.best_epoch} ({head.best_val_loss:.4f})', fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=7)
            
            # Clear unused axes
            for i in range(len(self.heads) + 1, len(axes)):
                axes[i].axis('off')
            
            fig.suptitle(
                f'CondorBrain Training Monitor - Epoch {epoch}/{total_epochs}\n'
                f'Global Best: Epoch {self.best_epoch} | Val Loss: {self.best_val_loss:.4f}',
                fontsize=12, fontweight='bold'
            )
            fig.tight_layout()
            
            # Save to file (non-blocking)
            if output_path is None:
                import os
                os.makedirs(self.plot_dir, exist_ok=True)
                output_path = f"{self.plot_dir}/monitor_epoch_{epoch:03d}.png"
            
            fig.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            return output_path
            
        except Exception as e:
            print(f"[Plot error] {e}")
            return None
    
    def display_inline(self, epoch: int, total_epochs: int):
        """
        Display plot inline in Colab/Jupyter (may block briefly).
        
        Use save_plot_to_file() for guaranteed non-blocking.
        """
        try:
            import matplotlib.pyplot as plt
            from IPython.display import display, clear_output, Image
            
            # Save to temp file first
            path = self.save_plot_to_file(epoch, total_epochs)
            if path:
                clear_output(wait=True)
                display(Image(filename=path))
                
        except Exception as e:
            print(f"[Display error] {e}")
    
    def save_checkpoint_to_disk(self, path: str, checkpoint: TrainingState = None):
        """Save a checkpoint to disk."""
        ckpt = checkpoint or self._best_checkpoint
        if ckpt is None:
            print("[Monitor] No checkpoint to save!")
            return
        
        torch.save({
            'epoch': ckpt.epoch,
            'model_state_dict': ckpt.model_state_dict,
            'optimizer_state_dict': ckpt.optimizer_state_dict,
            'scheduler_state_dict': ckpt.scheduler_state_dict,
            'train_loss': ckpt.train_loss,
            'val_loss': ckpt.val_loss,
            'head_val_losses': ckpt.head_val_losses,
        }, path)
        print(f"[Monitor] ✓ Saved checkpoint to {path}")


def compute_val_head_losses(
    model: torch.nn.Module,
    get_batch_fn,
    n_batches: int,
    device: torch.device,
    amp_dtype: torch.dtype
) -> Dict[str, float]:
    """
    Compute per-head validation losses including regime accuracy.
    
    PERFORMANCE: 
    - Accumulates on GPU, only calls .item() once at end
    - No CPU sync in loop
    
    Args:
        model: The model
        get_batch_fn: Function(batch_idx) -> (x, y, r)
        n_batches: Number of batches
        device: Device
        amp_dtype: AMP dtype
        
    Returns:
        Dict mapping head name to average loss (Python floats)
    """
    from torch.amp import autocast
    
    model.eval()
    
    # Accumulate on GPU
    head_loss_accum = {name: torch.tensor(0.0, device=device) for name in MAIN_HEADS}
    regime_correct = torch.tensor(0, device=device, dtype=torch.long)
    regime_total = torch.tensor(0, device=device, dtype=torch.long)
    
    with torch.no_grad():
        for bi in range(n_batches):
            batch_x, batch_y, batch_r = get_batch_fn(bi)
            
            # Move to device
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_r = batch_r.to(device, non_blocking=True)
            
            with autocast('cuda', dtype=amp_dtype):
                res = model(batch_x, return_regime=True, forecast_days=0)
                # Handle simplified V2.1 tuple return (outputs, regime, etc...)
                if isinstance(res, tuple):
                    outputs = res[0]
                    regime_logits = res[1] if len(res) > 1 else None
                else:
                    outputs = res
                    regime_logits = None
            
            # Compute per-head losses (accumulate on GPU, no .item())
            for i, name in enumerate(MAIN_HEADS):
                loss = torch.mean((outputs[:, i].float() - batch_y[:, i].float()) ** 2)
                head_loss_accum[name] += loss
            
            # Compute regime classification accuracy (on GPU)
            if regime_logits is not None:
                pred_regime = torch.argmax(regime_logits, dim=-1)
                regime_correct += (pred_regime == batch_r).sum()
                regime_total += batch_r.numel()
    
    # NOW move to CPU and call .item() (once per head)
    head_losses = {}
    for name in MAIN_HEADS:
        head_losses[name] = (head_loss_accum[name] / max(n_batches, 1)).item()
    
    # Regime accuracy (stored as 1 - accuracy so lower is better)
    regime_acc = regime_correct.float() / regime_total.float().clamp(min=1)
    head_losses['regime_accuracy'] = (1.0 - regime_acc).item()
    
    return head_losses


def sample_predictions(
    model: torch.nn.Module,
    get_batch_fn,
    device: torch.device,
    amp_dtype: torch.dtype,
    n_samples: int = 16
) -> Dict[str, np.ndarray]:
    """
    Sample predictions vs actuals for visualization.
    
    Returns dict with 'preds' and 'targets', each (n_samples, 8) numpy arrays.
    """
    from torch.amp import autocast
    
    model.eval()
    
    with torch.no_grad():
        batch_x, batch_y, batch_r = get_batch_fn(0)  # Get first batch
        
        # Move to device
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        batch_r = batch_r.to(device, non_blocking=True)
        
        with autocast('cuda', dtype=amp_dtype):
            # Request all outputs: regime, experts, and 45-day forecast
            outputs, regime_logits, horizon_forecast, experts = model(
                batch_x, 
                return_regime=True, 
                return_experts=True, 
                forecast_days=45
            )
        
        # Take first n_samples
        preds = outputs[:n_samples].float().cpu().numpy()
        targets = batch_y[:n_samples].float().cpu().numpy()
        
        # Expert specific predictions (Low, Normal, High) - may be None if TopKMoE is used
        if experts is not None:
            expert_preds = {
                k: v[:n_samples].float().cpu().numpy() for k, v in experts.items()
            }
        else:
            expert_preds = None
        
        # Price trajectory forecast
        forecast_data = None
        if horizon_forecast is not None:
            # daily_forecast: (B, num_days, 4) -> [close, high, low, vol]
            forecast_data = horizon_forecast['daily_forecast'][:n_samples].float().cpu().numpy()
            max_range = horizon_forecast['max_range'][:n_samples].float().cpu().numpy()
        
        # Regime predictions and probabilities
        if regime_logits is not None:
            regime_probs = torch.softmax(regime_logits[:n_samples].float(), dim=-1).cpu().numpy()
            pred_regime = np.argmax(regime_probs, axis=-1)
            true_regime = batch_r[:n_samples].cpu().numpy()
            
            # Compute mean regime distribution across batch
            regime_dist = regime_probs.mean(axis=0)  # (3,) - Low, Normal, High
        else:
            pred_regime = np.zeros(n_samples)
            true_regime = np.zeros(n_samples)
            regime_dist = np.array([0.33, 0.34, 0.33])
    
    return {
        'preds': preds,
        'targets': targets,
        'expert_preds': expert_preds,
        'forecast_data': forecast_data,
        'pred_regime': pred_regime,
        'true_regime': true_regime,
        'regime_probs_low': float(regime_dist[0]),
        'regime_probs_normal': float(regime_dist[1]),
        'regime_probs_high': float(regime_dist[2]),
    }


def visualize_predictions(
    samples: Dict[str, np.ndarray],
    epoch: int,
    total_epochs: int,
    head_losses: Dict[str, float],
    output_path: str = None,
    plot_dir: str = "monitor_plots"
) -> str:
    """
    Create visualization of predicted vs actual for all 8 output heads.
    
    Shows:
    - Scatter plot (pred vs actual) for each head
    - Mean absolute error
    - Correlation
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        preds = samples['preds']
        targets = samples['targets']
        n_samples = preds.shape[0]
        
        # Create 2x4 grid for 8 heads
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        # Head display info
        head_info = {
            'call_offset': ('Call Offset %', 'blue'),
            'put_offset': ('Put Offset %', 'red'),
            'wing_width': ('Wing Width $', 'green'),
            'dte': ('DTE (days)', 'orange'),
            'pop': ('Prob of Profit', 'purple'),
            'roi': ('Expected ROI', 'brown'),
            'max_loss': ('Max Loss %', 'pink'),
            'confidence': ('Confidence', 'cyan')
        }
        
        for i, name in enumerate(MAIN_HEADS):
            ax = axes[i]
            p = preds[:, i]
            t = targets[:, i]
            
            # Scatter: pred vs actual
            ax.scatter(t, p, alpha=0.6, s=40, c=head_info[name][1], edgecolors='black', linewidth=0.5)
            
            # Perfect prediction line
            all_vals = np.concatenate([p, t])
            vmin, vmax = all_vals.min(), all_vals.max()
            margin = (vmax - vmin) * 0.1 + 0.01
            ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin], 
                   'k--', alpha=0.5, linewidth=1, label='Perfect')
            
            # Stats
            mae = np.mean(np.abs(p - t))
            corr = np.corrcoef(p, t)[0, 1] if np.std(p) > 1e-6 and np.std(t) > 1e-6 else 0
            loss = head_losses.get(name, 0)
            
            ax.set_xlabel('Actual', fontsize=9)
            ax.set_ylabel('Predicted', fontsize=9)
            ax.set_title(f'{head_info[name][0]}\nMAE={mae:.3f} | r={corr:.2f} | MSE={loss:.4f}', 
                        fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(vmin - margin, vmax + margin)
            ax.set_ylim(vmin - margin, vmax + margin)
            ax.tick_params(labelsize=8)
        
        fig.suptitle(
            f'CondorBrain Predictions vs Actuals - Epoch {epoch}/{total_epochs}\n'
            f'Sample size: {n_samples} | Regime acc: {100*(1-head_losses.get("regime_accuracy", 0)):.1f}%',
            fontsize=12, fontweight='bold'
        )
        fig.tight_layout()
        
        # Save
        if output_path is None:
            import os
            os.makedirs(plot_dir, exist_ok=True)
            output_path = f"{plot_dir}/predictions_epoch_{epoch:03d}.png"
        
        fig.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return output_path
        
    except Exception as e:
        print(f"[Viz error] {e}")
        return None


def display_predictions_inline(
    samples: Dict[str, np.ndarray],
    epoch: int,
    total_epochs: int,
    head_losses: Dict[str, float],
    plot_dir: str = "monitor_plots"
):
    """Display predictions inline in Colab."""
    try:
        from IPython.display import display, Image
        
        path = visualize_predictions(samples, epoch, total_epochs, head_losses, plot_dir=plot_dir)
        if path:
            display(Image(filename=path))
    except Exception as e:
        print(f"[Display error] {e}")

