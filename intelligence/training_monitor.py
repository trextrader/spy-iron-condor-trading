"""
CondorBrain Training Monitor: Per-Head Metrics & In-Memory Checkpointing

Features:
- Per-output-head loss tracking (8 predictors)
- In-memory best state checkpointing (no disk I/O during training)
- Real-time multi-panel visualization
- Manual interrupt → reconstruct from best point per head
"""
import copy
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# Output head names (matching CondorLoss order + regime)
HEAD_NAMES = [
    'call_offset', 'put_offset', 'wing_width', 'dte',
    'pop', 'roi', 'max_loss', 'confidence',
    'regime_accuracy'  # Regime classification accuracy
]


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
    Advanced training monitor with per-head tracking and in-memory checkpointing.
    
    Allows interruption at any point and reconstruction from optimal checkpoint
    for each output head independently.
    """
    
    def __init__(self, checkpoint_capacity: int = 3):
        """
        Args:
            checkpoint_capacity: How many recent checkpoints to keep in memory
        """
        self.capacity = checkpoint_capacity
        
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
        self._best_per_head: Dict[str, TrainingState] = {}
        
        # Visualization state
        self._fig = None
        self._axes = None
        
    def compute_per_head_losses(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute MSE loss for each output head.
        
        Args:
            pred: (B, 8) predictions
            target: (B, 8) targets
            
        Returns:
            Dict mapping head name to loss value
        """
        with torch.no_grad():
            losses = {}
            for i, name in enumerate(HEAD_NAMES):
                head_loss = torch.mean((pred[:, i].float() - target[:, i].float()) ** 2)
                losses[name] = head_loss.item()
        return losses
    
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
        Update monitor with epoch results and optionally checkpoint.
        
        Returns:
            Dict with status info including which heads improved
        """
        # Update global metrics
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        # Update per-head metrics
        improved_heads = []
        for name, loss in head_val_losses.items():
            head = self.heads[name]
            head.val_losses.append(loss)
            
            if loss < head.best_val_loss:
                head.best_val_loss = loss
                head.best_epoch = epoch
                improved_heads.append(name)
        
        # Create checkpoint
        state = TrainingState(
            epoch=epoch,
            model_state_dict=copy.deepcopy(model.state_dict()),
            optimizer_state_dict=copy.deepcopy(optimizer.state_dict()),
            scheduler_state_dict=copy.deepcopy(scheduler.state_dict()) if scheduler else None,
            train_loss=train_loss,
            val_loss=val_loss,
            head_val_losses=head_val_losses.copy()
        )
        
        # Update checkpoints
        self._checkpoints.append(state)
        if len(self._checkpoints) > self.capacity:
            self._checkpoints.pop(0)
        
        # Update best global checkpoint
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self._best_checkpoint = state
        
        # Update best per-head checkpoints
        for name in improved_heads:
            self._best_per_head[name] = state
        
        return {
            'improved_global': val_loss < self.best_val_loss or epoch == self.best_epoch,
            'improved_heads': improved_heads,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'head_summary': {name: (h.best_epoch, h.best_val_loss) for name, h in self.heads.items()}
        }
    
    def get_best_checkpoint(self) -> Optional[TrainingState]:
        """Get the globally best checkpoint."""
        return self._best_checkpoint
    
    def get_best_checkpoint_for_head(self, head_name: str) -> Optional[TrainingState]:
        """Get the best checkpoint for a specific output head."""
        return self._best_per_head.get(head_name)
    
    def restore_best(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None):
        """Restore model to globally best checkpoint."""
        if self._best_checkpoint is None:
            print("[Monitor] No checkpoint available!")
            return False
        
        model.load_state_dict(self._best_checkpoint.model_state_dict)
        if optimizer and self._best_checkpoint.optimizer_state_dict:
            optimizer.load_state_dict(self._best_checkpoint.optimizer_state_dict)
        print(f"[Monitor] ✓ Restored to epoch {self._best_checkpoint.epoch} (val_loss={self._best_checkpoint.val_loss:.4f})")
        return True
    
    def print_summary(self):
        """Print summary of best epochs per head."""
        print("\n" + "="*70)
        print("TRAINING MONITOR SUMMARY - Best Epochs Per Output Head")
        print("="*70)
        print(f"{'Head':<15} {'Best Epoch':>12} {'Best Val Loss':>15}")
        print("-"*70)
        for name, head in self.heads.items():
            print(f"{name:<15} {head.best_epoch:>12} {head.best_val_loss:>15.6f}")
        print("-"*70)
        print(f"{'GLOBAL':<15} {self.best_epoch:>12} {self.best_val_loss:>15.6f}")
        print("="*70)
    
    def plot_live(self, epoch: int, total_epochs: int):
        """Create/update live multi-panel visualization."""
        try:
            import matplotlib.pyplot as plt
            from IPython.display import display, clear_output
            
            if self._fig is None:
                # Create 3x3 grid: 8 heads + 1 global
                self._fig, self._axes = plt.subplots(3, 3, figsize=(14, 10))
                self._axes = self._axes.flatten()
                plt.ion()
            
            clear_output(wait=True)
            
            epochs_x = list(range(1, len(self.train_losses) + 1))
            
            # Plot global loss
            ax = self._axes[0]
            ax.clear()
            ax.plot(epochs_x, self.train_losses, 'b-', label='Train', linewidth=2)
            if any(v > 0 for v in self.val_losses):
                ax.plot(epochs_x, self.val_losses, 'r-', label='Val', linewidth=2)
            ax.axvline(x=self.best_epoch, color='g', linestyle='--', alpha=0.7)
            ax.set_title(f'GLOBAL Loss\nBest: E{self.best_epoch} ({self.best_val_loss:.4f})', fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Plot per-head losses
            for i, (name, head) in enumerate(self.heads.items()):
                ax = self._axes[i + 1]
                ax.clear()
                
                if len(head.val_losses) > 0:
                    ax.plot(epochs_x[:len(head.val_losses)], head.val_losses, 'r-', linewidth=1.5)
                    ax.axvline(x=head.best_epoch, color='g', linestyle='--', alpha=0.7)
                    ax.scatter([head.best_epoch], [head.best_val_loss], color='g', s=50, zorder=5)
                
                ax.set_title(f'{name}\nBest: E{head.best_epoch} ({head.best_val_loss:.4f})', fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=7)
            
            self._fig.suptitle(
                f'CondorBrain Training Monitor - Epoch {epoch}/{total_epochs}\n'
                f'Global Best: Epoch {self.best_epoch} | Val Loss: {self.best_val_loss:.4f}',
                fontsize=12, fontweight='bold'
            )
            self._fig.tight_layout()
            display(self._fig)
            
        except Exception as e:
            print(f"[Plot error] {e}")
    
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
    
    Args:
        model: The model
        get_batch_fn: Function(batch_idx) -> (x, y, r)
        n_batches: Number of batches
        device: Device
        amp_dtype: AMP dtype
        
    Returns:
        Dict mapping head name to average loss (or 1-accuracy for regime)
    """
    from torch.amp import autocast
    
    model.eval()
    
    # 8 main outputs + regime
    main_head_names = HEAD_NAMES[:8]  # Exclude regime_accuracy
    head_losses = {name: 0.0 for name in main_head_names}
    regime_correct = 0
    regime_total = 0
    
    with torch.no_grad():
        for bi in range(n_batches):
            batch_x, batch_y, batch_r = get_batch_fn(bi)
            
            with autocast('cuda', dtype=amp_dtype):
                outputs, regime_logits, _ = model(batch_x, return_regime=True, forecast_days=0)
            
            # Compute per-head losses for 8 main outputs
            for i, name in enumerate(main_head_names):
                loss = torch.mean((outputs[:, i].float() - batch_y[:, i].float()) ** 2)
                head_losses[name] += loss.item()
            
            # Compute regime classification accuracy
            if regime_logits is not None:
                pred_regime = torch.argmax(regime_logits, dim=-1)
                regime_correct += (pred_regime == batch_r).sum().item()
                regime_total += batch_r.numel()
    
    # Average main head losses
    for name in main_head_names:
        head_losses[name] /= max(n_batches, 1)
    
    # Compute regime accuracy (stored as 1 - accuracy so lower is better)
    regime_acc = regime_correct / max(regime_total, 1)
    head_losses['regime_accuracy'] = 1.0 - regime_acc  # Lower = better
    
    return head_losses

