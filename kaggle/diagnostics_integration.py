"""
Kaggle Training Diagnostics Integration
Drop-in snippet for condor_brain_train.py

This file provides the minimal integration code to add visual diagnostics
to your training loop without modifying the core training logic significantly.

Usage:
    1. Copy this file to your Kaggle notebook
    2. Import and initialize at the start of training
    3. Call log_step() in your training loop
    4. Call save_and_plot() at the end of training
"""

import torch
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path


# =============================================================================
# Lightweight Diagnostics Tracker (Kaggle-optimized)
# =============================================================================

class KaggleDiagnostics:
    """
    Lightweight diagnostics tracker optimized for Kaggle notebooks.

    Features:
    - Minimal memory footprint
    - No external dependencies beyond numpy/pandas/matplotlib
    - Auto-saves periodically to prevent data loss
    - Generates plots inline
    """

    def __init__(
        self,
        num_fuzzy_gates: int = 10,
        log_interval: int = 10,
        save_dir: str = "/kaggle/working",
        run_name: str = None,
    ):
        self.num_fuzzy_gates = num_fuzzy_gates
        self.log_interval = log_interval
        self.save_dir = Path(save_dir)
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Storage (lists for memory efficiency)
        self.steps: List[int] = []
        self.epochs: List[int] = []

        # Loss components
        self.loss_vals: List[float] = []
        self.mse_vals: List[float] = []
        self.npdd_vals: List[float] = []
        self.sharpe_vals: List[float] = []
        self.dd_vals: List[float] = []
        self.turnover_vals: List[float] = []
        self.fuzzy_vals: List[float] = []
        self.pattern_ent_vals: List[float] = []
        self.group_inv_vals: List[float] = []
        self.rho_vals: List[float] = []
        self.energy_vals: List[float] = []
        self.growth_vals: List[float] = []

        # Fuzzy activations: list of [G] arrays
        self.fuzzy_activations: List[List[float]] = []

        # Optional extras
        self.lr_vals: List[float] = []
        self.grad_norm_vals: List[float] = []
        self.scaler_vals: List[float] = []

        self.start_time = datetime.now()
        self._last_save_step = 0

    def _to_float(self, x) -> float:
        """Convert tensor to float safely."""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().item()
        return float(x) if x is not None else 0.0

    def log_step(
        self,
        step: int,
        epoch: int,
        loss: Any,
        mse: Any,
        npdd: Any,
        sharpe: Any,
        dd: Any,
        turnover: Any,
        fuzzy: Any,
        pattern_ent: Any,
        group_inv: Any,
        rho: Any,
        energy: Any,
        growth: Any,
        fuzzy_activations: Optional[torch.Tensor] = None,
        lr: Optional[float] = None,
        grad_norm: Optional[float] = None,
        scaler_scale: Optional[float] = None,
    ):
        """Log a training step. Call this every N steps in your training loop."""

        if step % self.log_interval != 0:
            return

        self.steps.append(step)
        self.epochs.append(epoch)

        # Loss components
        self.loss_vals.append(self._to_float(loss))
        self.mse_vals.append(self._to_float(mse))
        self.npdd_vals.append(self._to_float(npdd))
        self.sharpe_vals.append(self._to_float(sharpe))
        self.dd_vals.append(self._to_float(dd))
        self.turnover_vals.append(self._to_float(turnover))
        self.fuzzy_vals.append(self._to_float(fuzzy))
        self.pattern_ent_vals.append(self._to_float(pattern_ent))
        self.group_inv_vals.append(self._to_float(group_inv))
        self.rho_vals.append(self._to_float(rho))
        self.energy_vals.append(self._to_float(energy))
        self.growth_vals.append(self._to_float(growth))

        # Optional
        self.lr_vals.append(self._to_float(lr) if lr else 0.0)
        self.grad_norm_vals.append(self._to_float(grad_norm) if grad_norm else 0.0)
        self.scaler_vals.append(self._to_float(scaler_scale) if scaler_scale else 0.0)

        # Fuzzy activations
        if fuzzy_activations is not None:
            with torch.no_grad():
                if fuzzy_activations.dim() == 2:
                    acts = fuzzy_activations.mean(dim=0)
                else:
                    acts = fuzzy_activations
                self.fuzzy_activations.append([self._to_float(a) for a in acts])

        # Auto-save every 1000 steps
        if step - self._last_save_step >= 1000:
            self.save_checkpoint()
            self._last_save_step = step

    def save_checkpoint(self):
        """Save current diagnostics to disk."""
        data = self.to_dict()
        path = self.save_dir / f"diagnostics_{self.run_name}_checkpoint.json"
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"[Diagnostics] Checkpoint saved: {path}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metadata": {
                "run_name": self.run_name,
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "num_steps": len(self.steps),
                "num_fuzzy_gates": self.num_fuzzy_gates,
            },
            "steps": self.steps,
            "epochs": self.epochs,
            "loss": self.loss_vals,
            "mse": self.mse_vals,
            "npdd": self.npdd_vals,
            "sharpe": self.sharpe_vals,
            "dd": self.dd_vals,
            "turnover": self.turnover_vals,
            "fuzzy": self.fuzzy_vals,
            "pattern_ent": self.pattern_ent_vals,
            "group_inv": self.group_inv_vals,
            "rho": self.rho_vals,
            "energy": self.energy_vals,
            "growth": self.growth_vals,
            "lr": self.lr_vals,
            "grad_norm": self.grad_norm_vals,
            "scaler_scale": self.scaler_vals,
            "fuzzy_activations": self.fuzzy_activations,
        }

    def save(self, filename: Optional[str] = None):
        """Save final diagnostics."""
        if filename is None:
            filename = f"diagnostics_{self.run_name}_final.json"
        path = self.save_dir / filename
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[Diagnostics] Saved to: {path}")
        return path

    # =========================================================================
    # Plotting
    # =========================================================================

    def plot_trajectories(
        self,
        smoothing: int = 200,
        save_path: Optional[str] = None,
    ):
        """Plot loss component trajectories with smoothing."""
        import pandas as pd
        import matplotlib.pyplot as plt

        components = {
            "loss": self.loss_vals,
            "mse": self.mse_vals,
            "npdd": self.npdd_vals,
            "sharpe": self.sharpe_vals,
            "dd": self.dd_vals,
            "turnover": self.turnover_vals,
            "fuzzy": self.fuzzy_vals,
            "pattern_ent": self.pattern_ent_vals,
            "energy": self.energy_vals,
            "growth": self.growth_vals,
        }

        df = pd.DataFrame({"step": self.steps, **components})
        smooth = df.rolling(smoothing, min_periods=1).mean()

        fig, axes = plt.subplots(len(components), 1, figsize=(14, 2.5 * len(components)), sharex=True)

        colors = plt.cm.tab10(np.linspace(0, 1, len(components)))

        for ax, (name, _), color in zip(axes, components.items(), colors):
            ax.plot(df["step"], smooth[name], label=name, linewidth=1.5, color=color)
            ax.set_ylabel(name, fontsize=10)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        axes[-1].set_xlabel("Training Step", fontsize=12)
        plt.suptitle(f"Loss Component Trajectories - {self.run_name}", fontsize=14, y=1.01)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[Diagnostics] Trajectory plot saved: {save_path}")

        plt.show()

    def plot_fuzzy_heatmap(self, save_path: Optional[str] = None):
        """Plot fuzzy gate activation heatmap."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        if not self.fuzzy_activations:
            print("[Diagnostics] No fuzzy activations logged")
            return

        mat = np.array(self.fuzzy_activations)  # [T, G]

        plt.figure(figsize=(16, 8))
        sns.heatmap(
            mat.T,
            cmap="magma",
            cbar=True,
            xticklabels=False,
            yticklabels=[f"F{i}" for i in range(mat.shape[1])],
        )
        plt.xlabel("Training Step", fontsize=12)
        plt.ylabel("Fuzzy Gates", fontsize=12)
        plt.title(f"Fuzzy Gate Activation Heatmap - {self.run_name}", fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[Diagnostics] Heatmap saved: {save_path}")

        plt.show()

    def plot_gate_distributions(self, epoch: int, save_path: Optional[str] = None):
        """Plot fuzzy gate distributions for a specific epoch."""
        import matplotlib.pyplot as plt

        if not self.fuzzy_activations:
            print("[Diagnostics] No fuzzy activations logged")
            return

        # Filter by epoch
        epoch_indices = [i for i, e in enumerate(self.epochs) if e == epoch]
        if not epoch_indices:
            print(f"[Diagnostics] No data for epoch {epoch}")
            return

        epoch_acts = [self.fuzzy_activations[i] for i in epoch_indices if i < len(self.fuzzy_activations)]
        if not epoch_acts:
            print(f"[Diagnostics] No fuzzy data for epoch {epoch}")
            return

        mat = np.array(epoch_acts)  # [T_epoch, G]
        n_gates = mat.shape[1]

        ncols = min(5, n_gates)
        nrows = (n_gates + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
        axes = axes.ravel() if hasattr(axes, 'ravel') else [axes]

        for i in range(n_gates):
            ax = axes[i]
            ax.hist(mat[:, i], bins=30, alpha=0.8, color='coral', edgecolor='black')
            ax.set_title(f"F{i}", fontsize=10)
            ax.axvline(mat[:, i].mean(), color='blue', linestyle='--', label=f"mean={mat[:, i].mean():.3f}")
            ax.legend(fontsize=6)

        for ax in axes[n_gates:]:
            ax.axis('off')

        plt.suptitle(f"Fuzzy Gate Distributions - Epoch {epoch}", fontsize=14, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()

    def plot_all(self, save_prefix: Optional[str] = None):
        """Generate all diagnostic plots."""
        prefix = save_prefix or f"{self.save_dir}/diagnostics_{self.run_name}"

        self.plot_trajectories(save_path=f"{prefix}_trajectories.png")
        self.plot_fuzzy_heatmap(save_path=f"{prefix}_fuzzy_heatmap.png")

        # Plot distributions for last epoch
        if self.epochs:
            last_epoch = max(self.epochs)
            self.plot_gate_distributions(last_epoch, save_path=f"{prefix}_gates_epoch{last_epoch}.png")

    def summary(self):
        """Print summary statistics."""
        print("\n" + "=" * 60)
        print(f"TRAINING DIAGNOSTICS SUMMARY - {self.run_name}")
        print("=" * 60)
        print(f"Total steps logged: {len(self.steps)}")
        print(f"Epochs: {min(self.epochs) if self.epochs else 0} - {max(self.epochs) if self.epochs else 0}")
        print(f"Duration: {datetime.now() - self.start_time}")
        print()
        print("Final Loss Components (last 100 steps avg):")
        print(f"  Loss:       {np.mean(self.loss_vals[-100:]):.4f}")
        print(f"  MSE:        {np.mean(self.mse_vals[-100:]):.4f}")
        print(f"  NPDD:       {np.mean(self.npdd_vals[-100:]):.4f}")
        print(f"  Sharpe:     {np.mean(self.sharpe_vals[-100:]):.4f}")
        print(f"  DD:         {np.mean(self.dd_vals[-100:]):.4f}")
        print(f"  Turnover:   {np.mean(self.turnover_vals[-100:]):.4f}")
        print(f"  Fuzzy:      {np.mean(self.fuzzy_vals[-100:]):.4f}")
        print(f"  PatternEnt: {np.mean(self.pattern_ent_vals[-100:]):.4f}")
        print(f"  Energy:     {np.mean(self.energy_vals[-100:]):.4f}")
        print(f"  Growth:     {np.mean(self.growth_vals[-100:]):.4f}")
        print("=" * 60 + "\n")


# =============================================================================
# INTEGRATION EXAMPLE
# =============================================================================

INTEGRATION_EXAMPLE = '''
# ============================================================================
# TRAINING LOOP INTEGRATION EXAMPLE
# ============================================================================

# 1. Initialize at start of training
diagnostics = KaggleDiagnostics(
    num_fuzzy_gates=10,  # Match your model's fuzzy gates
    log_interval=10,      # Log every 10 steps
    save_dir="/kaggle/working",
    run_name="condornet_v1",
)

# 2. In your training loop, after computing loss:
for epoch in range(num_epochs):
    for batch_idx, (x_b, y_b) in enumerate(train_loader):
        global_step = epoch * len(train_loader) + batch_idx

        # ... your forward pass ...
        outputs = model(x_b)

        # ... your loss computation ...
        loss, components = compute_loss(outputs, y_b)
        # where components = {"mse": mse, "npdd": npdd, "sharpe": sharpe, ...}

        # LOG THE STEP
        diagnostics.log_step(
            step=global_step,
            epoch=epoch,
            loss=loss,
            mse=components["mse"],
            npdd=components["npdd"],
            sharpe=components["sharpe"],
            dd=components["dd"],
            turnover=components["turnover"],
            fuzzy=components["fuzzy"],
            pattern_ent=components["pattern_ent"],
            group_inv=components.get("group_inv", 0),
            rho=components.get("rho", 0),
            energy=components.get("energy", 0),
            growth=components.get("growth", 0),
            fuzzy_activations=model.get_fuzzy_activations() if hasattr(model, 'get_fuzzy_activations') else None,
            lr=scheduler.get_last_lr()[0] if scheduler else None,
            scaler_scale=scaler.get_scale() if scaler else None,
        )

        # ... rest of training step ...

# 3. After training, generate visualizations
diagnostics.summary()
diagnostics.plot_all()
diagnostics.save()
'''

if __name__ == "__main__":
    print(INTEGRATION_EXAMPLE)
