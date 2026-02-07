"""
Training Diagnostics Module
Phase 6 - Visual Diagnostics for CondorNet Training

Provides:
1. Loss component trajectory logging
2. Fuzzy gate activation tracking
3. Real-time visualization utilities
4. Export formats for GUI integration
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import torch


@dataclass
class LossComponents:
    """Container for all loss components at a training step."""
    step: int
    epoch: int
    loss: float
    mse: float
    npdd: float
    sharpe: float
    dd: float
    turnover: float
    fuzzy: float
    pattern_ent: float
    group_inv: float
    rho: float
    energy: float
    growth: float
    lr: Optional[float] = None
    grad_norm: Optional[float] = None
    scaler_scale: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FuzzyActivations:
    """Container for fuzzy gate activations at a training step."""
    step: int
    epoch: int
    activations: List[float]  # [G] gate activations (mean over batch)

    def to_dict(self) -> Dict[str, Any]:
        d = {"step": self.step, "epoch": self.epoch}
        for i, v in enumerate(self.activations):
            d[f"fuzzy_{i}"] = v
        return d


class TrainingDiagnostics:
    """
    Comprehensive training diagnostics tracker.

    Usage:
        diagnostics = TrainingDiagnostics(num_fuzzy_gates=10)

        # In training loop:
        diagnostics.log_loss(step, epoch, loss, mse, npdd, ...)
        diagnostics.log_fuzzy(step, epoch, fuzzy_activations)

        # After training:
        diagnostics.plot_trajectories()
        diagnostics.plot_fuzzy_heatmap()
        diagnostics.save("training_diagnostics.json")
    """

    def __init__(
        self,
        num_fuzzy_gates: int = 10,
        smoothing_window: int = 200,
        log_interval: int = 10,  # Log every N steps
    ):
        self.num_fuzzy_gates = num_fuzzy_gates
        self.smoothing_window = smoothing_window
        self.log_interval = log_interval

        # Storage
        self.loss_logs: List[LossComponents] = []
        self.fuzzy_logs: List[FuzzyActivations] = []

        # Metadata
        self.start_time = datetime.now()
        self.metadata: Dict[str, Any] = {}

    def log_loss(
        self,
        step: int,
        epoch: int,
        loss: torch.Tensor,
        mse: torch.Tensor,
        npdd: torch.Tensor,
        sharpe: torch.Tensor,
        dd: torch.Tensor,
        turnover: torch.Tensor,
        fuzzy: torch.Tensor,
        pattern_ent: torch.Tensor,
        group_inv: torch.Tensor,
        rho: torch.Tensor,
        energy: torch.Tensor,
        growth: torch.Tensor,
        lr: Optional[float] = None,
        grad_norm: Optional[float] = None,
        scaler_scale: Optional[float] = None,
    ):
        """Log all loss components at a training step."""
        if step % self.log_interval != 0:
            return

        entry = LossComponents(
            step=step,
            epoch=epoch,
            loss=self._to_float(loss),
            mse=self._to_float(mse),
            npdd=self._to_float(npdd),
            sharpe=self._to_float(sharpe),
            dd=self._to_float(dd),
            turnover=self._to_float(turnover),
            fuzzy=self._to_float(fuzzy),
            pattern_ent=self._to_float(pattern_ent),
            group_inv=self._to_float(group_inv),
            rho=self._to_float(rho),
            energy=self._to_float(energy),
            growth=self._to_float(growth),
            lr=lr,
            grad_norm=grad_norm,
            scaler_scale=scaler_scale,
        )
        self.loss_logs.append(entry)

    def log_fuzzy(
        self,
        step: int,
        epoch: int,
        fuzzy_activations: torch.Tensor,  # [B, G] or [G]
    ):
        """Log fuzzy gate activations at a training step."""
        if step % self.log_interval != 0:
            return

        with torch.no_grad():
            if fuzzy_activations.dim() == 2:
                # [B, G] -> mean over batch
                acts = fuzzy_activations.mean(dim=0)
            else:
                acts = fuzzy_activations

            activations = [self._to_float(a) for a in acts]

        entry = FuzzyActivations(
            step=step,
            epoch=epoch,
            activations=activations,
        )
        self.fuzzy_logs.append(entry)

    def _to_float(self, x: Any) -> float:
        """Convert tensor or number to float."""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().item()
        return float(x)

    # =========================================================================
    # Data Access
    # =========================================================================

    def get_loss_df(self) -> pd.DataFrame:
        """Get loss logs as a DataFrame."""
        return pd.DataFrame([log.to_dict() for log in self.loss_logs])

    def get_fuzzy_df(self) -> pd.DataFrame:
        """Get fuzzy logs as a DataFrame."""
        return pd.DataFrame([log.to_dict() for log in self.fuzzy_logs])

    def get_smoothed_loss_df(self) -> pd.DataFrame:
        """Get smoothed loss trajectories."""
        df = self.get_loss_df()
        if len(df) == 0:
            return df
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return df[numeric_cols].rolling(self.smoothing_window, min_periods=1).mean()

    def get_epoch_summary(self) -> pd.DataFrame:
        """Get per-epoch aggregated statistics."""
        df = self.get_loss_df()
        if len(df) == 0:
            return df
        numeric_cols = [c for c in df.columns if c not in ["step", "epoch"]]
        return df.groupby("epoch")[numeric_cols].mean().reset_index()

    # =========================================================================
    # Visualization
    # =========================================================================

    def plot_trajectories(
        self,
        save_path: Optional[str] = None,
        components: Optional[List[str]] = None,
    ):
        """
        Plot loss component trajectories.

        Args:
            save_path: Optional path to save the figure
            components: List of components to plot (default: all)
        """
        import matplotlib.pyplot as plt

        df = self.get_loss_df()
        if len(df) == 0:
            print("No loss data to plot")
            return

        smooth = self.get_smoothed_loss_df()

        if components is None:
            components = [
                "loss", "mse", "npdd", "sharpe", "dd", "turnover",
                "fuzzy", "pattern_ent", "group_inv", "rho", "energy", "growth"
            ]

        # Filter to available components
        components = [c for c in components if c in df.columns]

        fig, axes = plt.subplots(
            len(components), 1,
            figsize=(12, 2 * len(components)),
            sharex=True
        )

        if len(components) == 1:
            axes = [axes]

        for ax, comp in zip(axes, components):
            ax.plot(df["step"], smooth[comp], label=comp, linewidth=1.5)
            ax.set_ylabel(comp, fontsize=10)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Training Step", fontsize=12)
        plt.suptitle("Loss Component Trajectories", fontsize=14, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved trajectory plot to {save_path}")

        plt.show()

    def plot_fuzzy_heatmap(
        self,
        save_path: Optional[str] = None,
        cmap: str = "magma",
    ):
        """
        Plot fuzzy gate activation heatmap over training.

        Args:
            save_path: Optional path to save the figure
            cmap: Colormap to use
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        fdf = self.get_fuzzy_df()
        if len(fdf) == 0:
            print("No fuzzy data to plot")
            return

        fdf = fdf.sort_values("step")
        fuzzy_cols = [c for c in fdf.columns if c.startswith("fuzzy_")]

        if len(fuzzy_cols) == 0:
            print("No fuzzy columns found")
            return

        mat = fdf[fuzzy_cols].to_numpy()  # [T, G]

        plt.figure(figsize=(14, 6))
        sns.heatmap(
            mat.T,
            cmap=cmap,
            cbar=True,
            xticklabels=False,
            yticklabels=[c.replace("fuzzy_", "F") for c in fuzzy_cols],
        )
        plt.xlabel("Training Step", fontsize=12)
        plt.ylabel("Fuzzy Gates", fontsize=12)
        plt.title("Fuzzy Gate Activation Heatmap", fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved fuzzy heatmap to {save_path}")

        plt.show()

    def plot_gate_distributions(
        self,
        epoch: int,
        save_path: Optional[str] = None,
    ):
        """
        Plot distribution of fuzzy gate activations for a specific epoch.

        Args:
            epoch: Epoch number to analyze
            save_path: Optional path to save the figure
        """
        import matplotlib.pyplot as plt

        fdf = self.get_fuzzy_df()
        if len(fdf) == 0:
            print("No fuzzy data to plot")
            return

        sub = fdf[fdf["epoch"] == epoch]
        if len(sub) == 0:
            print(f"No data for epoch {epoch}")
            return

        fuzzy_cols = [c for c in sub.columns if c.startswith("fuzzy_")]
        n_gates = len(fuzzy_cols)

        if n_gates == 0:
            print("No fuzzy columns found")
            return

        # Determine grid layout
        ncols = min(5, n_gates)
        nrows = (n_gates + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
        axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

        for i, (ax, col) in enumerate(zip(axes, fuzzy_cols)):
            ax.hist(sub[col], bins=30, alpha=0.8, color="coral")
            ax.set_title(col.replace("fuzzy_", "F"), fontsize=10)
            ax.set_yticks([])

        # Hide unused axes
        for ax in axes[n_gates:]:
            ax.axis("off")

        plt.suptitle(f"Fuzzy Gate Distributions - Epoch {epoch}", fontsize=14, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved gate distributions to {save_path}")

        plt.show()

    def plot_epoch_comparison(
        self,
        metrics: List[str] = ["loss", "mse", "npdd", "sharpe"],
        save_path: Optional[str] = None,
    ):
        """Plot per-epoch metric comparison."""
        import matplotlib.pyplot as plt

        epoch_df = self.get_epoch_summary()
        if len(epoch_df) == 0:
            print("No epoch data to plot")
            return

        # Filter to available metrics
        metrics = [m for m in metrics if m in epoch_df.columns]

        fig, ax = plt.subplots(figsize=(10, 6))

        for metric in metrics:
            ax.plot(epoch_df["epoch"], epoch_df[metric], marker="o", label=metric)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.title("Per-Epoch Metric Trajectories", fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()

    # =========================================================================
    # Serialization
    # =========================================================================

    def save(self, path: str):
        """Save diagnostics to JSON file."""
        data = {
            "metadata": {
                **self.metadata,
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "num_fuzzy_gates": self.num_fuzzy_gates,
                "smoothing_window": self.smoothing_window,
                "log_interval": self.log_interval,
                "total_steps": len(self.loss_logs),
            },
            "loss_logs": [log.to_dict() for log in self.loss_logs],
            "fuzzy_logs": [log.to_dict() for log in self.fuzzy_logs],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved diagnostics to {path}")

    @classmethod
    def load(cls, path: str) -> "TrainingDiagnostics":
        """Load diagnostics from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        meta = data.get("metadata", {})
        diagnostics = cls(
            num_fuzzy_gates=meta.get("num_fuzzy_gates", 10),
            smoothing_window=meta.get("smoothing_window", 200),
            log_interval=meta.get("log_interval", 10),
        )
        diagnostics.metadata = meta

        # Restore loss logs
        for entry in data.get("loss_logs", []):
            diagnostics.loss_logs.append(LossComponents(**entry))

        # Restore fuzzy logs
        for entry in data.get("fuzzy_logs", []):
            step = entry.pop("step")
            epoch = entry.pop("epoch")
            activations = [entry[f"fuzzy_{i}"] for i in range(len(entry))]
            diagnostics.fuzzy_logs.append(FuzzyActivations(step, epoch, activations))

        return diagnostics

    def to_gui_format(self) -> Dict[str, Any]:
        """
        Export data in format suitable for GUI API.

        Returns dict with:
            - loss_trajectory: smoothed loss components
            - fuzzy_heatmap: matrix data for heatmap
            - epoch_summary: per-epoch aggregates
        """
        loss_df = self.get_smoothed_loss_df()
        fuzzy_df = self.get_fuzzy_df()
        epoch_df = self.get_epoch_summary()

        fuzzy_cols = [c for c in fuzzy_df.columns if c.startswith("fuzzy_")]

        return {
            "loss_trajectory": {
                "steps": loss_df.index.tolist() if len(loss_df) > 0 else [],
                "components": {
                    col: loss_df[col].tolist()
                    for col in loss_df.columns
                    if col not in ["step", "epoch"]
                } if len(loss_df) > 0 else {},
            },
            "fuzzy_heatmap": {
                "steps": fuzzy_df["step"].tolist() if len(fuzzy_df) > 0 else [],
                "gates": fuzzy_cols,
                "values": fuzzy_df[fuzzy_cols].values.tolist() if len(fuzzy_df) > 0 else [],
            },
            "epoch_summary": epoch_df.to_dict("records") if len(epoch_df) > 0 else [],
            "metadata": self.metadata,
        }


# =============================================================================
# Integration Helper
# =============================================================================

def create_training_diagnostics_hook(
    diagnostics: TrainingDiagnostics,
) -> callable:
    """
    Create a hook function for easy integration with training loops.

    Usage:
        diagnostics = TrainingDiagnostics()
        log_step = create_training_diagnostics_hook(diagnostics)

        # In training loop:
        log_step(
            step=global_step,
            epoch=epoch,
            loss_dict={"loss": loss, "mse": mse, ...},
            fuzzy_activations=fuzzy_acts,
            lr=scheduler.get_last_lr()[0],
        )
    """
    def hook(
        step: int,
        epoch: int,
        loss_dict: Dict[str, torch.Tensor],
        fuzzy_activations: Optional[torch.Tensor] = None,
        lr: Optional[float] = None,
        grad_norm: Optional[float] = None,
        scaler_scale: Optional[float] = None,
    ):
        # Log loss components
        diagnostics.log_loss(
            step=step,
            epoch=epoch,
            loss=loss_dict.get("loss", 0),
            mse=loss_dict.get("mse", 0),
            npdd=loss_dict.get("npdd", 0),
            sharpe=loss_dict.get("sharpe", 0),
            dd=loss_dict.get("dd", 0),
            turnover=loss_dict.get("turnover", 0),
            fuzzy=loss_dict.get("fuzzy", 0),
            pattern_ent=loss_dict.get("pattern_ent", 0),
            group_inv=loss_dict.get("group_inv", 0),
            rho=loss_dict.get("rho", 0),
            energy=loss_dict.get("energy", 0),
            growth=loss_dict.get("growth", 0),
            lr=lr,
            grad_norm=grad_norm,
            scaler_scale=scaler_scale,
        )

        # Log fuzzy activations if provided
        if fuzzy_activations is not None:
            diagnostics.log_fuzzy(step, epoch, fuzzy_activations)

    return hook


# =============================================================================
# Quick CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Training Diagnostics Viewer")
    parser.add_argument("--load", type=str, help="Path to diagnostics JSON file")
    parser.add_argument("--plot-loss", action="store_true", help="Plot loss trajectories")
    parser.add_argument("--plot-fuzzy", action="store_true", help="Plot fuzzy heatmap")
    parser.add_argument("--epoch", type=int, help="Epoch for gate distributions")
    args = parser.parse_args()

    if args.load:
        diag = TrainingDiagnostics.load(args.load)
        print(f"Loaded {len(diag.loss_logs)} loss entries, {len(diag.fuzzy_logs)} fuzzy entries")

        if args.plot_loss:
            diag.plot_trajectories()

        if args.plot_fuzzy:
            diag.plot_fuzzy_heatmap()

        if args.epoch is not None:
            diag.plot_gate_distributions(args.epoch)
    else:
        print("Usage: python diagnostics.py --load <path> [--plot-loss] [--plot-fuzzy] [--epoch N]")
