"""
CondorBrain Training Script (Ultra-Fast GPU Dataset)

Optimizations:
- GPU-resident data with unfold() views (zero-copy sequence slicing)
- No H2D transfers in training loop
- BF16 end-to-end for Mamba fast path
"""
import sys
import os

# CRITICAL: Set CUDA memory allocator BEFORE importing torch
# Note: Only use expandable_segments - other options may conflict with PyTorch versions
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import time
import argparse
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from intelligence.condor_brain import CondorBrain, CondorLoss, HAS_MAMBA
from intelligence.condor_loss import CompositeCondorLoss
from intelligence.canonical_feature_registry import (
    FEATURE_COLS_V22,
    VERSION_V22,
    select_feature_frame,
)
from intelligence.training_monitor import (
    TrainingMonitor, compute_val_head_losses, MAIN_HEADS,
    sample_predictions, display_predictions_inline
)


# ============================================================================
# KERNEL PROBES (Your throughput depends on this)
# ============================================================================

def probe_fast_kernels() -> dict:
    """
    Returns a dict describing whether fused CUDA kernels are available.
    If these are missing, you will be stuck in the ~1-3 it/s range for big models.
    """
    info = {
        "mamba_selective_scan_cuda": False,
        "causal_conv1d": False,
    }
    try:
        # selective_scan_cuda is the core fast path for mamba_ssm
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # noqa: F401
        info["mamba_selective_scan_cuda"] = True
    except Exception:
        info["mamba_selective_scan_cuda"] = False
    try:
        import causal_conv1d  # noqa: F401
        info["causal_conv1d"] = True
    except Exception:
        info["causal_conv1d"] = False
    return info

# ============================================================================
# ROBUST NORMALIZATION HELPERS (Fixes NaN training issue)
# ============================================================================

EPS = 1e-6

def safe_nan_to_num(X: np.ndarray) -> np.ndarray:
    """Never replace inf with huge constants. Replace NaN/Inf with 0."""
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def robust_zscore_fit(X: np.ndarray):
    """Robust scaler using median + MAD (much safer than mean/std on heavy tails)."""
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    scale = 1.4826 * mad  # Convert MAD to std-like scale
    scale = np.where(scale < EPS, 1.0, scale)
    return med.astype(np.float32), scale.astype(np.float32)

def robust_zscore_transform(X: np.ndarray, med: np.ndarray, scale: np.ndarray, clip_val: float = 10.0) -> np.ndarray:
    """Apply robust z-score and clip to prevent extreme values."""
    X = (X - med) / (scale + EPS)
    X = np.clip(X, -clip_val, clip_val)
    return X.astype(np.float32)

def clamp_targets(y: np.ndarray) -> np.ndarray:
    """Clamp targets to reasonable ranges to prevent BF16 overflow."""
    y = safe_nan_to_num(y)
    # Target ordering: call_offset, put_offset, wing_width, dte, pop, roi, max_loss, confidence
    y[:, 0] = np.clip(y[:, 0], -100.0, 100.0)  # call offset
    y[:, 1] = np.clip(y[:, 1], -100.0, 100.0)  # put offset
    y[:, 2] = np.clip(y[:, 2], 0.5, 50.0)       # wing width
    y[:, 3] = np.clip(y[:, 3], 0.0, 120.0)      # dte
    y[:, 4] = np.clip(y[:, 4], 0.0, 1.0)        # pop (probability)
    y[:, 5] = np.clip(y[:, 5], -5.0, 5.0)       # expected roi
    y[:, 6] = np.clip(y[:, 6], 0.0, 5.0)        # max loss
    y[:, 7] = np.clip(y[:, 7], 0.0, 1.0)        # confidence
    return y.astype(np.float32)

def debug_tensor_stats(name: str, X: np.ndarray):
    """Print tensor statistics for debugging NaN issues."""
    finite = np.isfinite(X)
    bad = np.size(X) - finite.sum()
    mx = np.max(np.abs(X[finite])) if finite.any() else np.nan
    print(f"[DEBUG] {name}: shape={X.shape} bad={bad} max|x|={mx:.4f}")

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    'd_model': 1024,
    'n_layers': 32,
    'lookback': 240,
    'batch_size': 64,
    'epochs': 100,
    'lr': 1e-4,
    'model_path': 'models/condor_brain.pth'
}

FEATURE_COLS = FEATURE_COLS_V22

# ============================================================================
# DATA PREPARATION (Fast In-Memory)
# ============================================================================

def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare features and targets for CondorBrain training."""
    print("[CondorBrain] Preparing features...")
    
    # Handle call_put encoding
    if 'call_put' in df.columns and 'cp_num' not in df.columns:
        df['cp_num'] = df['call_put'].map({'C': 1.0, 'P': -1.0}).fillna(0)
    
    # Generate synthetic targets (default offsets)
    df['target_call_offset'] = 2.0
    df['target_put_offset'] = 2.0
    df['target_wing_width'] = 5.0
    df['target_dte'] = 14.0
    df['was_profitable'] = 0.5
    df['realized_roi'] = 0.0
    df['realized_max_loss'] = 0.2
    df['confidence_target'] = 0.5
    
    # Regime labeling (with NaN safety)
    if 'regime_label' not in df.columns:
        if 'ivr' in df.columns:
            df['regime_label'] = pd.cut(
                df['ivr'].fillna(50),  # Default to normal regime
                bins=[-0.1, 30, 70, 101], 
                labels=[0, 1, 2]
            ).fillna(1).astype(int)  # Default to normal (1)
        else:
            df['regime_label'] = 1
    
    # Fill ALL NaNs
    df = df.ffill().bfill().fillna(0)
    
    # Build arrays
    target_cols = [
        'target_call_offset', 'target_put_offset', 'target_wing_width', 'target_dte',
        'was_profitable', 'realized_roi', 'realized_max_loss', 'confidence_target'
    ]
    
    X = select_feature_frame(df, FEATURE_COLS, strict=True).values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)
    regime = df['regime_label'].values.astype(np.int64)
    
    # ROBUST SANITIZATION (never use 1e6 - it explodes in BF16!)
    print("[CondorBrain] Sanitizing data (safe)...")
    X = safe_nan_to_num(X)
    y = clamp_targets(y)
    
    # Scale volume (index 4) to avoid huge gradients
    if X.shape[1] > 4:
        X[:, 4] = np.log1p(np.clip(X[:, 4], 0.0, 1e9)).astype(np.float32)
    
    # ROBUST NORMALIZATION (median/MAD + clip to ±10)
    print("[CondorBrain] Robust-normalizing features (median/MAD)...")
    med, scale = robust_zscore_fit(X)
    X = robust_zscore_transform(X, med, scale, clip_val=10.0)
    
    # Debug output
    debug_tensor_stats("X_scaled", X)
    debug_tensor_stats("y_clamped", y)
    
    # Sanity check: confirm dtypes and ranges
    print(f"[SANITY] X dtype: {X.dtype}, min/max: {X.min():.4f}/{X.max():.4f}")
    print(f"[SANITY] y dtype: {y.dtype}, min/max: {y.min():.4f}/{y.max():.4f}")
    
    # Verify clean
    if np.isnan(X).any() or np.isinf(X).any():
        raise RuntimeError("[CRITICAL ERROR] X still contains NaN/Inf after sanitization!")
    if np.isnan(y).any() or np.isinf(y).any():
        raise RuntimeError("[CRITICAL ERROR] y still contains NaN/Inf after sanitization!")
    
    print(f"[CondorBrain] Features: {X.shape}, Targets: {y.shape}")
    return X, y, regime



class BatchedSequenceDataset(IterableDataset):
    """High-performance batched sequence dataset.
    
    Builds entire batches at once using advanced indexing.
    Eliminates per-sample DataLoader collation overhead entirely.
    Typically 5-20x faster than SequenceDataset + DataLoader.
    """
    def __init__(self, X2d, y2d, r1d, lookback, batch_size, drop_last=True, pin_memory=True):
        super().__init__()
        self.L = int(lookback)
        self.B = int(batch_size)
        self.drop_last = drop_last
        
        # Store as CPU tensors ONCE - X as BF16 to avoid per-batch conversion
        self.X = torch.from_numpy(X2d).to(torch.bfloat16)   # (N, F) bf16 - no per-batch conversion!
        self.y = torch.from_numpy(y2d).to(torch.float32)    # (N, T) fp32
        self.r = torch.from_numpy(r1d).long()               # (N,)  int64
        
        if pin_memory:
            self.X = self.X.pin_memory()
            self.y = self.y.pin_memory()
            self.r = self.r.pin_memory()
        
        self.n_seq = self.X.shape[0] - self.L
        self.n_batches = self.n_seq // self.B if drop_last else math.ceil(self.n_seq / self.B)
        
        # Pre-compute lookback indices
        self._arL = torch.arange(self.L, dtype=torch.int64)
    
    def __len__(self):
        return self.n_batches
    
    def __iter__(self):
        for b in range(self.n_batches):
            start = b * self.B
            end = start + self.B
            if end > self.n_seq:
                if self.drop_last:
                    break
                end = self.n_seq
            
            # Build batch using advanced indexing (super fast)
            idx0 = torch.arange(start, end, dtype=torch.int64)
            idx = idx0[:, None] + self._arL[None, :]    # (B, L)
            xb = self.X[idx]                             # (B, L, F)
            yb = self.y[idx0 + self.L]                   # (B, T)
            rb = self.r[idx0 + self.L]                   # (B,)
            yield xb, yb, rb


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train CondorBrain")
    parser.add_argument("--local-data", type=str, required=True, help="Path to institutional CSV")
    parser.add_argument("--output", type=str, default="auto", help="Output model path ('auto' = generate from params)")
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lookback", type=int, default=240)
    parser.add_argument("--max-rows", type=int, default=0, help="Limit rows for debugging")
    parser.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation steps (effective_batch = batch_size * accum_steps)")
    parser.add_argument("--grad-checkpoint", action="store_true", help="Enable gradient checkpointing to save memory")
    parser.add_argument("--require-fused-kernels", action="store_true", help="Hard fail if selective_scan_cuda/causal_conv1d are missing")
    parser.add_argument("--compile", action="store_true", help="Try torch.compile (guarded). May improve throughput if supported by your mamba build.")
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead", choices=["reduce-overhead", "max-autotune", "default"], help="torch.compile mode")
    parser.add_argument("--gpu-dataset", action="store_true", default=True,
                        help="Store X/y/regime on GPU and use unfold() views (FASTEST). Default: enabled.")
    parser.add_argument("--no-gpu-dataset", dest="gpu_dataset", action="store_false",
                        help="Disable gpu-dataset and use CPU loaders (debug only).")
    # NEW: High-impact A100 performance flags
    parser.add_argument("--materialize-seqs", action="store_true",
                        help="Materialize unfold() views to contiguous GPU tensors (uses ~50GB VRAM, much faster).")
    parser.add_argument("--no-val", action="store_true",
                        help="Skip validation entirely (for throughput benchmarks).")
    parser.add_argument("--val-every", type=int, default=1,
                        help="Validate every N epochs instead of every epoch (default: 1).")
    # NEW: Early stopping and live visualization
    parser.add_argument("--early-stop", action="store_true",
                        help="Enable early stopping when val loss stops improving.")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience: epochs to wait after last improvement (default: 5).")
    parser.add_argument("--live-plot", action="store_true",
                        help="Display live training curve (train vs val loss) in Colab/notebook.")
    parser.add_argument("--log-every", type=int, default=100,
                        help="Log batch-level metrics every N batches (default: 100).")
    parser.add_argument("--monitor", action="store_true",
                        help="Enable advanced multi-head training monitor with per-predictor tracking.")
    parser.add_argument("--monitor-every", type=int, default=1,
                        help="Update monitor plots every N epochs (default: 1, use higher for faster training).")
    parser.add_argument("--viz-every", type=int, default=20,
                        help="Intra-epoch visualization: update plots every N batches (default: 20 ~1min, 0=disabled).")
    parser.add_argument("--tensorboard", action="store_true",
                        help="Enable TensorBoard logging for all metrics and predictions.")
    parser.add_argument("--tb-logdir", type=str, default="runs/condor_brain",
                        help="TensorBoard log directory (default: runs/condor_brain).")
    parser.add_argument("--tb-port", type=int, default=3500,
                        help="TensorBoard port (default: 3500).")
    
    # === NEW: Model Enhancement Flags ===
    parser.add_argument("--composite-loss", action="store_true",
                        help="Use CompositeCondorLoss (Sharpe/drawdown/turnover penalties).")
    parser.add_argument("--loss-lambdas", type=str, default="1.0,0.5,0.1,0.1",
                        help="Composite loss weights: pred,sharpe,drawdown,turnover (default: 1.0,0.5,0.1,0.1).")
    parser.add_argument("--vol-gated-attn", action="store_true", default=True,
                        help="Enable VolGatedAttn after layers 8,16,24 (default: enabled).")
    parser.add_argument("--no-vol-gated-attn", dest="vol_gated_attn", action="store_false",
                        help="Disable VolGatedAttn.")
    parser.add_argument("--topk-moe", action="store_true",
                        help="Use TopKMoE instead of traditional 3-expert MoE.")
    parser.add_argument("--moe-experts", type=int, default=3,
                        help="Number of MoE experts (default: 3).")
    parser.add_argument("--moe-k", type=int, default=1,
                        help="Number of experts to activate per sample (default: 1).")
    
    args = parser.parse_args()
    
    # Parse loss lambdas
    args.loss_lambdas_tuple = tuple(float(x) for x in args.loss_lambdas.split(','))
    
    # Auto-generate output filename from params
    if args.output == "auto":
        lr_str = f"{args.lr:.0e}".replace("-", "")
        args.output = f"models/condor_brain_e{args.epochs}_d{args.d_model}_L{args.layers}_lr{lr_str}.pth"
    
    return args


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_condor_brain(args):
    """Training with proven in-memory pattern."""
    
    if not HAS_MAMBA:
        print("[Error] mamba-ssm not available.")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[CondorBrain] Device: {device}")
    
    # Probe kernel availability early (your throughput depends on this)
    kinfo = probe_fast_kernels()
    print(f"[CondorBrain] Kernel probe: "
          f"selective_scan_cuda={'YES' if kinfo['mamba_selective_scan_cuda'] else 'NO'}, "
          f"causal_conv1d={'YES' if kinfo['causal_conv1d'] else 'NO'}")
    
    if args.require_fused_kernels:
        missing = []
        if not kinfo["mamba_selective_scan_cuda"]:
            missing.append("mamba_ssm selective_scan_cuda")
        if not kinfo["causal_conv1d"]:
            missing.append("causal_conv1d")
        if missing:
            raise RuntimeError(
                "[CRITICAL] Missing fused CUDA kernels: "
                + ", ".join(missing)
                + ". Install/compile these or throughput will be severely limited."
            )
    
    # Detect GPU capabilities
    use_bf16 = False
    is_h100 = False
    
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        compute_cap = torch.cuda.get_device_capability(0)
        
        print(f"[CondorBrain] GPU: {gpu_name}")
        print(f"[CondorBrain] GPU Memory: {gpu_mem:.1f} GB")
        print(f"[CondorBrain] Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
        
        # H100 detection (compute capability 9.0+)
        is_h100 = compute_cap[0] >= 9 or 'H100' in gpu_name
        # BF16 supported on Ampere+ (8.0+)
        use_bf16 = compute_cap[0] >= 8
        
        # === MAXIMUM CUDA OPTIMIZATIONS ===
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # Auto-tune kernels
        torch.backends.cudnn.deterministic = False  # Speed over reproducibility
        torch.set_float32_matmul_precision('high')  # Use Tensor Cores for FP32
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True  # Fast BF16 reductions
        
        if is_h100:
            print("[CondorBrain] 🚀 H100 DETECTED - Maximum optimizations enabled!")
            # H100 has 80GB HBM3, can handle larger batches
            torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory
        
        opt_str = f"TF32, cuDNN benchmark, {'BF16' if use_bf16 else 'FP16'}"
        print(f"[CondorBrain] CUDA optimizations enabled: {opt_str}")
    # Load data
    print(f"\n[CondorBrain] Loading data from {args.local_data}...")
    if args.max_rows > 0:
        df = pd.read_csv(args.local_data, nrows=args.max_rows)
    else:
        df = pd.read_csv(args.local_data)
    print(f"[CondorBrain] Loaded {len(df):,} rows")
    
    # Prepare features
    X, y, regime = prepare_features(df)
    
    # Free dataframe memory
    del df
    import gc
    gc.collect()
    
    # Split data at 2D row level (NO sequence creation here)
    split_row = int(len(X) * 0.8)
    X_train, X_val = X[:split_row], X[split_row:]
    y_train, y_val = y[:split_row], y[split_row:]
    r_train, r_val = regime[:split_row], regime[split_row:]
    
    # Number of sequences (start index positions)
    L = int(args.lookback)
    B = int(args.batch_size)
    n_train_seq = len(X_train) - L
    n_val_seq = len(X_val) - L
    print(f"[CondorBrain] Creating sequences (lookback={L})...")
    print(f"[CondorBrain] Train: {n_train_seq:,} | Val: {n_val_seq:,}")
    
    # ==========================================================================
    # FAST PATH: GPU DATASET + UNFOLD (ZERO-COPY VIEWS)
    # ==========================================================================
    use_gpu_dataset = args.gpu_dataset and device.type == "cuda"
    
    if use_gpu_dataset:
        print("[CondorBrain] ✅ gpu-dataset ENABLED: data on GPU, using unfold() views (FASTEST)")
        
        # Move data to GPU ONCE (no H2D in training loop!)
        X_train_t = torch.from_numpy(X_train).to(device=device, dtype=torch.bfloat16)
        y_train_t = torch.from_numpy(y_train).to(device=device, dtype=torch.float32)
        r_train_t = torch.from_numpy(r_train).to(device=device, dtype=torch.long)
        
        X_val_t = torch.from_numpy(X_val).to(device=device, dtype=torch.bfloat16)
        y_val_t = torch.from_numpy(y_val).to(device=device, dtype=torch.float32)
        r_val_t = torch.from_numpy(r_val).to(device=device, dtype=torch.long)
        
        # Free numpy arrays
        del X_train, X_val, y_train, y_val, r_train, r_val, X, y, regime
        import gc; gc.collect()
        
        # Build sequence views via unfold (ZERO COPY - just stride metadata)
        # unfold(dim, size, step) -> (N-L+1, F, L), then permute to (N-L+1, L, F)
        X_train_seq = X_train_t.unfold(0, L, 1).permute(0, 2, 1)  # (n_train_seq, L, F)
        X_val_seq = X_val_t.unfold(0, L, 1).permute(0, 2, 1)      # (n_val_seq, L, F)
        
        # MASSIVE PERF WIN: Materialize strided views to contiguous tensors
        # Strided access from unfold() causes poor memory coalescing / hidden copies
        # On A100 (80GB), we can afford ~50GB for contiguous sequences
        if args.materialize_seqs:
            print("[CondorBrain] 🚀 Materializing sequences to CONTIGUOUS GPU tensors...")
            mat_start = time.time()
            X_train_seq = X_train_seq.contiguous()
            # Keep val strided to save ~10GB (val is less perf critical)
            torch.cuda.synchronize()
            mat_time = time.time() - mat_start
            alloc_gb = torch.cuda.memory_allocated() / 1e9
            print(f"[CondorBrain] ✅ Materialized in {mat_time:.1f}s | GPU alloc: {alloc_gb:.1f}GB")
        
        # Batch counts
        n_train_batches = n_train_seq // B
        n_val_batches = math.ceil(n_val_seq / B)
        
        # Batch getter closures (just slices, no data movement!)
        def get_train_batch(bi: int):
            s = bi * B
            e = s + B
            return X_train_seq[s:e], y_train_t[s + L:e + L], r_train_t[s + L:e + L]
        
        def get_val_batch(bi: int):
            s = bi * B
            e = min(s + B, n_val_seq)
            return X_val_seq[s:e], y_val_t[s + L:e + L], r_val_t[s + L:e + L]
        
        print(f"[CondorBrain] GPU tensors ready: {n_train_batches} train batches, {n_val_batches} val batches")
    else:
        # Fallback: CPU DataLoader path (much slower)
        print("[CondorBrain] ⚠️ gpu-dataset DISABLED: using CPU loaders (slower)")
        train_ds = BatchedSequenceDataset(X_train, y_train, r_train, L, B, drop_last=True)
        val_ds = BatchedSequenceDataset(X_val, y_val, r_val, L, B, drop_last=False)
        train_loader = DataLoader(train_ds, batch_size=None, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=None, num_workers=0)
        n_train_batches = len(train_ds)
        n_val_batches = len(val_ds)
    
    # Model with enhancement flags
    model = CondorBrain(
        d_model=args.d_model,
        n_layers=args.layers,
        input_dim=len(FEATURE_COLS),
        use_vol_gated_attn=args.vol_gated_attn,
        use_topk_moe=args.topk_moe,
        moe_n_experts=args.moe_experts,
        moe_k=args.moe_k
    ).to(device)
    
    # Force model weights to BF16 so Mamba kernels take the BF16 fast path
    if use_bf16:
        model = model.to(torch.bfloat16)
        print("[CondorBrain] Model weights converted to BF16 (including RMSNorm)")
        # CRITICAL: cuDNN GRU does NOT support BF16 - keep HorizonForecaster GRU in FP32
        if hasattr(model, 'horizon_forecaster') and hasattr(model.horizon_forecaster, 'forecast_rnn'):
            model.horizon_forecaster.forecast_rnn = model.horizon_forecaster.forecast_rnn.float()
            print("[CondorBrain] HorizonForecaster GRU kept in FP32 (cuDNN compatibility)")
        # NOTE: RMSNorm is BF16-friendly, no need for FP32 conversion
        # Loss is still computed in FP32 externally via .float() calls
    
    # Enable gradient checkpointing if requested (saves ~40% GPU memory)
    if args.grad_checkpoint:
        model.gradient_checkpointing = True
        print("[CondorBrain] Gradient checkpointing ENABLED (memory saver)")
    
    # torch.compile() (guarded). This CAN help, but depends on your exact mamba build.
    if args.compile and hasattr(torch, "compile") and device.type == "cuda":
        try:
            # Suppress compile failures and fall back cleanly
            torch._dynamo.config.suppress_errors = True
            cmode = args.compile_mode if args.compile_mode != "default" else None
            print(f"[CondorBrain] Trying torch.compile(mode={args.compile_mode}) ...")
            model = torch.compile(model, mode=cmode) if cmode else torch.compile(model)
            print("[CondorBrain] torch.compile enabled ✅")
        except Exception as e:
            print(f"[CondorBrain] torch.compile failed (continuing uncompiled): {e}")
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[CondorBrain] Model parameters: {n_params:,}")

    # Loss function selection
    if args.composite_loss:
        criterion = CompositeCondorLoss(lambdas=args.loss_lambdas_tuple)
        print(f"[CondorBrain] Using CompositeCondorLoss with lambdas={args.loss_lambdas_tuple}")
    else:
        criterion = CondorLoss()
    
    # Fused AdamW is faster on modern GPUs (avoids kernel launch overhead)
    use_fused = device.type == 'cuda' and 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=1e-4,
        fused=use_fused if use_fused else False
    )
    if use_fused:
        print("[CondorBrain] Using fused AdamW optimizer")
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Use BF16 on H100/Ampere, FP16 on older GPUs
    scaler = GradScaler() if not use_bf16 else None
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    # Training
    effective_batch = args.batch_size * args.accum_steps
    print(f"\n{'='*60}")
    print(f"CONDORBRAIN TRAINING")
    print(f"{'='*60}")
    print(f"Model: {args.d_model}d x {args.layers} layers ({n_params/1e6:.1f}M params)")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size} x {args.accum_steps} accum = {effective_batch} effective")
    print(f"LR: {args.lr}, Grad Checkpoint: {args.grad_checkpoint}")
    print(f"Output: {args.output}")
    if args.early_stop:
        print(f"Early stopping: patience={args.patience} epochs")
    if args.live_plot:
        print(f"Live plotting: enabled (updates every epoch)")
    print(f"{'='*60}\n")
    
    best_loss = float('inf')
    
    # === EARLY STOPPING & VISUALIZATION STATE ===
    patience_counter = 0
    train_losses = []
    val_losses = []
    best_epoch = 0
    
    # Live plotting setup (Colab/Jupyter compatible)
    if args.live_plot and not args.monitor:
        try:
            import matplotlib.pyplot as plt
            from IPython.display import display, clear_output
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            plt.ion()  # Interactive mode
            print("[CondorBrain] 📊 Live plotting enabled")
        except ImportError:
            print("[CondorBrain] ⚠️ matplotlib/IPython not available, disabling live-plot")
            args.live_plot = False
    
    # Advanced multi-head training monitor (replaces simple live_plot)
    monitor = None
    if args.monitor:
        monitor = TrainingMonitor(checkpoint_capacity=5)
        print("[CondorBrain] 🎯 Advanced multi-head monitor enabled (per-predictor tracking)")
        args.live_plot = False  # Monitor handles visualization
    
    # TensorBoard setup
    tb_writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            import datetime
            run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = f"{args.tb_logdir}/{run_name}"
            tb_writer = SummaryWriter(log_dir=log_dir)
            print(f"[CondorBrain] 📊 TensorBoard enabled: {log_dir}")
            print(f"[CondorBrain] 🚀 Start TensorBoard with: tensorboard --logdir={args.tb_logdir} --port={args.tb_port}")
            print(f"[CondorBrain] 🌐 Then open: http://localhost:{args.tb_port}")
        except ImportError:
            print("[CondorBrain] ⚠️ tensorboard not installed, skipping")
            args.tensorboard = False
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        epoch_start = time.time()
        optimizer.zero_grad(set_to_none=True)  # Zero at start of epoch
        
        # Direct iteration over GPU batches (FAST) or DataLoader (fallback)
        pbar = tqdm(
            range(n_train_batches),
            total=n_train_batches,
            desc=f"Epoch {epoch+1}",
            leave=False,
            mininterval=2.0,
            smoothing=0.0,
            dynamic_ncols=True,
        )
        # Throughput meter (batches/sec is misleading; report samples/sec and tokens/sec)
        _tp_last_t = time.time()
        _tp_last_i = 0
        
        for batch_idx in pbar:
            # Get batch (GPU slice if gpu_dataset, else CPU->GPU transfer)
            if use_gpu_dataset:
                batch_x, batch_y, batch_r = get_train_batch(batch_idx)
            else:
                # Fallback path: iterate through DataLoader
                _data = next(iter(train_loader))  # This is inefficient, but works as fallback
                batch_x = _data[0].to(device, non_blocking=True)
                batch_y = _data[1].to(device, non_blocking=True)
                batch_r = _data[2].to(device, non_blocking=True)
            
            with autocast('cuda', dtype=amp_dtype):
                outputs, regime_probs, _ = model(batch_x, return_regime=True, forecast_days=0)
                
                # Log dtype on first batch (sanity check)
                if batch_idx == 0 and epoch == 0:
                    print(f"[SANITY] batch_x dtype: {batch_x.dtype}")
                    print(f"[SANITY] outputs dtype: {outputs.dtype}, loss will compute in: float32")
                
                # HARD CHECK: Fail immediately on NaN outputs (don't silently skip!)
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"[NaN ERROR] outputs contained NaN/Inf at batch {batch_idx}")
                    print(f"  batch_x max: {torch.max(torch.abs(batch_x)).item():.4f}")
                    print(f"  outputs max: {torch.max(torch.abs(outputs[~torch.isnan(outputs)])).item() if (~torch.isnan(outputs)).any() else 'ALL NaN'}")
                    raise RuntimeError("Model produced NaN/Inf outputs - check data normalization!")
                
                loss = criterion(outputs.float(), batch_y.float(), regime_probs.float(), batch_r)
                # Scale loss for gradient accumulation
                loss = loss / args.accum_steps
            
            # Hard check on loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[NaN ERROR] Loss is NaN/Inf at batch {batch_idx}")
                print(f"  outputs range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
                print(f"  batch_y range: [{batch_y.min().item():.4f}, {batch_y.max().item():.4f}]")
                raise RuntimeError("Loss is NaN/Inf - check loss function!")
            
            # Backward pass (accumulate gradients)
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step every accum_steps batches
            if (batch_idx + 1) % args.accum_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            train_loss += loss.item() * args.accum_steps  # Un-scale for logging
            n_batches += 1
            
            # Update throughput display every 20 batches to keep tqdm overhead low
            if (batch_idx + 1) % 20 == 0:
                _now = time.time()
                _dt = max(_now - _tp_last_t, 1e-6)
                _batches = (batch_idx + 1) - _tp_last_i
                _samples = _batches * int(args.batch_size)
                _sps = _samples / _dt
                _tps = _sps * int(args.lookback)
                pbar.set_postfix_str(
                    f"{_sps:,.0f} samp/s | {_tps/1e6:.2f}M tok/s | loss {loss.item():.4f}"
                )
                _tp_last_t = _now
                _tp_last_i = (batch_idx + 1)
            
            # === INTRA-EPOCH VISUALIZATION (--viz-every) ===
            if args.viz_every > 0 and monitor is not None and (batch_idx + 1) % args.viz_every == 0:
                # Quick mini-validation sample (just one batch, no full val pass)
                model.eval()
                with torch.no_grad():
                    samples = sample_predictions(
                        model=model,
                        get_batch_fn=get_val_batch if use_gpu_dataset else lambda bi: next(iter(val_loader)),
                        device=device,
                        amp_dtype=amp_dtype,
                        n_samples=32
                    )
                    # Compute quick per-head metrics from this sample
                    preds_t = torch.from_numpy(samples['preds']).to(device)
                    targs_t = torch.from_numpy(samples['targets']).to(device)
                    quick_losses = {}
                    for i, name in enumerate(MAIN_HEADS):
                        quick_losses[name] = torch.mean((preds_t[:, i] - targs_t[:, i]) ** 2).item()
                    quick_losses['regime_accuracy'] = 0.0  # Placeholder
                
                # Display inline
                pbar.set_description(f"Epoch {epoch+1} [plotting]")
                display_predictions_inline(samples, epoch + 1, args.epochs, quick_losses)
                pbar.set_description(f"Epoch {epoch+1}")
                
                # Real-time TensorBoard logging (batch-level for smooth updates)
                if tb_writer is not None:
                    global_batch = epoch * n_train_batches + batch_idx
                    tb_writer.add_scalar('Batch/train_loss', loss.item(), global_batch)
                    for head_name, head_loss in quick_losses.items():
                        if head_name != 'regime_accuracy':
                            tb_writer.add_scalar(f'Batch/{head_name}', head_loss, global_batch)
                    
                    # Log regime expert activations (3 heads)
                    tb_writer.add_scalar('Regime/low_vol_expert', samples.get('regime_probs_low', 0), global_batch)
                    tb_writer.add_scalar('Regime/normal_vol_expert', samples.get('regime_probs_normal', 0), global_batch)
                    tb_writer.add_scalar('Regime/high_vol_expert', samples.get('regime_probs_high', 0), global_batch)
                    
                    # Real-time Table Summary (Markdown format for TensorBoard Text tab)
                    table_rows = [
                        "| Head | Loss | Status |",
                        "|------|------|--------|",
                    ]
                    for head_name, head_loss in quick_losses.items():
                        if head_name != 'regime_accuracy':
                            status = "✅" if head_loss < 0.5 else ("⚠️" if head_loss < 1.0 else "🔴")
                            table_rows.append(f"| {head_name} | {head_loss:.4f} | {status} |")
                    table_rows.append(f"| **Global Loss** | **{loss.item():.4f}** | - |")
                    table_md = "\n".join(table_rows)
                    tb_writer.add_text('Summary/HeadMetrics', table_md, global_batch)
                
                model.train()
        
        # Flush leftover gradients if epoch ended mid-accumulation
        if (batch_idx + 1) % args.accum_steps != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        train_loss /= max(n_batches, 1)
        
        # Validation (skip if --no-val, or run every N epochs with --val-every)
        run_val = not args.no_val and ((epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1)
        val_loss = 0.0
        
        if run_val:
            print(f"[Epoch {epoch+1}] Running validation...")
            model.eval()
            _n_val_done = 0
            
            with torch.no_grad():
                for bi in tqdm(
                    range(n_val_batches),
                    desc=f"Val {epoch+1}",
                    leave=False,
                    mininterval=2.0,
                    smoothing=0.0,
                    dynamic_ncols=True,
                ):
                    if use_gpu_dataset:
                        batch_x, batch_y, batch_r = get_val_batch(bi)
                    else:
                        _data = next(iter(val_loader))
                        batch_x = _data[0].to(device, non_blocking=True)
                        batch_y = _data[1].to(device, non_blocking=True)
                        batch_r = _data[2].to(device, non_blocking=True)
                    
                    with autocast('cuda', dtype=amp_dtype):
                        outputs, regime_probs, _ = model(batch_x, return_regime=True, forecast_days=0)
                        loss = criterion(outputs.float(), batch_y.float(), regime_probs.float(), batch_r)
                    val_loss += loss.item()
                    _n_val_done += 1
            
            val_loss /= max(_n_val_done, 1)
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']
        
        if device.type == 'cuda':
            alloc = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            val_str = f"Val: {val_loss:.4f}" if run_val else "Val: SKIP"
            print(f"Epoch {epoch+1:3d}/{args.epochs} | Train: {train_loss:.4f} | {val_str} | "
                  f"LR: {lr:.2e} | Time: {epoch_time:.1f}s | GPU alloc: {alloc:.1f}GB | reserved: {reserved:.1f}GB")
        else:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr:.2e}")
        
        # Save best (use train_loss when val is skipped)
        save_loss = val_loss if run_val else train_loss
        
        # Track losses for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss if run_val else None)
        
        # === LIVE PLOTTING ===
        if args.live_plot and len(train_losses) > 0:
            try:
                clear_output(wait=True)
                ax.clear()
                epochs_x = list(range(1, len(train_losses) + 1))
                ax.plot(epochs_x, train_losses, 'b-', label='Train Loss', linewidth=2)
                if any(v is not None for v in val_losses):
                    val_plot = [v for v in val_losses if v is not None]
                    val_x = [i+1 for i, v in enumerate(val_losses) if v is not None]
                    ax.plot(val_x, val_plot, 'r-', label='Val Loss', linewidth=2)
                    # Mark best epoch
                    if best_epoch > 0 and best_epoch <= len(val_losses) and val_losses[best_epoch-1] is not None:
                        ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best @ E{best_epoch}')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title(f'CondorBrain Training - Epoch {epoch+1}/{args.epochs}\n'
                            f'Train: {train_loss:.4f} | Val: {val_loss:.4f if run_val else "N/A"} | Best: {best_loss:.4f}')
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                display(fig)
            except Exception as e:
                print(f"[Plot error] {e}")
        
        # === ADVANCED MULTI-HEAD MONITOR ===
        if monitor is not None and run_val:
            # Compute per-head validation losses (GPU-accumulated, fast)
            head_losses = compute_val_head_losses(
                model=model,
                get_batch_fn=get_val_batch if use_gpu_dataset else lambda bi: next(iter(val_loader)),
                n_batches=n_val_batches,
                device=device,
                amp_dtype=amp_dtype
            )
            
            # Update monitor with epoch results (only checkpoints on global improvement)
            status = monitor.update(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                head_val_losses=head_losses,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler
            )
            
            # Print per-head improvement summary
            if status['improved_heads']:
                print(f"  [Monitor] Improved heads: {', '.join(status['improved_heads'])}")
            if status['improved_global']:
                print(f"  [Monitor] ★ New best global: E{status['best_epoch']} val_loss={status['best_val_loss']:.4f}")
            
            # Throttled visualization (every --monitor-every epochs, or last epoch)
            should_plot = ((epoch + 1) % args.monitor_every == 0) or (epoch == args.epochs - 1)
            if should_plot:
                # Show loss curves
                monitor.display_inline(epoch + 1, args.epochs)
                
                # Show predicted vs actual scatter plots
                samples = sample_predictions(
                    model=model,
                    get_batch_fn=get_val_batch if use_gpu_dataset else lambda bi: next(iter(val_loader)),
                    device=device,
                    amp_dtype=amp_dtype,
                    n_samples=32
                )
                display_predictions_inline(samples, epoch + 1, args.epochs, head_losses)
            
            # === TENSORBOARD LOGGING ===
            if tb_writer is not None:
                global_step = epoch + 1
                
                # Log global losses
                tb_writer.add_scalar('Loss/train', train_loss, global_step)
                tb_writer.add_scalar('Loss/val', val_loss, global_step)
                tb_writer.add_scalar('Loss/best_val', monitor.best_val_loss if monitor else best_loss, global_step)
                
                # Log per-head val losses (separate tabs in TensorBoard)
                for head_name, head_loss in head_losses.items():
                    tb_writer.add_scalar(f'HeadLoss/{head_name}', head_loss, global_step)
                
                # Log prediction vs actual scatter plots as images
                if should_plot:
                    try:
                        import matplotlib
                        matplotlib.use('Agg')
                        import matplotlib.pyplot as plt
                        import io
                        from PIL import Image
                        import numpy as np
                        
                        for i, head_name in enumerate(MAIN_HEADS):
                            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                            p = samples['preds'][:, i]
                            t = samples['targets'][:, i]
                            
                            ax.scatter(t, p, alpha=0.6, s=50, c='blue', edgecolors='black')
                            vmin, vmax = min(p.min(), t.min()), max(p.max(), t.max())
                            margin = (vmax - vmin) * 0.1 + 0.01
                            ax.plot([vmin-margin, vmax+margin], [vmin-margin, vmax+margin], 'k--', alpha=0.5)
                            
                            mae = np.mean(np.abs(p - t))
                            corr = np.corrcoef(p, t)[0, 1] if np.std(p) > 1e-6 else 0
                            ax.set_title(f'{head_name}\nMAE={mae:.4f} | r={corr:.3f}')
                            ax.set_xlabel('Actual')
                            ax.set_ylabel('Predicted')
                            ax.grid(True, alpha=0.3)
                            fig.tight_layout()
                            
                            # Convert to image for TensorBoard
                            buf = io.BytesIO()
                            fig.savefig(buf, format='png', dpi=80)
                            buf.seek(0)
                            img = Image.open(buf)
                            img_array = np.array(img)
                            plt.close(fig)
                            
                            # Add to TensorBoard (HWC format, needs CHW)
                            tb_writer.add_image(f'Predictions/{head_name}', img_array, global_step, dataformats='HWC')
                        
                        # --- LOG HORIZON FORECASTER (45-DAY TRAJECTORY) ---
                        if samples.get('forecast_data') is not None:
                            forecast = samples['forecast_data'][0]  # Sample 0
                            num_days = forecast.shape[0]
                            days = np.arange(num_days)
                            
                            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                            # forecast components: [close, high, low, vol]
                            ax.plot(days, forecast[:, 0], 'b-', label='Expected Close', linewidth=2)
                            ax.fill_between(days, forecast[:, 2], forecast[:, 1], color='blue', alpha=0.2, label='High/Low Envelope')
                            
                            ax.set_title(f'HorizonForecaster: 45-Day Price Trajectory (Epoch {global_step})')
                            ax.set_xlabel('Days from Now')
                            ax.set_ylabel('Normalized Price Change')
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            fig.tight_layout()
                            
                            buf = io.BytesIO()
                            fig.savefig(buf, format='png', dpi=100)
                            buf.seek(0)
                            img = Image.open(buf)
                            img_array = np.array(img)
                            plt.close(fig)
                            tb_writer.add_image('Horizon/Trajectory', img_array, global_step, dataformats='HWC')
                            
                        # --- LOG EXPERT SPECIFIC PREDICTIONS ---
                        if samples.get('expert_preds') is not None:
                            for expert_name, preds in samples['expert_preds'].items():
                                # Just log one representative head (e.g., call_offset) to see divergence
                                for head_idx, head_name in enumerate(['call_offset', 'put_offset']):
                                    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                                    p = preds[:, head_idx]
                                    t = samples['targets'][:, head_idx]
                                    
                                    ax.scatter(t, p, alpha=0.6, s=50, c='green', edgecolors='black')
                                    vmin, vmax = min(p.min(), t.min()), max(p.max(), t.max())
                                    margin = (vmax - vmin) * 0.1 + 0.01
                                    ax.plot([vmin-margin, vmax+margin], [vmin-margin, vmax+margin], 'k--', alpha=0.5)
                                    
                                    ax.set_title(f'Expert: {expert_name} | {head_name}')
                                    ax.grid(True, alpha=0.3)
                                    fig.tight_layout()
                                    
                                    buf = io.BytesIO()
                                    fig.savefig(buf, format='png', dpi=80)
                                    buf.seek(0)
                                    tb_writer.add_image(f'Experts_{expert_name}/{head_name}', np.array(Image.open(buf)), global_step, dataformats='HWC')
                                    plt.close(fig)
                        
                    except Exception as e:
                        print(f"[TensorBoard] Image logging error: {e}")
            
            # === SAVE EPOCH SNAPSHOT TO DISK (prevents race conditions) ===
            if monitor is not None and run_val:
                monitor.save_epoch_snapshot(epoch + 1, args.epochs)
        
        # === EARLY STOPPING CHECK ===
        if save_loss < best_loss:
            best_loss = save_loss
            best_epoch = epoch + 1
            patience_counter = 0  # Reset patience
            os.makedirs(os.path.dirname(args.output) or 'models', exist_ok=True)
            print(f"  [Saving] Writing model to {args.output}...")
            save_start = time.time()
            ckpt = {
                "state_dict": model.state_dict(),
                "feature_cols": list(FEATURE_COLS),
                "input_dim": int(len(FEATURE_COLS)),
                "seq_len": int(args.lookback),
                "version": VERSION_V22,
                "model_config": {
                    "d_model": int(args.d_model),
                    "n_layers": int(args.layers),
                    "d_state": 32,
                    "d_conv": 4,
                    "expand": 2,
                    "use_vol_gated_attn": bool(args.vol_gated_attn),
                    "use_topk_moe": bool(args.topk_moe),
                    "moe_n_experts": int(args.moe_experts),
                    "moe_k": int(args.moe_k),
                },
                "training_config": {
                    "epochs": int(args.epochs),
                    "batch_size": int(args.batch_size),
                    "lr": float(args.lr),
                    "lookback": int(args.lookback),
                    "composite_loss": bool(args.composite_loss),
                    "loss_lambdas": tuple(args.loss_lambdas_tuple),
                },
            }
            torch.save(ckpt, args.output)
            save_time = time.time() - save_start
            loss_type = "val_loss" if run_val else "train_loss"
            print(f"  ✓ Saved best model ({loss_type}={save_loss:.4f}) in {save_time:.1f}s")
        elif args.early_stop and run_val:
            patience_counter += 1
            print(f"  [Early Stop] No improvement. Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"\n🛑 EARLY STOPPING triggered at epoch {epoch+1}!")
                print(f"   Best model was at epoch {best_epoch} with val_loss={best_loss:.4f}")
                break
        
        # Clear cache periodically
        if device.type == 'cuda' and epoch % 10 == 0:
            torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"Best Val Loss: {best_loss:.4f}")
    print(f"Model saved to: {args.output}")
    print(f"{'='*60}")
    
    # Print advanced monitor summary if enabled
    if monitor is not None:
        monitor.print_summary()
        
        # Offer to save best checkpoint (model already saved, but with full state)
        ckpt_path = args.output.replace('.pth', '_full_ckpt.pth')
        monitor.save_checkpoint_to_disk(ckpt_path)
        print(f"\n\ud83d\udcbe Full checkpoint (with optimizer state) saved to: {ckpt_path}")
        print("\ud83d\udca1 TIP: Use monitor.restore_best(model) to restore optimal weights after interruption")
        
        # Save analytics JSON for post-analysis
        monitor.save_analytics_to_file("training_analytics")
        print("\ud83d\udcca Per-head best epochs saved to training_analytics/analytics.json")
    
    # Close TensorBoard writer
    if tb_writer is not None:
        tb_writer.close()
        print(f"\n\ud83d\udcca TensorBoard logs saved to: {args.tb_logdir}")
        print(f"\ud83c\udf10 View with: tensorboard --logdir={args.tb_logdir} --port={args.tb_port}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    args = parse_args()
    train_condor_brain(args)
