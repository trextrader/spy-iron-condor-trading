# ============================================================
# CONDORBRAIN RETRAINING TFT (PyTorch Forecasting)
# ============================================================
print("🚀 Starting CondorBrain TFT Retraining (V2.2 features + policy targets)...")

import os
import sys
import json
import csv
import argparse
import random
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

try:
    import pytorch_lightning as pl
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.metrics import MultiLoss, MSE
    HAS_TFT = True
except Exception as e:
    HAS_TFT = False
    _tft_err = e

# Add repo root to path
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
except NameError:
    repo_root = os.getcwd()

sys.path.insert(0, "/content/spy-iron-condor-trading")
sys.path.insert(0, "/kaggle/working/spy-iron-condor-trading")
sys.path.insert(0, os.getcwd())

from intelligence.canonical_feature_registry import (
    FEATURE_COLS_V22,
    INPUT_DIM_V22,
    VERSION_V22,
    apply_semantic_nan_fill,
)
from intelligence.features.dynamic_features import (
    compute_all_dynamic_features,
    compute_all_primitive_features_v22,
)
from intelligence.rule_engine.dsl_parser import RuleDSLParser
from intelligence.rule_engine.executor import RuleExecutionEngine
from audit.contract_snapshot import generate_contract_snapshot

# ------------------------------
# CONFIG + ARGS
# ------------------------------
ROWS_TO_LOAD = 500_000
EPOCHS = 6
SEQ_LEN = 256
LR = 1e-4

parser = argparse.ArgumentParser(description="CondorBrain TFT Training (V2.2)")
parser.add_argument("--start-date", type=str, default="2024-01-01")
parser.add_argument("--end-date", type=str, default="2025-01-01")
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--rows", type=int, default=None)
parser.add_argument("--input", type=str, default=None)
parser.add_argument("--artifact-dir", type=str, default="artifacts/epochs_tft")
parser.add_argument("--learned-export", dest="learned_export", action="store_true")
parser.add_argument("--no-learned-export", dest="learned_export", action="store_false")
parser.add_argument("--allow-missing-diffusion", action="store_true", help="Fill missing diffusion predictors with 0.0")
parser.set_defaults(learned_export=True)

try:
    args, _ = parser.parse_known_args()
except Exception:
    args = argparse.Namespace(
        start_date=None,
        end_date=None,
        epochs=None,
        rows=None,
        input=None,
        artifact_dir="artifacts/epochs_tft",
        learned_export=True,
    )

if args.epochs:
    EPOCHS = args.epochs
if args.rows:
    ROWS_TO_LOAD = args.rows

ARTIFACT_DIR = args.artifact_dir
LEARNED_COND_DIR = os.path.join(ARTIFACT_DIR, "learned_conditions")
os.makedirs(LEARNED_COND_DIR, exist_ok=True)
ENABLE_LEARNED_EXPORT = bool(args.learned_export)

ATTR_EVERY_N = 5
ATTR_BATCH = 4
SURROGATE_DEPTH = 6
SURROGATE_MAX_SAMPLES = 2048
ATTR_HEAD_NAMES = [
    "call_off",
    "put_off",
    "width",
    "te",
    "prob_profit",
    "expected_roi",
    "max_loss_pct",
    "confidence",
]

RULESET_PATH = "docs/Complete_Ruleset_DSL.yaml"
BASE_EXTRA_FEATURES = ["bandwidth", "sma", "psar_mark"]
DIFFUSION_FEATURE_COLS = [
    "diff_forward_return",
    "diff_forward_volatility",
    "diff_high_low_envelope",
    "diff_regime_shift",
    "diff_liquidity_shock",
    "diff_curvature_energy",
    "diff_chaos_membership",
]


def _append_csv(path: str, header: list, rows: list) -> None:
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerows(rows)


def _tree_to_markdown_rules(tree, feature_names: list) -> str:
    try:
        from sklearn.tree import _tree
    except Exception:
        return "sklearn not available; cannot render rules."
    t = tree.tree_
    feat = t.feature
    thr = t.threshold
    def rec(node: int, indent: int) -> list:
        pad = "  " * indent
        if feat[node] != _tree.TREE_UNDEFINED:
            name = feature_names[feat[node]]
            lines = []
            lines.append(f"- IF `{name} <= {thr[node]:.6g}`:")
            lines += rec(t.children_left[node], indent + 1)
            lines.append(f"- ELSE (`{name} > {thr[node]:.6g}`):")
            lines += rec(t.children_right[node], indent + 1)
            return [pad + line if line.startswith("- ") else pad + line for line in lines]
        pred = float(t.value[node][0][0])
        return [f"{pad}- THEN => **pred≈{pred:.6g}**"]
    return "\n".join(rec(0, 0))


def _export_surrogate_rules(epoch_num: int, X_flat: np.ndarray, Y: np.ndarray, feature_cols: list) -> None:
    if not ENABLE_LEARNED_EXPORT:
        return
    try:
        from sklearn.tree import DecisionTreeRegressor
    except Exception:
        err_path = os.path.join(LEARNED_COND_DIR, f"surrogate_rules_epoch_{epoch_num}.md")
        with open(err_path, "w", encoding="utf-8") as f:
            f.write("# Surrogate Rules\n\nsklearn not available; cannot train surrogate trees.\n")
        return
    combined = []
    for head_idx in range(Y.shape[1]):
        reg = DecisionTreeRegressor(max_depth=SURROGATE_DEPTH, random_state=42)
        reg.fit(X_flat, Y[:, head_idx])
        rules = _tree_to_markdown_rules(reg, feature_cols)
        head_name = ATTR_HEAD_NAMES[head_idx] if head_idx < len(ATTR_HEAD_NAMES) else f"head_{head_idx}"
        out_path = os.path.join(LEARNED_COND_DIR, f"{head_name}_rules_epoch_{epoch_num}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# {head_name} surrogate rules (depth={SURROGATE_DEPTH})\n\n")
            f.write(rules + "\n")
        latest_path = os.path.join(LEARNED_COND_DIR, f"{head_name}_rules_latest.md")
        with open(latest_path, "w", encoding="utf-8") as f:
            f.write(f"# {head_name} surrogate rules (depth={SURROGATE_DEPTH})\n\n")
            f.write(rules + "\n")
        combined.append(f"## {head_name}\n\n{rules}\n")
    combined_path = os.path.join(LEARNED_COND_DIR, f"all_rules_epoch_{epoch_num}.md")
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write(f"# All heads surrogate rules (depth={SURROGATE_DEPTH})\n\n")
        f.write("\n".join(combined))
    latest_combined = os.path.join(LEARNED_COND_DIR, "all_rules_latest.md")
    with open(latest_combined, "w", encoding="utf-8") as f:
        f.write(f"# All heads surrogate rules (depth={SURROGATE_DEPTH})\n\n")
        f.write("\n".join(combined))


def _export_attribution(epoch_num: int, batch_idx: int, model, batch_x, feature_cols: list, schema_id: str) -> None:
    if not ENABLE_LEARNED_EXPORT:
        return
    enc = batch_x.get("encoder_cont")
    if enc is None or enc.shape[0] == 0:
        return
    batch_size = min(ATTR_BATCH, enc.shape[0])
    x_attr = enc[:batch_size].detach().clone().requires_grad_(True)

    model.zero_grad(set_to_none=True)
    pred = model({"encoder_cont": x_attr, **{k: v[:batch_size] for k, v in batch_x.items() if k != "encoder_cont"}})
    if isinstance(pred, dict):
        y_hat = pred["prediction"]
    else:
        y_hat = pred
    y_hat = y_hat[:, 0, :]  # [B, heads]

    header = [
        "ts_utc",
        "epoch",
        "batch",
        "head_index",
        "head_name",
        "feature_schema_id",
        "feature_count",
        "head_count",
        "head_names",
        "feature",
        "importance",
        "contribution",
    ]
    rows = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    for head_idx in range(y_hat.shape[1]):
        model.zero_grad(set_to_none=True)
        out = y_hat[:, head_idx].sum()
        grads = torch.autograd.grad(out, x_attr, retain_graph=True)[0]
        attr = grads * x_attr
        imp = attr.abs().mean(dim=(0, 1)).detach().cpu().numpy()
        contrib = attr.mean(dim=(0, 1)).detach().cpu().numpy()
        head_name = ATTR_HEAD_NAMES[head_idx] if head_idx < len(ATTR_HEAD_NAMES) else f"head_{head_idx}"
        for j, feat in enumerate(feature_cols):
            rows.append([
                ts,
                int(epoch_num),
                int(batch_idx),
                int(head_idx),
                head_name,
                schema_id,
                len(feature_cols),
                int(y_hat.shape[1]),
                "|".join(ATTR_HEAD_NAMES),
                feat,
                float(imp[j]),
                float(contrib[j]),
            ])
    out_path = os.path.join(LEARNED_COND_DIR, "live_attribution.csv")
    _append_csv(out_path, header, rows)


def _surrogate_update(buf_x: list, buf_y: list, seen: int, x_last: np.ndarray, y_out: np.ndarray) -> int:
    if not ENABLE_LEARNED_EXPORT:
        return seen
    for i in range(x_last.shape[0]):
        seen += 1
        if len(buf_x) < SURROGATE_MAX_SAMPLES:
            buf_x.append(x_last[i].copy())
            buf_y.append(y_out[i].copy())
        else:
            j = random.randint(0, seen - 1)
            if j < SURROGATE_MAX_SAMPLES:
                buf_x[j] = x_last[i].copy()
                buf_y[j] = y_out[i].copy()
    return seen


def _rule_feature_name(rule_id: str) -> str:
    name = rule_id.lower()
    if name.startswith("rule_"):
        name = name[5:]
    return f"rule_{name}_signal"


def _apply_rule_engine(spot_df: pd.DataFrame) -> list:
    parser = RuleDSLParser(RULESET_PATH)
    ruleset = parser.load()
    engine = RuleExecutionEngine(ruleset)
    rule_data = {col: spot_df[col] for col in spot_df.columns}
    rule_results = engine.execute(rule_data)
    rule_cols = []
    for rid in sorted(rule_results.keys()):
        res = rule_results[rid]
        sig_long = res.get("signal_long", 0.0)
        sig_short = res.get("signal_short", 0.0)
        sig_exit = res.get("signal_exit", 0.0)
        sig_block = res.get("blocked", 0.0)
        if not hasattr(sig_long, "astype"):
            sig_long = pd.Series(sig_long, index=spot_df.index)
        if not hasattr(sig_short, "astype"):
            sig_short = pd.Series(sig_short, index=spot_df.index)
        if not hasattr(sig_exit, "astype"):
            sig_exit = pd.Series(sig_exit, index=spot_df.index)
        if not hasattr(sig_block, "astype"):
            sig_block = pd.Series(sig_block, index=spot_df.index)
        signal = sig_long.astype(float).fillna(0.0) - sig_short.astype(float).fillna(0.0)
        signal = signal.where(~sig_exit.astype(bool), 2.0)
        signal = signal.where(~sig_block.astype(bool), -2.0)
        col = _rule_feature_name(rid)
        spot_df[col] = signal.astype(np.float32)
        rule_cols.append(col)
    return rule_cols


def _ensure_base_extras(df: pd.DataFrame) -> None:
    if "sma" not in df.columns:
        df["sma"] = df["close"].rolling(20, min_periods=1).mean().astype(np.float32)
    if "psar_mark" not in df.columns:
        if "psar_trend" in df.columns:
            df["psar_mark"] = np.where(df["psar_trend"] >= 0, 1.0, -1.0).astype(np.float32)
        else:
            df["psar_mark"] = 0.0
    if "bandwidth" not in df.columns:
        if "bb_sigma_dyn" in df.columns:
            df["bandwidth"] = df["bb_sigma_dyn"].astype(np.float32)
        else:
            df["bandwidth"] = 0.0


if not HAS_TFT:
    raise SystemExit(
        "pytorch_forecasting not installed. Install with:\n"
        "pip install pytorch-forecasting pytorch-lightning"
    )

# ------------------------------
# DATA LOAD + FEATURES
# ------------------------------
input_path = args.input
search_paths = [
    input_path,
    "data/processed/mamba_institutional_2024_1m_last 1mil.csv",
    "data/processed/mamba_institutional_2025_1m.csv",
]
search_paths = [p for p in search_paths if p]
data_path = None
for p in search_paths:
    if os.path.exists(p):
        data_path = p
        break
if not data_path:
    raise SystemExit("No input dataset found. Use --input to specify.")

df = pd.read_csv(data_path, nrows=ROWS_TO_LOAD)
if "timestamp" in df.columns and "dt" not in df.columns:
    df = df.rename(columns={"timestamp": "dt"})
if "dt" in df.columns:
    df["dt"] = pd.to_datetime(df["dt"], utc=True)
    if args.start_date and args.end_date:
        mask = (df["dt"] >= args.start_date) & (df["dt"] < args.end_date)
        df = df.loc[mask].copy()

# Compute dynamic + primitive features on unique spot bars, then merge back
ohlcv_cols = ["open", "high", "low", "close", "volume"]
spot_key_cols = ["symbol", "dt"] if "dt" in df.columns and "symbol" in df.columns else ["dt"] if "dt" in df.columns else []
if not spot_key_cols:
    spot_key_cols = ["timestamp"] if "timestamp" in df.columns else ["index"]
if "index" in spot_key_cols:
    df = df.reset_index(drop=False).rename(columns={"index": "index"})

aux_cols = [c for c in ["spread_ratio", "lag_minutes"] if c in df.columns]
spot_df = df.drop_duplicates(subset=spot_key_cols)[spot_key_cols + ohlcv_cols + aux_cols].copy()
spot_df = spot_df.sort_values(spot_key_cols).reset_index(drop=True)

spot_df = compute_all_dynamic_features(spot_df, close_col="close", high_col="high", low_col="low")
spot_df = compute_all_primitive_features_v22(
    spot_df,
    close_col="close",
    high_col="high",
    low_col="low",
    spread_col="spread_ratio" if "spread_ratio" in spot_df.columns else "close",
)

try:
    rule_feature_cols = _apply_rule_engine(spot_df)
except Exception as e:
    print(f"⚠️ Rule engine error: {e}. Filling rule features with zeros.")
    try:
        ruleset = RuleDSLParser(RULESET_PATH).load()
        rule_ids = sorted(ruleset.rules.keys())
    except Exception:
        rule_ids = []
    rule_feature_cols = []
    for rid in rule_ids:
        col = _rule_feature_name(rid)
        spot_df[col] = 0.0
        rule_feature_cols.append(col)
_ensure_base_extras(spot_df)

exclude_cols = spot_key_cols + ohlcv_cols + aux_cols
dynamic_cols = [c for c in spot_df.columns if c not in exclude_cols]
cols_to_drop = [c for c in dynamic_cols if c in df.columns]
if cols_to_drop:
    df.drop(columns=cols_to_drop, inplace=True)
merge_cols = spot_key_cols + dynamic_cols
df = df.merge(spot_df[merge_cols], on=spot_key_cols, how="left")

_ensure_base_extras(df)

base_feature_cols = FEATURE_COLS_V22 + BASE_EXTRA_FEATURES
feature_cols = base_feature_cols + rule_feature_cols + DIFFUSION_FEATURE_COLS

missing_diffusion = [c for c in DIFFUSION_FEATURE_COLS if c not in df.columns]
if missing_diffusion:
    if args.allow_missing_diffusion:
        for c in missing_diffusion:
            df[c] = 0.0
    else:
        raise SystemExit(f"Missing diffusion predictors: {missing_diffusion}. Use --allow-missing-diffusion to fill with zeros.")

for col in feature_cols:
    if col not in df.columns:
        df[col] = 0.0

X_np = df[feature_cols].values.astype(np.float32)
X_np = apply_semantic_nan_fill(X_np, feature_cols)

# Split
split_idx = int(len(X_np) * 0.8)
X_train_np = X_np[:split_idx]
X_val_np = X_np[split_idx:]

median = np.median(X_train_np, axis=0, keepdims=True).astype(np.float32)
mad = np.median(np.abs(X_train_np - median), axis=0, keepdims=True).astype(np.float32)
mad = np.maximum(mad, 1e-6).astype(np.float32)

def robust_norm(x):
    x = np.nan_to_num(x, nan=0.0)
    out = (x - median) / (1.4826 * mad)
    return np.clip(out, -10.0, 10.0).astype(np.float32)

X_train_np = robust_norm(X_train_np)
X_val_np = robust_norm(X_val_np)

# Policy targets
close = df["close"].values.astype(np.float32)
future_close = np.empty_like(close)
future_close[:-60] = close[60:]
future_close[-60:] = np.nan
returns_60m = (future_close - close).astype(np.float32)
returns_60m = np.nan_to_num(returns_60m, nan=0.0)
vol_60m = (
    pd.Series(returns_60m)
    .rolling(60, min_periods=1)
    .std(ddof=0)
    .fillna(0.0)
    .to_numpy(np.float32)
)

Y_policy_np = np.zeros((len(X_np), 8), dtype=np.float32)
Y_policy_np[:, 0] = 2.0 + np.clip(returns_60m * 0.5, -1.0, 1.0)
Y_policy_np[:, 1] = 2.0 - np.clip(returns_60m * 0.5, -1.0, 1.0)
Y_policy_np[:, 2] = 5.0 + np.clip(vol_60m * 20, 0, 5)
Y_policy_np[:, 4] = 0.5 + np.clip(returns_60m * 0.1, -0.4, 0.4)
Y_policy_np[:, 7] = 0.3 + np.clip(np.abs(returns_60m), 0.0, 0.6)

target_cols = [f"y_{i}" for i in range(8)]
df_targets = pd.DataFrame(Y_policy_np, columns=target_cols)
df_targets["time_idx"] = np.arange(len(Y_policy_np))
df_targets["series_id"] = 0
df_features = pd.DataFrame(X_np, columns=feature_cols)
df_all = pd.concat([df_targets, df_features], axis=1)

categorical_cols = []
for col in ["symbol", "call_put"]:
    if col in df.columns:
        df_all[col] = df[col].astype(str).fillna("NA")
        categorical_cols.append(col)

FEATURE_SCHEMA_ID = hashlib.sha256(",".join(feature_cols).encode("utf-8")).hexdigest()
if ENABLE_LEARNED_EXPORT:
    meta_path = os.path.join(LEARNED_COND_DIR, "live_attribution_meta.json")
    if not os.path.exists(meta_path):
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "feature_schema_id": FEATURE_SCHEMA_ID,
                "feature_count": len(feature_cols),
                "feature_cols": feature_cols,
                "base_feature_cols": base_feature_cols,
                "rule_feature_cols": rule_feature_cols,
                "diffusion_feature_cols": DIFFUSION_FEATURE_COLS,
                "categorical_cols": categorical_cols,
                "head_names": ATTR_HEAD_NAMES,
                "attr_every_n_batches": ATTR_EVERY_N,
                "attr_batch_size": ATTR_BATCH,
            }, f, indent=2)

train_df = df_all.iloc[:split_idx].copy()
val_df = df_all.iloc[split_idx:].copy()

training = TimeSeriesDataSet(
    train_df,
    time_idx="time_idx",
    target=target_cols,
    group_ids=["series_id"],
    max_encoder_length=SEQ_LEN,
    max_prediction_length=1,
    time_varying_unknown_reals=feature_cols,
    time_varying_known_categoricals=categorical_cols,
)

validation = TimeSeriesDataSet(
    val_df,
    time_idx="time_idx",
    target=target_cols,
    group_ids=["series_id"],
    max_encoder_length=SEQ_LEN,
    max_prediction_length=1,
    time_varying_unknown_reals=feature_cols,
    time_varying_known_categoricals=categorical_cols,
)

train_loader = training.to_dataloader(train=True, batch_size=128, num_workers=0)
val_loader = validation.to_dataloader(train=False, batch_size=128, num_workers=0)

model = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=LR,
    hidden_size=64,
    attention_head_size=4,
    dropout=0.1,
    loss=MultiLoss([MSE()] * len(target_cols)),
    output_size=[1] * len(target_cols),
)

class LearnedConditionsCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.surrogate_x = []
        self.surrogate_y = []
        self.surrogate_seen = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not ENABLE_LEARNED_EXPORT:
            return
        x, _y = batch
        if (batch_idx + 1) % ATTR_EVERY_N == 0:
            try:
                _export_attribution(trainer.current_epoch + 1, batch_idx + 1, pl_module, x, feature_cols, FEATURE_SCHEMA_ID)
            except Exception as e:
                print(f"⚠️ Attribution export failed: {e}")
        try:
            enc = x["encoder_cont"]
            pred = pl_module(x)
            if isinstance(pred, dict):
                y_hat = pred["prediction"][:, 0, :]
            else:
                y_hat = pred[:, 0, :]
            x_last = enc[:, -1, :].detach().cpu().numpy()
            y_out = y_hat.detach().cpu().numpy()
            self.surrogate_seen = _surrogate_update(self.surrogate_x, self.surrogate_y, self.surrogate_seen, x_last, y_out)
        except Exception as e:
            print(f"⚠️ Surrogate buffer update failed: {e}")

    def on_train_epoch_end(self, trainer, pl_module):
        if not ENABLE_LEARNED_EXPORT:
            return
        if self.surrogate_x and self.surrogate_y:
            try:
                X_flat = np.stack(self.surrogate_x, axis=0)
                Y_out = np.stack(self.surrogate_y, axis=0)
                _export_surrogate_rules(trainer.current_epoch + 1, X_flat, Y_out, feature_cols)
            except Exception as e:
                print(f"⚠️ Surrogate rule export failed: {e}")
        epoch_num = trainer.current_epoch + 1
        ckpt = {
            "state_dict": pl_module.state_dict(),
            "arch": "tft",
            "version": VERSION_V22,
            "feature_cols": feature_cols,
            "input_dim": INPUT_DIM_V22,
            "median": median.astype(np.float32),
            "mad": mad.astype(np.float32),
            "seq_len": SEQ_LEN,
            "head_names": ATTR_HEAD_NAMES,
        }
        epoch_dir = os.path.join(ARTIFACT_DIR, f"epoch_{epoch_num:03d}")
        os.makedirs(epoch_dir, exist_ok=True)
        save_path = os.path.join(epoch_dir, f"checkpoint_e{epoch_num}.pth")
        torch.save(ckpt, save_path)
        generate_contract_snapshot(
            os.path.join(ARTIFACT_DIR, "contract_snapshot.json"),
            repo_root,
            feature_cols=feature_cols,
            checkpoint_path=save_path,
            extra={
                "epoch": epoch_num,
                "arch": "tft",
                "base_feature_cols": base_feature_cols,
                "rule_feature_cols": rule_feature_cols,
                "diffusion_feature_cols": DIFFUSION_FEATURE_COLS,
                "categorical_cols": categorical_cols,
                "ruleset_path": RULESET_PATH,
                "allow_missing_diffusion": bool(args.allow_missing_diffusion),
            },
        )
        print(f"      💾 Saved: {save_path}")

trainer = pl.Trainer(
    max_epochs=EPOCHS,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    enable_checkpointing=False,
    logger=False,
    callbacks=[LearnedConditionsCallback()],
)

trainer.fit(model, train_loader, val_loader)

print("✅ TFT Retraining Complete.")
