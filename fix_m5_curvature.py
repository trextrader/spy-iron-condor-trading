"""
fix_m5_curvature.py — Restore m5 pivot/curvature columns in-place.

The audit script zeroed out all 13 curvature/pivot columns in m5_dataset_v43_final.csv
while m1/m15/h1 remained intact. This script re-derives them from:
  - data/Datasetv4/v43/m5_dataset_v43_final.csv  (existing file, OHLC intact)
  - data/Datasetv4/pivots_manual.csv             (manual pivot definitions)

Combines both pipeline steps in one pass:
  Step 1: apply_manual_pivots  → PivotHigh, PivotLow, Slope
  Step 2: compute_institutional → PivotResidual, PivotCurvatureProxy, SlopeATR, etc.

Writes back to the same _final.csv (creates .bak first).

Run on Lightning AI:
  cd ~/spy-iron-condor-trading
  python fix_m5_curvature.py
"""
import sys
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data/Datasetv4/v43")
PIVOT_FILE = Path("data/Datasetv4/pivots_manual.csv")
M5_FILE    = DATA_DIR / "m5_dataset_v43_final.csv"

MINUTES_PER_BAR = 5
USE_HLC3        = False
EPS             = 1e-12

PIVOT_COLS = [
    "PivotHigh", "PivotLow", "Slope",
    "PivotResidual", "PivotResidualZ", "PivotCurvatureProxy",
    "SlopeATR", "PivotResidualATR", "PivotCurvatureATR",
    "PivotSegmentLengthBars", "PivotSegmentLengthMinutes",
    "PivotSegmentResidualStd", "PivotSegmentVolatility",
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_m5(path: Path) -> pd.DataFrame:
    print(f"Loading {path} ...")
    df = pd.read_csv(path)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    # Parse timestamp — try multiple formats
    for fmt in ("%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", None):
        try:
            if fmt:
                df["_ts"] = pd.to_datetime(df["timestamp"], format=fmt, errors="raise")
            else:
                df["_ts"] = pd.to_datetime(df["timestamp"], infer_datetime_format=True)
            df["_ts"] = df["_ts"].dt.floor("min")
            print(f"  Timestamp parsed with format: {fmt or 'inferred'}")
            break
        except Exception:
            continue
    else:
        raise ValueError("Could not parse timestamp column in m5 file")

    for c in ["open", "high", "low", "close", "atr_pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["_p"] = (df["high"] + df["low"] + df["close"]) / 3.0 if USE_HLC3 else df["close"]
    df["ATR_recon"] = df["atr_pct"] * df["close"] if "atr_pct" in df.columns else np.nan

    return df


def load_pivots(path: Path) -> pd.DataFrame:
    print(f"Loading {path} ...")
    piv = pd.read_csv(path)
    piv["timeframe"] = piv["timeframe"].astype(str).str.upper().str.strip()
    piv["type"]      = piv["type"].astype(str).str.upper().str.strip()
    piv["price"]     = pd.to_numeric(piv["price"], errors="coerce")
    # Parse pivot timestamps
    for fmt in ("%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M", None):
        try:
            if fmt:
                piv["ts"] = pd.to_datetime(piv["timestamp"], format=fmt, errors="raise")
            else:
                piv["ts"] = pd.to_datetime(piv["timestamp"], infer_datetime_format=True)
            piv["ts"] = piv["ts"].dt.floor("min")
            break
        except Exception:
            continue
    m5_pivots = piv[piv["timeframe"] == "M5"].copy()
    print(f"  {len(m5_pivots)} M5 pivot entries found")
    return m5_pivots


def apply_pivots(df: pd.DataFrame, m5_pivots: pd.DataFrame) -> pd.DataFrame:
    """Reset and apply manual M5 pivots → PivotHigh / PivotLow."""
    df = df.copy()
    df["PivotHigh"] = np.nan
    df["PivotLow"]  = np.nan

    ts_set = set(df["_ts"].tolist())
    applied = 0
    skipped = 0

    min_ts, max_ts = df["_ts"].min(), df["_ts"].max()
    in_range = m5_pivots[(m5_pivots["ts"] >= min_ts) & (m5_pivots["ts"] <= max_ts)]

    for _, r in in_range.iterrows():
        ts, typ, price = r["ts"], r["type"], float(r["price"])
        if ts not in ts_set:
            skipped += 1
            continue
        if typ == "HIGH":
            df.loc[df["_ts"] == ts, "PivotHigh"] = price
        elif typ == "LOW":
            df.loc[df["_ts"] == ts, "PivotLow"] = price
        else:
            skipped += 1
            continue
        applied += 1

    print(f"  Applied: {applied} pivots | Skipped (no matching bar): {skipped}")
    return df


def compute_slopes(df: pd.DataFrame) -> pd.DataFrame:
    """OLS slope per segment between consecutive pivot points."""
    df = df.copy()
    df["Slope"] = np.nan
    df["_pivot_price"] = df["PivotHigh"].where(df["PivotHigh"].notna(), df["PivotLow"])
    piv_idx = df.index[df["_pivot_price"].notna()].to_numpy()

    if len(piv_idx) < 2:
        print("  WARNING: fewer than 2 pivots — cannot compute slopes")
        return df.drop(columns=["_pivot_price"], errors="ignore")

    for i in range(len(piv_idx) - 1):
        a, b = piv_idx[i], piv_idx[i + 1]
        seg = df.loc[a:b]
        if len(seg) < 2:
            continue
        t0   = df.loc[a, "_ts"]
        x    = (df.loc[a:b, "_ts"] - t0).dt.total_seconds().to_numpy() / 60.0
        y    = df.loc[a:b, "_p"].to_numpy(dtype=float)
        xm, ym = x.mean(), y.mean()
        denom = ((x - xm) ** 2).sum()
        if denom == 0:
            continue
        slope = ((x - xm) * (y - ym)).sum() / denom
        df.loc[a:b, "Slope"] = slope

    n_slope = int(df["Slope"].notna().sum())
    print(f"  Slope filled: {n_slope:,} rows")
    return df.drop(columns=["_pivot_price"], errors="ignore")


def compute_institutional(df: pd.DataFrame) -> pd.DataFrame:
    """Curvature, residuals, ATR-normalised features per segment."""
    df = df.copy()
    for c in ["PivotResidual", "PivotResidualZ", "PivotCurvatureProxy",
              "SlopeATR", "PivotResidualATR", "PivotCurvatureATR",
              "PivotSegmentLengthBars", "PivotSegmentLengthMinutes",
              "PivotSegmentResidualStd", "PivotSegmentVolatility"]:
        df[c] = np.nan

    df["_pivot_price"] = df["PivotHigh"].where(df["PivotHigh"].notna(), df["PivotLow"])
    piv_idx = df.index[df["_pivot_price"].notna()].to_numpy()

    if len(piv_idx) < 2:
        return df.drop(columns=["_pivot_price"], errors="ignore")

    filled_resid = 0
    for i in range(len(piv_idx) - 1):
        a, b = piv_idx[i], piv_idx[i + 1]
        seg = df.loc[a:b]
        if len(seg) < 3:
            continue

        x = np.arange(len(seg), dtype=float)
        y = seg["_p"].to_numpy(dtype=float)
        xm, ym = x.mean(), y.mean()
        denom = ((x - xm) ** 2).sum()
        if denom == 0:
            continue

        slope_bar = ((x - xm) * (y - ym)).sum() / denom
        intercept = ym - slope_bar * xm
        yhat      = intercept + slope_bar * x
        residual  = y - yhat

        resid_std = float(np.nanstd(residual))
        seg_vol   = float(np.nanstd(y))

        curvature         = np.full(len(seg), np.nan)
        curvature[1:-1]   = y[2:] - 2 * y[1:-1] + y[:-2]

        idx_range = seg.index
        atr_vals  = df.loc[idx_range, "ATR_recon"].to_numpy(dtype=float)
        slope_min = slope_bar / MINUTES_PER_BAR   # $/bar → $/minute

        df.loc[idx_range, "PivotResidual"]            = residual
        df.loc[idx_range, "PivotSegmentResidualStd"]  = resid_std
        df.loc[idx_range, "PivotSegmentVolatility"]   = seg_vol
        df.loc[idx_range, "PivotSegmentLengthBars"]   = len(seg)
        df.loc[idx_range, "PivotSegmentLengthMinutes"]= len(seg) * MINUTES_PER_BAR
        df.loc[idx_range, "PivotCurvatureProxy"]      = curvature
        if resid_std > 0:
            df.loc[idx_range, "PivotResidualZ"]       = residual / resid_std
        df.loc[idx_range, "SlopeATR"]                 = slope_min / (atr_vals + EPS)
        df.loc[idx_range, "PivotResidualATR"]         = residual  / (atr_vals + EPS)
        df.loc[idx_range, "PivotCurvatureATR"]        = curvature / (atr_vals + EPS)
        filled_resid += int(np.isfinite(residual).sum())

    print(f"  PivotResidual filled: {int(df['PivotResidual'].notna().sum()):,} rows")
    print(f"  PivotCurvatureProxy filled: {int(df['PivotCurvatureProxy'].notna().sum()):,} rows")
    print(f"  SlopeATR filled: {int(df['SlopeATR'].notna().sum()):,} rows")
    return df.drop(columns=["_pivot_price"], errors="ignore")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not M5_FILE.exists():
        print(f"ERROR: {M5_FILE} not found"); sys.exit(1)
    if not PIVOT_FILE.exists():
        print(f"ERROR: {PIVOT_FILE} not found"); sys.exit(1)

    # Step 1: Load
    df        = load_m5(M5_FILE)
    m5_pivots = load_pivots(PIVOT_FILE)

    # Step 2: Apply pivots
    print("\n[Step 1] Applying M5 manual pivots ...")
    df = apply_pivots(df, m5_pivots)
    ph = int(df["PivotHigh"].notna().sum())
    pl = int(df["PivotLow"].notna().sum())
    print(f"  PivotHigh: {ph} | PivotLow: {pl}")
    if ph + pl == 0:
        print("  ERROR: No pivots matched. Check pivots_manual.csv timeframe column.")
        sys.exit(1)

    # Step 3: Slopes
    print("\n[Step 2] Computing segment slopes ...")
    df = compute_slopes(df)

    # Step 4: Institutional features
    print("\n[Step 3] Computing curvature/residual features ...")
    df = compute_institutional(df)

    # Step 5: Backup + write
    bak = M5_FILE.with_suffix(".csv.bak")
    print(f"\nBacking up → {bak}")
    shutil.copy2(M5_FILE, bak)

    drop_cols = ["_ts", "_p", "ATR_recon"]
    # Keep ATR_recon if it was already in the original file
    orig_cols = pd.read_csv(M5_FILE, nrows=0).columns.tolist()
    if "ATR_recon" in orig_cols:
        drop_cols = ["_ts", "_p"]

    out_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    # Preserve original column order
    final_cols = [c for c in orig_cols if c in out_df.columns]
    extra_cols  = [c for c in out_df.columns if c not in orig_cols]
    out_df = out_df[final_cols + extra_cols]

    print(f"Writing {len(out_df):,} rows, {len(out_df.columns)} columns → {M5_FILE}")
    out_df.to_csv(M5_FILE, index=False)

    # Step 6: Verify
    print("\n[Verify] Spot-check curvature non-null rates:")
    check = pd.read_csv(M5_FILE, usecols=["PivotCurvatureProxy", "SlopeATR", "PivotHigh"])
    for col in ["PivotHigh", "PivotCurvatureProxy", "SlopeATR"]:
        pct = check[col].notna().mean() * 100
        print(f"  {col}: {pct:.1f}% non-null")

    if check["PivotCurvatureProxy"].notna().mean() == 0:
        print("\nERROR: curvature still all-NaN after fix — check pivots_manual.csv M5 entries")
        sys.exit(1)
    else:
        print("\nSUCCESS: m5 curvature restored.")


if __name__ == "__main__":
    main()
