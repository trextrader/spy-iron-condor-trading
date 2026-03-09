#!/usr/bin/env python3
"""
pivot_proximity_chart_pack.py

Generate yearly landscape chart packs for pivot proximity pattern studies.

Signals:
    A = BB touch/break
    B = BB touch/break + MA condition
    C = BB touch/break + MA condition + adaptive PSAR bear-flip within 0..4 bars

Markers:
    PivotHigh  -> large red triangle
    PivotLow   -> large green triangle
    A-event    -> orange square
    B-event    -> blue square
    C-event    -> gold star

For pivots matched to a C-event within +/- {5,10,20} bars:
    - draw a purple downward arrow at the pivot
    - label with the smallest matching window: w5, w10, or w20

Outputs:
    charts/
        pivot_pattern_pack_2020.png
        pivot_pattern_pack_2020.pdf
        ...
    features/   (optional)
        pivot_event_features_2020_h1.csv
        ...

Usage example:
    python pivot_proximity_chart_pack.py \
        --data-root normalized_shells \
        --project-root /teamspace/studios/this_studio/spy-iron-condor-trading \
        --outdir charts \
        --export-features

Assumptions:
    - normalized shells contain:
        PivotHigh, PivotLow,
        bb_upper_dyn, bb_lower_dyn, bb_mu_dyn
    - OHLC names may be either:
        open/high/low/close
      or:
        o/h/l/c
    - datetime column may be:
        datetime, Datetime, timestamp, or date
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG
# =============================================================================

YEARS = [2020, 2021, 2022, 2023, 2024]
TIMEFRAMES = ["h1", "m15", "m5", "m1"]   # order chosen for readability in 2x2
PROX_WINDOWS = [5, 10, 20]
CSI_MAX = 4  # bars after anchor for PSAR confirmation


# =============================================================================
# COLUMN RESOLUTION
# =============================================================================

def first_existing(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Missing required column. Tried: {candidates}")
    return None


def resolve_ohlc_datetime_columns(df):
    col_open = first_existing(df, ["open", "o", "Open", "O"])
    col_high = first_existing(df, ["high", "h", "High", "H"])
    col_low = first_existing(df, ["low", "l", "Low", "L"])
    col_close = first_existing(df, ["close", "c", "Close", "C"])
    col_dt = first_existing(df, ["datetime", "Datetime", "timestamp", "Timestamp", "date", "Date"], required=False)
    return col_open, col_high, col_low, col_close, col_dt


# =============================================================================
# PSAR IMPORT
# =============================================================================

def import_compute_psar_full(project_root: str):
    """
    Import canonical adaptive PSAR from repo:
        scripts/indicators/psar_adaptive.py
    """
    if project_root:
        sys.path.append(os.path.abspath(project_root))

    try:
        from scripts.indicators.psar_adaptive import compute_psar_full
        return compute_psar_full
    except Exception as e:
        raise ImportError(
            "Could not import compute_psar_full from scripts.indicators.psar_adaptive.\n"
            "Pass --project-root to the repo root. Original error:\n"
            f"{e}"
        )


# =============================================================================
# SIGNAL CONSTRUCTION
# =============================================================================

def build_signals(df: pd.DataFrame, compute_psar_full):
    """
    Returns:
        dict with:
            x              : x-axis array
            dt             : datetime-like index or integer index
            close          : close series
            piv_hi_mask    : bool Series
            piv_lo_mask    : bool Series
            event_A        : bool Series
            event_B        : bool Series
            event_C        : bool Series
    """
    col_open, col_high, col_low, col_close, col_dt = resolve_ohlc_datetime_columns(df)

    high = pd.to_numeric(df[col_high], errors="coerce")
    low = pd.to_numeric(df[col_low], errors="coerce")
    close = pd.to_numeric(df[col_close], errors="coerce")

    bb_upper = pd.to_numeric(df["bb_upper_dyn"], errors="coerce")
    bb_lower = pd.to_numeric(df["bb_lower_dyn"], errors="coerce")
    bb_mu = pd.to_numeric(df["bb_mu_dyn"], errors="coerce")

    piv_hi_mask = df["PivotHigh"].notna()
    piv_lo_mask = df["PivotLow"].notna()
    pivot_mask = piv_hi_mask | piv_lo_mask

    # A: BB touch/break
    event_A = ((high >= bb_upper) | (low <= bb_lower)).fillna(False)

    # B: BB touch/break + MA condition
    ma_condition = ((low <= bb_mu) | (close <= bb_mu)).fillna(False)
    event_B = (event_A & ma_condition).fillna(False)

    # Canonical adaptive PSAR
    psar_df = compute_psar_full(high, low, close)
    if "psar_trend" not in psar_df.columns:
        raise KeyError("compute_psar_full(...) output does not contain 'psar_trend'.")

    psar_trend = pd.to_numeric(psar_df["psar_trend"], errors="coerce")

    # Bearish flip: +1 -> -1
    bear_flip = ((psar_trend.shift(1) == 1) & (psar_trend == -1)).fillna(False)

    # C: anchored at bars where B happens, confirmed by bearish PSAR flip within 0..4 bars
    n = len(df)
    event_C = pd.Series(False, index=df.index)
    b_idx = np.flatnonzero(event_B.to_numpy())
    bear_flip_arr = bear_flip.to_numpy()

    for j in b_idx:
        hi = min(n, j + CSI_MAX + 1)  # j..j+4 inclusive
        if bear_flip_arr[j:hi].any():
            event_C.iloc[j] = True

    # x-axis
    if col_dt is not None:
        dt = pd.to_datetime(df[col_dt], errors="coerce")
        x = dt
    else:
        dt = pd.Series(np.arange(len(df)), index=df.index)
        x = dt

    return {
        "x": x,
        "dt": dt,
        "close": close,
        "high": high,
        "low": low,
        "pivot_mask": pivot_mask,
        "piv_hi_mask": piv_hi_mask,
        "piv_lo_mask": piv_lo_mask,
        "event_A": event_A,
        "event_B": event_B,
        "event_C": event_C,
    }


# =============================================================================
# PROXIMITY / FEATURE HELPERS
# =============================================================================

def min_signed_distance(idx: int, event_idx: np.ndarray):
    """
    Signed distance from pivot idx to nearest event:
        negative => event before pivot
        positive => event after pivot
        zero     => same bar
    """
    if len(event_idx) == 0:
        return np.nan

    pos = np.searchsorted(event_idx, idx)

    candidates = []
    if pos > 0:
        candidates.append(event_idx[pos - 1] - idx)
    if pos < len(event_idx):
        candidates.append(event_idx[pos] - idx)

    if not candidates:
        return np.nan

    # choose smallest absolute distance; tie-break toward earlier event
    candidates = sorted(candidates, key=lambda z: (abs(z), z > 0))
    return int(candidates[0])


def smallest_matching_window(idx: int, event_idx: np.ndarray, windows=(5, 10, 20)):
    """
    Return smallest window label among w5/w10/w20 for which there exists
    an event within +/- window bars of pivot idx.
    """
    if len(event_idx) == 0:
        return None

    for w in windows:
        left = idx - w
        right = idx + w
        pos = np.searchsorted(event_idx, left)
        if pos < len(event_idx) and event_idx[pos] <= right:
            return f"w{w}"
    return None


def build_feature_table(df: pd.DataFrame, signals: dict, year: int, tf: str):
    """
    Export pivot-centric feature table:
      - pivot type
      - nearest signed distance to A/B/C event
      - before/after classification
      - smallest matching windows for C
    """
    piv_hi_idx = np.flatnonzero(signals["piv_hi_mask"].to_numpy())
    piv_lo_idx = np.flatnonzero(signals["piv_lo_mask"].to_numpy())

    event_A_idx = np.flatnonzero(signals["event_A"].to_numpy())
    event_B_idx = np.flatnonzero(signals["event_B"].to_numpy())
    event_C_idx = np.flatnonzero(signals["event_C"].to_numpy())

    _, _, _, _, col_dt = resolve_ohlc_datetime_columns(df)

    rows = []

    def add_rows(indices, pivot_type):
        for i in indices:
            dA = min_signed_distance(i, event_A_idx)
            dB = min_signed_distance(i, event_B_idx)
            dC = min_signed_distance(i, event_C_idx)

            rows.append({
                "year": year,
                "tf": tf,
                "pivot_index": int(i),
                "pivot_type": pivot_type,
                "datetime": df.iloc[i][col_dt] if col_dt and col_dt in df.columns else i,

                "dist_A_signed": dA,
                "dist_B_signed": dB,
                "dist_C_signed": dC,

                "A_before": int(pd.notna(dA) and dA < 0),
                "A_after": int(pd.notna(dA) and dA > 0),
                "A_samebar": int(pd.notna(dA) and dA == 0),

                "B_before": int(pd.notna(dB) and dB < 0),
                "B_after": int(pd.notna(dB) and dB > 0),
                "B_samebar": int(pd.notna(dB) and dB == 0),

                "C_before": int(pd.notna(dC) and dC < 0),
                "C_after": int(pd.notna(dC) and dC > 0),
                "C_samebar": int(pd.notna(dC) and dC == 0),

                "C_match_window": smallest_matching_window(i, event_C_idx, PROX_WINDOWS),
            })

    add_rows(piv_hi_idx, "PivotHigh")
    add_rows(piv_lo_idx, "PivotLow")

    feat = pd.DataFrame(rows).sort_values(["pivot_index", "pivot_type"]).reset_index(drop=True)
    return feat


# =============================================================================
# PLOTTING
# =============================================================================

def annotate_c_proximity(ax, x, high, pivot_idx, event_C_idx):
    """
    Add purple downward arrow + smallest matching window label above matched pivot.
    """
    y_pad = float((high.max() - high.min()) * 0.02) if len(high) else 1.0
    if not np.isfinite(y_pad) or y_pad <= 0:
        y_pad = 1.0

    for i in pivot_idx:
        label = smallest_matching_window(i, event_C_idx, PROX_WINDOWS)
        if label is None:
            continue

        xi = x.iloc[i] if hasattr(x, "iloc") else x[i]
        yi = high.iloc[i] if hasattr(high, "iloc") else high[i]

        if pd.isna(yi):
            continue

        ax.annotate(
            label,
            xy=(xi, yi + y_pad * 0.15),
            xytext=(xi, yi + y_pad * 1.3),
            textcoords="data",
            ha="center",
            va="bottom",
            fontsize=6,
            color="purple",
            arrowprops=dict(
                arrowstyle="-|>",
                color="purple",
                lw=0.8,
                shrinkA=0,
                shrinkB=0
            ),
            zorder=8,
        )


def plot_timeframe_panel(ax, df, signals, year, tf):
    x = signals["x"]
    close = signals["close"]
    high = signals["high"]

    piv_hi_mask = signals["piv_hi_mask"]
    piv_lo_mask = signals["piv_lo_mask"]
    event_A = signals["event_A"]
    event_B = signals["event_B"]
    event_C = signals["event_C"]

    # Base close line
    ax.plot(x, close, lw=0.45, alpha=0.85, label="Close", zorder=1)

    # Pivots
    # PivotHigh = red downward triangle above high
    ax.scatter(
        x[piv_hi_mask],
        high[piv_hi_mask] + hi_offset,
        marker="v",
        s=95,
        color="red",
        edgecolors="black",
        linewidths=0.35,
        zorder=6,
        label="PivotHigh"
    )

    # PivotLow = green upward triangle below low
    ax.scatter(
        x[piv_lo_mask],
        low[piv_lo_mask] - lo_offset,
        marker="^",
        s=95,
        color="green",
        edgecolors="black",
        linewidths=0.35,
        zorder=6,
        label="PivotLow"
)
    # A / B / C event markers
    ax.scatter(
        x[event_A], close[event_A],
        marker="s", s=16, color="orange", edgecolors="black", linewidths=0.20,
        label="BB break/touch", zorder=3
    )
    ax.scatter(
        x[event_B], close[event_B],
        marker="s", s=14, color="dodgerblue", edgecolors="black", linewidths=0.20,
        label="BB + MA", zorder=4
    )
    ax.scatter(
        x[event_C], close[event_C],
        marker="*", s=60, color="gold", edgecolors="black", linewidths=0.25,
        label="BB + MA + PSAR", zorder=6
    )

    # Purple arrow labels at pivots matched to event C within w5/w10/w20
    pivot_idx = np.flatnonzero((piv_hi_mask | piv_lo_mask).to_numpy())
    event_C_idx = np.flatnonzero(event_C.to_numpy())
    annotate_c_proximity(ax, x, high, pivot_idx, event_C_idx)

    # Title / stats
    n_piv = int((piv_hi_mask | piv_lo_mask).sum())
    nA = int(event_A.sum())
    nB = int(event_B.sum())
    nC = int(event_C.sum())

    ax.set_title(
        f"{tf.upper()} | year={year} | pivots={n_piv} | A={nA} | B={nB} | C={nC}",
        fontsize=10
    )
    ax.grid(alpha=0.18)
    ax.tick_params(axis="x", labelrotation=25, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)


def make_year_figure(year, data_root, compute_psar_full, outdir, export_features=False):
    fig, axes = plt.subplots(2, 2, figsize=(22, 13), constrained_layout=True)
    axes = axes.ravel()

    features_dir = Path(outdir) / "features"
    if export_features:
        features_dir.mkdir(parents=True, exist_ok=True)

    for ax, tf in zip(axes, TIMEFRAMES):
        f = Path(data_root) / str(year) / f"{tf}_dataset_v43_{year}.csv"

        if not f.exists():
            ax.set_title(f"{tf.upper()} | MISSING FILE")
            ax.axis("off")
            continue

        df = pd.read_csv(f)
        signals = build_signals(df, compute_psar_full)

        plot_timeframe_panel(ax, df, signals, year, tf)

        if export_features:
            feat = build_feature_table(df, signals, year, tf)
            feat.to_csv(features_dir / f"pivot_event_features_{year}_{tf}.csv", index=False)

    # Figure-level legend
    handles = [
        plt.Line2D([], [], color="black", lw=0.7, label="Close"),
        plt.Line2D([], [], marker="v", linestyle="None", markerfacecolor="red",
                   markeredgecolor="black", markersize=10, label="PivotHigh"),
        plt.Line2D([], [], marker="^", linestyle="None", markerfacecolor="green",
                   markeredgecolor="black", markersize=10, label="PivotLow"),
        plt.Line2D([], [], marker="s", linestyle="None", markerfacecolor="orange",
                   markeredgecolor="black", markersize=7, label="A: BB break/touch"),
        plt.Line2D([], [], marker="s", linestyle="None", markerfacecolor="dodgerblue",
                   markeredgecolor="black", markersize=7, label="B: BB + MA"),
        plt.Line2D([], [], marker="*", linestyle="None", markerfacecolor="gold",
                   markeredgecolor="black", markersize=11, label="C: BB + MA + PSAR"),
        plt.Line2D([], [], color="purple", marker=r'$\downarrow$', linestyle="None",
                   markersize=10, label="Pivot matched to C within w5/w10/w20"),
    ]

    fig.legend(handles=handles, loc="upper center", ncol=7, fontsize=9, frameon=True)

    fig.suptitle(
        f"Pivot Proximity Pattern Audit — {year}\n"
        f"A = BB touch/break | B = A + MA condition | C = B + adaptive-PSAR bear-flip within 0..{CSI_MAX} bars",
        fontsize=14,
        y=1.02
    )

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    png = outdir / f"pivot_pattern_pack_{year}.png"
    pdf = outdir / f"pivot_pattern_pack_{year}.pdf"

    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved {png}")
    print(f"[OK] Saved {pdf}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="normalized_shells",
                        help="Root folder containing year subfolders with normalized shell CSVs.")
    parser.add_argument("--project-root", type=str, default=".",
                        help="Repo root so scripts.indicators.psar_adaptive can be imported.")
    parser.add_argument("--outdir", type=str, default="charts",
                        help="Output directory for yearly chart packs.")
    parser.add_argument("--export-features", action="store_true",
                        help="Also export pivot-centric feature CSVs.")
    args = parser.parse_args()

    compute_psar_full = import_compute_psar_full(args.project_root)

    for year in YEARS:
        make_year_figure(
            year=year,
            data_root=args.data_root,
            compute_psar_full=compute_psar_full,
            outdir=args.outdir,
            export_features=args.export_features,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()