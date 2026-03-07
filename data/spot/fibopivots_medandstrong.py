#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(".")
YEAR_DIRS = ["2020", "2021", "2022", "2023", "2024"]
TIMEFRAMES = ["m1", "m5", "m15", "h1"]

NY_TZ = "America/New_York"

# Strong pivot parameters: set these to match TradingView exactly.
STRONG_LEFT = 35
STRONG_RIGHT = 40

# Fib levels to export/draw
FIB_LEVELS = [
    -0.618,
    -0.382,
    -0.236,
    0.0,
    0.236,
    0.382,
    0.5,
    0.618,
    0.786,
    1.0,
    1.272,
    1.414,
    1.618,
    2.618,
    3.618,
    4.236,
]

# Plotting
MAKE_PLOTS = True
MAX_PLOT_BARS = 1200          # avoids giant unreadable plots
PLOT_LAST_N_SEGMENTS = 8      # fib segments to draw on chart
OUTPUT_DIRNAME = "pivot_exports"


# ============================================================
# IO / NORMALIZATION
# ============================================================

def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    ts_col = None
    for c in ["datetime", "timestamp", "Datetime", "Timestamp", "date", "time"]:
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        raise ValueError(f"No timestamp column found in {path}. Columns={list(df.columns)}")

    df[ts_col] = pd.to_datetime(df[ts_col], utc=True).dt.tz_convert(NY_TZ)
    df = df.rename(columns={ts_col: "datetime"})

    rename_map = {}
    if "o" in df.columns:
        rename_map["o"] = "open"
    if "h" in df.columns:
        rename_map["h"] = "high"
    if "l" in df.columns:
        rename_map["l"] = "low"
    if "c" in df.columns:
        rename_map["c"] = "close"
    if "v" in df.columns:
        rename_map["v"] = "volume"

    df = df.rename(columns=rename_map)

    required = ["datetime", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df = df.sort_values("datetime").reset_index(drop=True)
    return df


# ============================================================
# PIVOT DETECTION
# ============================================================

def pivot_high_tv(high: np.ndarray, left: int, right: int) -> np.ndarray:
    n = len(high)
    out = np.zeros(n, dtype=bool)

    for i in range(left, n - right):
        center = high[i]
        if not np.isfinite(center):
            continue
        window = high[i - left : i + right + 1]
        if center == np.nanmax(window) and np.sum(window == center) == 1:
            out[i] = True

    return out


def pivot_low_tv(low: np.ndarray, left: int, right: int) -> np.ndarray:
    n = len(low)
    out = np.zeros(n, dtype=bool)

    for i in range(left, n - right):
        center = low[i]
        if not np.isfinite(center):
            continue
        window = low[i - left : i + right + 1]
        if center == np.nanmin(window) and np.sum(window == center) == 1:
            out[i] = True

    return out


def build_raw_pivot_df(
    df: pd.DataFrame,
    left: int,
    right: int,
    strength_name: str,
) -> pd.DataFrame:
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    ph = pivot_high_tv(high, left, right)
    pl = pivot_low_tv(low, left, right)

    raw = pd.DataFrame({
        "pivot_index": np.arange(len(df), dtype=int),
        "pivot_time": df["datetime"].to_numpy(),
        "pivot_high": ph.astype(int),
        "pivot_low": pl.astype(int),
        "pivot_price": np.full(len(df), np.nan, dtype=float),
        "pivot_type": pd.Series([""] * len(df), dtype="object"),
        "confirmed_on": pd.Series([pd.NaT] * len(df), dtype=f"datetime64[ns, {NY_TZ}]"),
        "left_bars": np.full(len(df), left, dtype=int),
        "right_bars": np.full(len(df), right, dtype=int),
        "strength": pd.Series([strength_name] * len(df), dtype="object"),
    })

    raw.loc[raw["pivot_high"] == 1, "pivot_price"] = df.loc[raw["pivot_high"] == 1, "high"].to_numpy()
    raw.loc[raw["pivot_low"] == 1, "pivot_price"] = df.loc[raw["pivot_low"] == 1, "low"].to_numpy()
    raw.loc[raw["pivot_high"] == 1, "pivot_type"] = "HIGH"
    raw.loc[raw["pivot_low"] == 1, "pivot_type"] = "LOW"

    pivot_idx = np.where(ph | pl)[0]
    confirm_idx = pivot_idx + right
    valid = confirm_idx < len(df)

    if np.any(valid):
        confirmed_vals = pd.Series(
            pd.DatetimeIndex(df.loc[confirm_idx[valid], "datetime"]).tz_convert(NY_TZ),
            index=raw.index[pivot_idx[valid]],
            dtype=f"datetime64[ns, {NY_TZ}]",
        )
        raw.loc[pivot_idx[valid], "confirmed_on"] = confirmed_vals

    raw = raw[(raw["pivot_high"] == 1) | (raw["pivot_low"] == 1)].copy()
    raw = raw.reset_index(drop=True)
    return raw


# ============================================================
# ALTERNATING STRONG STRUCTURE
# ============================================================

def reduce_to_alternating_structure(raw_pivots: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce raw pivots to alternating HIGH/LOW structure.
    If consecutive pivots have same type, keep the stronger extreme:
      - HIGH: keep higher high
      - LOW : keep lower low
    """
    if raw_pivots.empty:
        return raw_pivots.copy()

    kept: List[dict] = []

    for row in raw_pivots.to_dict("records"):
        if not kept:
            kept.append(row)
            continue

        prev = kept[-1]

        if row["pivot_type"] == prev["pivot_type"]:
            if row["pivot_type"] == "HIGH":
                if float(row["pivot_price"]) > float(prev["pivot_price"]):
                    kept[-1] = row
            else:
                if float(row["pivot_price"]) < float(prev["pivot_price"]):
                    kept[-1] = row
        else:
            kept.append(row)

    out = pd.DataFrame(kept).reset_index(drop=True)
    out["pivot_seq_id"] = np.arange(len(out), dtype=int)

    if "confirmed_on" in out.columns:
        out["confirmed_on"] = pd.to_datetime(out["confirmed_on"], utc=False)
        if getattr(out["confirmed_on"].dt, "tz", None) is None:
            out["confirmed_on"] = out["confirmed_on"].dt.tz_localize(NY_TZ, nonexistent="shift_forward", ambiguous="NaT")
        else:
            out["confirmed_on"] = out["confirmed_on"].dt.tz_convert(NY_TZ)

    return out


# ============================================================
# FIB SEGMENTS
# ============================================================

def fib_price_from_leg(start_price: float, end_price: float, level: float) -> float:
    """
    Retracement/extension from leg start->end.
    level=0   => start
    level=1   => end
    >1        => extension beyond end
    <0        => extension beyond start opposite side
    """
    return start_price + (end_price - start_price) * level


def build_fib_segments(pivots: pd.DataFrame, fib_levels: List[float]) -> pd.DataFrame:
    """
    Build one fib segment per alternating pivot leg.
    """
    rows: List[dict] = []

    if len(pivots) < 2:
        return pd.DataFrame()

    for i in range(1, len(pivots)):
        p0 = pivots.iloc[i - 1]
        p1 = pivots.iloc[i]

        start_idx = int(p0["pivot_index"])
        end_idx = int(p1["pivot_index"])
        start_time = p0["pivot_time"]
        end_time = p1["pivot_time"]
        start_price = float(p0["pivot_price"])
        end_price = float(p1["pivot_price"])

        direction = "UP" if end_price > start_price else "DOWN"
        magnitude = abs(end_price - start_price)

        row = {
            "segment_id": i - 1,
            "start_pivot_seq_id": int(p0["pivot_seq_id"]),
            "end_pivot_seq_id": int(p1["pivot_seq_id"]),
            "start_index": start_idx,
            "end_index": end_idx,
            "start_time": start_time,
            "end_time": end_time,
            "start_type": p0["pivot_type"],
            "end_type": p1["pivot_type"],
            "start_price": start_price,
            "end_price": end_price,
            "direction": direction,
            "magnitude": magnitude,
            "bars_in_leg": end_idx - start_idx,
        }

        for lvl in fib_levels:
            col = f"fib_{str(lvl).replace('-', 'neg_').replace('.', '_')}"
            row[col] = fib_price_from_leg(start_price, end_price, lvl)

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# BAR-LEVEL MERGESET FOR MODEL INPUTS
# ============================================================

def build_bar_level_dataset(
    df: pd.DataFrame,
    strong_pivots: pd.DataFrame,
    medium_pivots: pd.DataFrame,
    fib_segments: pd.DataFrame,
    fib_levels: List[float],
) -> pd.DataFrame:
    out = df.copy()

    # Strong pivot columns
    out["strong_pivot_high"] = pd.Series(np.zeros(len(out), dtype=int), index=out.index)
    out["strong_pivot_low"] = pd.Series(np.zeros(len(out), dtype=int), index=out.index)
    out["strong_pivot_price"] = pd.Series(np.full(len(out), np.nan, dtype=float), index=out.index)
    out["strong_pivot_type"] = pd.Series([""] * len(out), index=out.index, dtype="object")
    out["strong_pivot_confirmed_on"] = pd.Series([pd.NaT] * len(out), index=out.index, dtype=f"datetime64[ns, {NY_TZ}]")

    # Medium pivot columns
    out["medium_pivot_high"] = pd.Series(np.zeros(len(out), dtype=int), index=out.index)
    out["medium_pivot_low"] = pd.Series(np.zeros(len(out), dtype=int), index=out.index)
    out["medium_pivot_price"] = pd.Series(np.full(len(out), np.nan, dtype=float), index=out.index)
    out["medium_pivot_type"] = pd.Series([""] * len(out), index=out.index, dtype="object")
    out["medium_pivot_confirmed_on"] = pd.Series([pd.NaT] * len(out), index=out.index, dtype=f"datetime64[ns, {NY_TZ}]")

    if not strong_pivots.empty:
        high_mask = strong_pivots["pivot_type"] == "HIGH"
        low_mask = strong_pivots["pivot_type"] == "LOW"

        sidx = strong_pivots["pivot_index"].to_numpy()
        out.loc[strong_pivots.loc[high_mask, "pivot_index"].to_numpy(), "strong_pivot_high"] = 1
        out.loc[strong_pivots.loc[low_mask, "pivot_index"].to_numpy(), "strong_pivot_low"] = 1
        out.loc[sidx, "strong_pivot_price"] = strong_pivots["pivot_price"].to_numpy()
        out.loc[sidx, "strong_pivot_type"] = strong_pivots["pivot_type"].to_numpy()

        confirmed_vals = pd.Series(
            pd.DatetimeIndex(strong_pivots["confirmed_on"]).tz_convert(NY_TZ),
            index=out.index[sidx],
            dtype=f"datetime64[ns, {NY_TZ}]",
        )
        out.loc[sidx, "strong_pivot_confirmed_on"] = confirmed_vals

    if not medium_pivots.empty:
        high_mask = medium_pivots["pivot_type"] == "HIGH"
        low_mask = medium_pivots["pivot_type"] == "LOW"

        midx = medium_pivots["pivot_index"].to_numpy()
        out.loc[medium_pivots.loc[high_mask, "pivot_index"].to_numpy(), "medium_pivot_high"] = 1
        out.loc[medium_pivots.loc[low_mask, "pivot_index"].to_numpy(), "medium_pivot_low"] = 1
        out.loc[midx, "medium_pivot_price"] = medium_pivots["pivot_price"].to_numpy()
        out.loc[midx, "medium_pivot_type"] = medium_pivots["pivot_type"].to_numpy()

        confirmed_vals = pd.Series(
            pd.DatetimeIndex(medium_pivots["confirmed_on"]).tz_convert(NY_TZ),
            index=out.index[midx],
            dtype=f"datetime64[ns, {NY_TZ}]",
        )
        out.loc[midx, "medium_pivot_confirmed_on"] = confirmed_vals

    out["last_strong_pivot_price"] = out["strong_pivot_price"].ffill()
    out["last_strong_pivot_type"] = out["strong_pivot_type"].replace("", np.nan).ffill()

    out["active_segment_id"] = pd.Series(np.full(len(out), np.nan, dtype=float), index=out.index)
    out["active_segment_direction"] = pd.Series([None] * len(out), index=out.index, dtype="object")
    out["active_segment_start_price"] = pd.Series(np.full(len(out), np.nan, dtype=float), index=out.index)
    out["active_segment_end_price"] = pd.Series(np.full(len(out), np.nan, dtype=float), index=out.index)

    for lvl in fib_levels:
        col = f"active_fib_{str(lvl).replace('-', 'neg_').replace('.', '_')}"
        out[col] = pd.Series(np.full(len(out), np.nan, dtype=float), index=out.index)

    if not fib_segments.empty:
        for _, seg in fib_segments.iterrows():
            start_idx = int(seg["end_index"])
            out.loc[start_idx:, "active_segment_id"] = int(seg["segment_id"])
            out.loc[start_idx:, "active_segment_direction"] = str(seg["direction"])
            out.loc[start_idx:, "active_segment_start_price"] = float(seg["start_price"])
            out.loc[start_idx:, "active_segment_end_price"] = float(seg["end_price"])

            for lvl in fib_levels:
                fib_col_src = f"fib_{str(lvl).replace('-', 'neg_').replace('.', '_')}"
                fib_col_dst = f"active_fib_{str(lvl).replace('-', 'neg_').replace('.', '_')}"
                out.loc[start_idx:, fib_col_dst] = float(seg[fib_col_src])

    return out


# ============================================================
# PLOTTING
# ============================================================

def draw_candles(ax, df_plot: pd.DataFrame) -> None:
    x = np.arange(len(df_plot))
    width = 0.6

    for i, row in enumerate(df_plot.itertuples(index=False)):
        o = float(row.open)
        h = float(row.high)
        l = float(row.low)
        c = float(row.close)

        up = c >= o
        color = "#2ca02c" if up else "#d62728"

        ax.vlines(i, l, h, linewidth=0.8)
        body_low = min(o, c)
        body_h = max(abs(c - o), 1e-9)
        rect = Rectangle((i - width / 2, body_low), width, body_h, fill=False)
        ax.add_patch(rect)
        rect.set_linewidth(0.8)
        rect.set_edgecolor(color)
        rect.set_facecolor("none")


def plot_dataset_with_pivots_and_fibs(
    df: pd.DataFrame,
    strong_pivots: pd.DataFrame,
    medium_pivots: pd.DataFrame,
    fib_segments: pd.DataFrame,
    title: str,
    out_png: Path,
    max_bars: int = MAX_PLOT_BARS,
    last_n_segments: int = PLOT_LAST_N_SEGMENTS,
) -> None:
    if len(df) == 0:
        return

    if len(df) > max_bars:
        df_plot = df.iloc[-max_bars:].copy()
        start_global_idx = len(df) - max_bars
    else:
        df_plot = df.copy()
        start_global_idx = 0

    fig, ax = plt.subplots(figsize=(18, 9))
    draw_candles(ax, df_plot)

    # Medium pivots
    if not medium_pivots.empty:
        mp = medium_pivots[medium_pivots["pivot_index"] >= start_global_idx].copy()
        if not mp.empty:
            mx = mp["pivot_index"].to_numpy() - start_global_idx
            my = mp["pivot_price"].to_numpy()
            mh = mp["pivot_type"] == "HIGH"
            ml = mp["pivot_type"] == "LOW"

            ax.scatter(mx[mh], my[mh], marker="v", s=24, alpha=0.75, label="Medium Pivot High")
            ax.scatter(mx[ml], my[ml], marker="^", s=24, alpha=0.75, label="Medium Pivot Low")

    # Strong pivots
    if not strong_pivots.empty:
        sp = strong_pivots[strong_pivots["pivot_index"] >= start_global_idx].copy()
        if not sp.empty:
            sx = sp["pivot_index"].to_numpy() - start_global_idx
            sy = sp["pivot_price"].to_numpy()
            sh = sp["pivot_type"] == "HIGH"
            sl = sp["pivot_type"] == "LOW"

            ax.scatter(sx[sh], sy[sh], marker="v", s=70, label="Strong Pivot High")
            ax.scatter(sx[sl], sy[sl], marker="^", s=70, label="Strong Pivot Low")
            ax.plot(sx, sy, linewidth=1.4, alpha=0.9, label="Strong Pivot Structure")

    # Fib segments from strong pivots
    if not fib_segments.empty:
        segs = fib_segments.copy()
        segs = segs[segs["end_index"] >= start_global_idx]
        segs = segs.tail(last_n_segments)

        for seg in segs.itertuples(index=False):
            x0 = int(seg.start_index) - start_global_idx
            x1 = int(seg.end_index) - start_global_idx

            ax.plot([x0, x1], [seg.start_price, seg.end_price], linewidth=1.8)

            for col in segs.columns:
                if col.startswith("fib_"):
                    y = getattr(seg, col)
                    ax.hlines(y, xmin=x0, xmax=len(df_plot) - 1, linewidth=0.8, linestyles="dashed", alpha=0.45)

    xticks = np.linspace(0, len(df_plot) - 1, min(12, len(df_plot)), dtype=int)
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        [df_plot["datetime"].iloc[i].strftime("%Y-%m-%d\n%H:%M") for i in xticks],
        rotation=0
    )

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close(fig)


# ============================================================
# PIPELINE PER FILE
# ============================================================

def process_one_file(input_csv: Path, output_root: Path) -> None:
    tf_key, year = input_csv.stem.split("_")
    df = load_ohlcv(input_csv)

    medium_raw = build_raw_pivot_df(df, MEDIUM_LEFT, MEDIUM_RIGHT, "MEDIUM")
    strong_raw = build_raw_pivot_df(df, STRONG_LEFT, STRONG_RIGHT, "STRONG")

    # Keep raw medium pivots as-is
    medium_pivots = medium_raw.copy()

    # Strong pivots drive structure + fib legs
    strong_pivots = reduce_to_alternating_structure(strong_raw)
    fib_segments = build_fib_segments(strong_pivots, FIB_LEVELS)

    bar_level = build_bar_level_dataset(
        df=df,
        strong_pivots=strong_pivots,
        medium_pivots=medium_pivots,
        fib_segments=fib_segments,
        fib_levels=FIB_LEVELS,
    )

    year_tf_dir = output_root / year / tf_key
    year_tf_dir.mkdir(parents=True, exist_ok=True)

    medium_raw.to_csv(year_tf_dir / f"{tf_key}_{year}_medium_pivots.csv", index=False)
    strong_raw.to_csv(year_tf_dir / f"{tf_key}_{year}_raw_strong_pivots.csv", index=False)
    strong_pivots.to_csv(year_tf_dir / f"{tf_key}_{year}_strong_pivots.csv", index=False)
    fib_segments.to_csv(year_tf_dir / f"{tf_key}_{year}_fib_segments.csv", index=False)
    bar_level.to_csv(year_tf_dir / f"{tf_key}_{year}_barlevel_with_pivots_fibs.csv", index=False)

    if MAKE_PLOTS:
        plot_dataset_with_pivots_and_fibs(
            df=df,
            strong_pivots=strong_pivots,
            medium_pivots=medium_pivots,
            fib_segments=fib_segments,
            title=f"{tf_key.upper()} {year} Medium + Strong Pivots + Fib Segments",
            out_png=year_tf_dir / f"{tf_key}_{year}_medium_strong_pivots_fibs.png",
        )

    print(
        f"[OK] {input_csv} | rows={len(df)} | "
        f"medium_pivots={len(medium_pivots)} | "
        f"raw_strong={len(strong_raw)} | "
        f"alt_strong={len(strong_pivots)} | "
        f"fib_segments={len(fib_segments)}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    output_root = BASE_DIR / OUTPUT_DIRNAME
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("GENERATING STRONG PIVOTS, FIB SEGMENTS, BAR-LEVEL DATASETS, AND PLOTS")
    print("=" * 88)

    for year in YEAR_DIRS:
        year_dir = BASE_DIR / year
        if not year_dir.exists():
            print(f"[SKIP] Missing year directory: {year_dir}")
            continue

        for tf in TIMEFRAMES:
            input_csv = year_dir / f"{tf}_{year}.csv"
            if not input_csv.exists():
                print(f"[SKIP] Missing file: {input_csv}")
                continue

            try:
                process_one_file(input_csv, output_root)
            except Exception as e:
                print(f"[ERROR] {input_csv}: {e}")

    print("=" * 88)
    print("DONE")
    print("=" * 88)
    print(f"Exports written under: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())