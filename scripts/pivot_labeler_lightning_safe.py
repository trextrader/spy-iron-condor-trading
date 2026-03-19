# pivot_labeler_lightning_safe.py
# Lightning-safe Streamlit pivot labeler:
# - Visual price chart (candles/line) uses real datetimes (stable rendering)
# - Interaction overlay uses integer X (row index) to prevent the "single vertical line" iframe collapse
# - Box/Lasso + Click selections are captured via streamlit-plotly-events customdata indices
#
# Expected OHLC CSV headers (your files match this):
#   "Date Time", "Open", "High", "Low", "Close"
# Also supports lowercase equivalents and common aliases.

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# =========================
# Plotly events import
# =========================
try:
    from streamlit_plotly_events import plotly_events
except Exception:
    st.error("Missing dependency: streamlit-plotly-events. Install with: pip install streamlit-plotly-events")
    st.stop()

# =========================
# Page
# =========================
st.set_page_config(layout="wide", page_title="Pivot Labeler (Lightning-safe)")
st.title("🦅 Pivot Labeler — Lightning-safe (Candles + Lasso/Box Select)")

# =========================
# Sidebar
# =========================
st.sidebar.header("⚙️ Data")
input_csv = st.sidebar.text_input(
    "Input CSV Path",
    "data/spyohlc/SPY_2025_m1.csv",
)
output_csv = st.sidebar.text_input("Output Pivot CSV", "pivots_manual.csv")

st.sidebar.header("🧭 Labeling")
view_mode = st.sidebar.selectbox("View Mode", ["Candlesticks", "Lines"], index=0)
pivot_type = st.sidebar.radio("Pivot Type", ["High", "Low"], index=0)
pivot_strength = st.sidebar.slider("Pivot Strength", 1, 5, 1)

auto_commit_sweep = st.sidebar.checkbox(
    "Auto-commit sweep selections (lasso/box multi-select)",
    value=True,
)

max_bars_total = st.sidebar.slider("Max rows loaded (tail)", 1000, 500000, 50000, step=1000)
max_candles = st.sidebar.slider("Max candles rendered", 200, 20000, 4000, step=200)
max_overlay_points = st.sidebar.slider("Max overlay points (interaction)", 500, 50000, 8000, step=250)

show_debug = st.sidebar.checkbox("Show debug panel", value=False)

PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "scrollZoom": True,
    "doubleClick": "reset",
    "modeBarButtonsToAdd": ["lasso2d", "select2d"],
    "responsive": True,
}

# =========================
# Data loader (clean OHLC)
# =========================
@st.cache_data(show_spinner=False)
def load_clean_ohlc(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path, encoding="latin-1")

    # Drop junk columns like "Unnamed: 5"
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")].copy()

    # Normalize column names
    colmap = {}
    for c in df.columns:
        s = str(c).strip().lower().replace('"', "").replace(" ", "_")
        colmap[c] = s
    df = df.rename(columns=colmap)

    # Timestamp column
    if "timestamp" not in df.columns:
        if "date_time" in df.columns:
            df = df.rename(columns={"date_time": "timestamp"})
        elif "datetime" in df.columns:
            df = df.rename(columns={"datetime": "timestamp"})
        elif "date" in df.columns:
            df = df.rename(columns={"date": "timestamp"})
        else:
            return pd.DataFrame()

    # OHLC columns
    # support: open/high/low/close already lowercase, or Open/High/etc (normalized above)
    # also support o/h/l/c aliases
    alias = {
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "open_price": "open",
        "high_price": "high",
        "low_price": "low",
        "close_price": "close",
        "adj_close": "close",
        "adjclose": "close",
    }
    for c in list(df.columns):
        if c in alias and alias[c] not in df.columns:
            df = df.rename(columns={c: alias[c]})

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()

    # Make naive datetimes
    if getattr(df["timestamp"].dt, "tz", None) is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    # Coerce OHLC numeric
    for c in ["open", "high", "low", "close"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Require at least close; candles require full OHLC
    if "close" not in df.columns and not all(k in df.columns for k in ["open", "high", "low", "close"]):
        return pd.DataFrame()

    # Sort + reset
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def ensure_pivots_state(out_path: str) -> None:
    if "pivots" in st.session_state:
        return

    if os.path.exists(out_path):
        try:
            piv = pd.read_csv(out_path)
            if "timestamp" in piv.columns:
                piv["timestamp"] = pd.to_datetime(piv["timestamp"], errors="coerce")
                piv = piv.dropna(subset=["timestamp"])
            else:
                piv = pd.DataFrame(columns=["timestamp", "type", "strength", "price"])
        except Exception:
            piv = pd.DataFrame(columns=["timestamp", "type", "strength", "price"])
    else:
        piv = pd.DataFrame(columns=["timestamp", "type", "strength", "price"])

    # Normalize columns
    for col in ["timestamp", "type", "strength", "price"]:
        if col not in piv.columns:
            piv[col] = np.nan
    piv = piv[["timestamp", "type", "strength", "price"]].copy()
    st.session_state.pivots = piv


def commit_pivot(timestamp, type, strength, price) -> None:
    new_row = pd.DataFrame(
        [{
            "timestamp": pd.to_datetime(timestamp),
            "type": str(type),
            "strength": int(strength),
            "price": float(price),
        }]
    )
    st.session_state.pivots = (
        pd.concat([st.session_state.pivots, new_row], ignore_index=True)
        .drop_duplicates(subset=["timestamp", "type"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def build_price_figure(df_candle: pd.DataFrame, pivots: pd.DataFrame, mode: str) -> go.Figure:
    fig = go.Figure()

    have_ohlc = all(c in df_candle.columns for c in ["open", "high", "low", "close"])
    if mode == "Candlesticks" and not have_ohlc:
        mode = "Lines"

    if mode == "Candlesticks":
        valid = df_candle.dropna(subset=["timestamp", "open", "high", "low", "close"])
        if not valid.empty:
            fig.add_trace(go.Candlestick(
                x=valid["timestamp"],
                open=valid["open"],
                high=valid["high"],
                low=valid["low"],
                close=valid["close"],
                name="Price",
            ))
            y_min = float(valid["low"].min())
            y_max = float(valid["high"].max())
        else:
            y_min, y_max = 0.0, 1.0
    else:
        y_col = "close" if "close" in df_candle.columns else None
        valid = df_candle.dropna(subset=["timestamp", y_col]) if y_col else pd.DataFrame()
        if not valid.empty:
            fig.add_trace(go.Scatter(
                x=valid["timestamp"],
                y=valid[y_col],
                mode="lines",
                name="Price",
            ))
            y_min = float(valid[y_col].min())
            y_max = float(valid[y_col].max())
        else:
            y_min, y_max = 0.0, 1.0

    pad = max(0.01 * (y_max - y_min), 0.50)
    fig.update_layout(
        template="plotly_dark",
        height=700,
        margin=dict(t=30, b=20, l=10, r=10),
        xaxis=dict(type="date", rangeslider=dict(visible=False)),
        yaxis=dict(range=[y_min - pad, y_max + pad], autorange=False, side="right"),
        uirevision="price-ui",
        showlegend=False,
    )

    # Overlay pivots on the visual chart
    if pivots is not None and not pivots.empty:
        pv = pivots.copy()
        pv["timestamp"] = pd.to_datetime(pv["timestamp"], errors="coerce")
        pv = pv.dropna(subset=["timestamp", "price"])

        highs = pv[pv["type"] == "High"]
        lows = pv[pv["type"] == "Low"]

        if not highs.empty:
            fig.add_trace(go.Scatter(
                x=highs["timestamp"],
                y=highs["price"],
                mode="markers",
                marker=dict(symbol="triangle-down", size=11),
                name="High pivots",
            ))
        if not lows.empty:
            fig.add_trace(go.Scatter(
                x=lows["timestamp"],
                y=lows["price"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=11),
                name="Low pivots",
            ))

    return fig


def build_overlay_figure(overlay_valid: pd.DataFrame, y_col: str) -> go.Figure:
    """
    CRITICAL LIGHTNING FIX:
    - x is row_idx (0..N-1), not datetime, to prevent iframe event collapse into a vertical line
    - timestamp is shown via hovertemplate and still used for committing pivots
    """
    fig = go.Figure()

    # line for context
    fig.add_trace(go.Scatter(
        x=overlay_valid["row_idx"],
        y=overlay_valid[y_col],
        mode="lines",
        name="Overlay",
        line=dict(width=2),
        hoverinfo="skip",
    ))

    # visible points (so you can see it isn't blank)
    fig.add_trace(go.Scatter(
        x=overlay_valid["row_idx"],
        y=overlay_valid[y_col],
        mode="markers",
        name="Points",
        marker=dict(size=6, opacity=0.35),
        customdata=overlay_valid["row_idx"],
        hovertemplate="idx=%{x}<br>price=%{y}<br>ts=%{text}<extra></extra>",
        text=overlay_valid["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
    ))

    # fat invisible hitbox layer (best for lasso/box)
    fig.add_trace(go.Scatter(
        x=overlay_valid["row_idx"],
        y=overlay_valid[y_col],
        mode="markers",
        showlegend=False,
        marker=dict(color="rgba(0,0,0,0)", size=26),
        customdata=overlay_valid["row_idx"],
        hoverinfo="none",
        name="Hitbox",
    ))

    y_min = float(np.nanmin(overlay_valid[y_col].values))
    y_max = float(np.nanmax(overlay_valid[y_col].values))
    pad = max(0.01 * (y_max - y_min), 0.50)

    fig.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(t=10, b=10, l=10, r=10),
        dragmode="lasso",
        clickmode="event+select",
        xaxis=dict(title="row index (selection axis)", showgrid=True),
        yaxis=dict(range=[y_min - pad, y_max + pad], autorange=False, side="right", showgrid=True),
        uirevision="overlay-ui",
        legend=dict(orientation="h"),
    )
    return fig


def events_to_indices(events) -> list[int]:
    idx = []
    if not events:
        return idx
    for e in events:
        cd = e.get("customdata", None)
        if cd is None:
            continue
        try:
            idx.append(int(cd))
        except Exception:
            pass
    return sorted(set(idx))


# =========================
# Init state + load
# =========================
ensure_pivots_state(output_csv)

df = load_clean_ohlc(input_csv)
if df.empty:
    st.error(f"Failed to load/parse clean OHLC: {input_csv}")
    st.stop()

df = df.tail(max_bars_total).reset_index(drop=True)

df_candle = df.tail(max_candles).reset_index(drop=True)
df_overlay = df.tail(max_overlay_points).reset_index(drop=True)

if show_debug:
    st.subheader("🔍 Debug")
    st.write("Loaded rows:", len(df))
    st.write("Columns:", df.columns.tolist())
    if all(c in df.columns for c in ["open", "high", "low", "close"]):
        st.write("OHLC dtypes:", df[["open", "high", "low", "close"]].dtypes.astype(str).to_dict())
        st.write("OHLC NaNs:", df[["open", "high", "low", "close"]].isna().sum().to_dict())
    st.dataframe(df.head(10), use_container_width=True)

# =========================
# Price chart (visual only)
# =========================
fig_price = build_price_figure(df_candle, st.session_state.pivots, view_mode)
st.plotly_chart(fig_price, width="stretch")

# =========================
# Overlay (Selectable)
# =========================
st.subheader("🎯 Interaction Overlay (use Lasso or Box Select in modebar)")

if pivot_type == "High" and "high" in df_overlay.columns:
    overlay_y = "high"
elif pivot_type == "Low" and "low" in df_overlay.columns:
    overlay_y = "low"
else:
    overlay_y = "close" if "close" in df_overlay.columns else None

if overlay_y is None:
    st.error("Overlay needs at least one of: close/high/low.")
    st.stop()

overlay_valid = df_overlay.dropna(subset=["timestamp", overlay_y]).copy()
overlay_valid = overlay_valid.sort_values("timestamp").reset_index(drop=True)
overlay_valid["row_idx"] = np.arange(len(overlay_valid), dtype=int)

if overlay_valid.empty:
    st.error("Overlay is empty after filtering NaNs.")
    st.stop()

if show_debug:
    st.write("Overlay rows:", len(overlay_valid))
    st.write("Overlay time range:", overlay_valid["timestamp"].min(), "→", overlay_valid["timestamp"].max())

fig_overlay = build_overlay_figure(overlay_valid, overlay_y)

# Backward-compatible plotly_events() call (older versions don't accept config=)
try:
    events = plotly_events(
        fig_overlay,
        click_event=True,
        select_event=True,
        hover_event=False,
        key="overlay_events",
        config=PLOTLY_CONFIG,
    )
except TypeError:
    events = plotly_events(
        fig_overlay,
        click_event=True,
        select_event=True,
        hover_event=False,
        key="overlay_events",
    )

selected_idx = events_to_indices(events)_to_indices(events)

# =========================
# Candidate logic
# =========================
if selected_idx:
    if len(selected_idx) >= 2:
        sub = overlay_valid.iloc[selected_idx]
    else:
        i = selected_idx[0]
        lo = max(0, i - 5)
        hi = min(len(overlay_valid), i + 6)
        sub = overlay_valid.iloc[lo:hi]

    if not sub.empty:
        if pivot_type == "High":
            row = sub.loc[sub[overlay_y].idxmax()]
        else:
            row = sub.loc[sub[overlay_y].idxmin()]

        candidate = {
            "timestamp": row["timestamp"],
            "type": pivot_type,
            "strength": int(pivot_strength),
            "price": float(row[overlay_y]),
        }
        st.session_state["last_candidate"] = candidate

        if auto_commit_sweep and len(selected_idx) >= 2:
            commit_pivot(**candidate)
            st.success(f"🎯 Auto-committed sweep: {candidate['type']} @ {candidate['timestamp']} ({candidate['price']:.2f})")
            st.rerun()

# Sidebar candidate UI
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Candidate")

if "last_candidate" in st.session_state:
    c = st.session_state["last_candidate"]
    st.sidebar.write(f"Time: **{c['timestamp']}**")
    st.sidebar.write(f"Type: **{c['type']}** | Strength: **{c['strength']}**")
    st.sidebar.write(f"Price: **{c['price']:.2f}**")

    if st.sidebar.button("✅ Commit Candidate", use_container_width=True):
        commit_pivot(**c)
        st.sidebar.success("Committed.")
        st.rerun()
else:
    st.sidebar.info("Use lasso/box select (or click) on the overlay to create a candidate.")

# Optional debug payload
if show_debug:
    with st.expander("🔧 Selection payload (debug)", expanded=False):
        st.write("Events received:", 0 if not events else len(events))
        if events:
            st.json(events[:10])
        st.write("Selected idx:", s# =========================
# Output
# =========================
st.subheader("📋 Labeled Pivots")

st.dataframe(
    st.session_state.pivots,
    width="stretch"
)

c1, c2 = st.columns([1, 1])

with c1:
    if st.button("🗑️ Clear pivots", use_container_width=True):
        st.session_state.pivots = pd.DataFrame(
            columns=["timestamp", "type", "strength", "price"]
        )
        if "last_candidate" in st.session_state:
            del st.session_state["last_candidate"]
        st.rerun()

with c2:
    if st.button("💾 Export CSV", use_container_width=True):
        abs_path = os.path.abspath(output_csv)
        st.session_state.pivots.to_csv(abs_path, index=False)
        st.success(f"Saved to {abs_path}")

st.caption(
    "Lightning tip: do all selecting in the overlay. "
    "The main candlestick chart is visual-only to avoid iframe drag-event issues."
)
etime collapse."
)
