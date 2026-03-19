# pivot_labeler_lightning_safe.py
# Lightning-safe Streamlit pivot labeler:
# - Price chart (candles/line) uses real datetimes (stable rendering)
# - Interaction overlay uses integer X (row index) to prevent the "single vertical line" iframe collapse
# - Box/Lasso + Click selections captured via streamlit-plotly-events using customdata indices
#
# Supports your OHLC headers (case-insensitive):
#   Date Time / timestamp, Open, High, Low, Close

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
input_csv = st.sidebar.text_input("Input CSV Path", "data/spyohlc/SPY_2025_h1.csv")
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
# Helpers
# =========================
def _coerce_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace('"', "", regex=False).str.replace(",", "", regex=False).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(s, errors="coerce")

# =========================
# Data loader (clean OHLC + common aliases)
# =========================
@st.cache_data(show_spinner=False)
def load_clean_ohlc(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path, encoding="latin-1")
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
        elif "date" in df.columns:
            df = df.rename(columns={"date": "timestamp"})
        elif "datetime" in df.columns:
            df = df.rename(columns={"datetime": "timestamp"})
        else:
            return pd.DataFrame()

    # OHLC aliases
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

    # Coerce OHLC numeric (strip commas etc)
    for c in ["open", "high", "low", "close"]:
        if c in df.columns:
            df[c] = _coerce_numeric(df[c])

    # Require at least close for lines; candles require full OHLC
    have_ohlc = all(c in df.columns for c in ["open", "high", "low", "close"])
    if ("close" not in df.columns) and (not have_ohlc):
        return pd.DataFrame()

    # Sort + reset
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

# =========================
# Pivot State
# =========================
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

    for col in ["timestamp", "type", "strength", "price"]:
        if col not in piv.columns:
            piv[col] = np.nan
    st.session_state.pivots = piv[["timestamp", "type", "strength", "price"]].copy()

def commit_pivot(timestamp, type, strength, price) -> None:
    ts = pd.to_datetime(timestamp, errors="coerce")
    if pd.isna(ts):
        return

    new_row = pd.DataFrame(
        [{
            "timestamp": ts,
            "type": "High" if str(type).strip().lower().startswith("h") else "Low",
            "strength": int(strength),
            "price": float(price),
        }]
    )

    piv = st.session_state.pivots.copy()

    # Ensure schema
    for col in ["timestamp", "type", "strength", "price"]:
        if col not in piv.columns:
            piv[col] = np.nan
    piv = piv[["timestamp", "type", "strength", "price"]].copy()

    piv["timestamp"] = pd.to_datetime(piv["timestamp"], errors="coerce")
    piv["type"] = piv["type"].astype(str)
    piv["strength"] = pd.to_numeric(piv["strength"], errors="coerce").fillna(1).astype(int)
    piv["price"] = pd.to_numeric(piv["price"], errors="coerce")

    piv = pd.concat([piv, new_row], ignore_index=True)

    # Deduplicate: same timestamp + type keep last
    piv = (
        piv.dropna(subset=["timestamp", "type", "price"])
           .drop_duplicates(subset=["timestamp", "type"], keep="last")
           .sort_values(["timestamp", "type"])
           .reset_index(drop=True)
    )

    st.session_state.pivots = piv


# =========================
# Figures
# =========================
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
    x = [int(v) for v in overlay_valid["row_idx"].tolist()]
    y = [float(v) for v in overlay_valid[y_col].tolist()]
    text = overlay_valid["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    custom = x[:]  # same indices

    fig = go.Figure()

    # context line
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines",
        name="Overlay",
        line=dict(width=2),
        hoverinfo="skip",
    ))

    # visible points (selectable)
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        name="Points",
        marker=dict(size=7, opacity=0.45),
        customdata=custom,
        text=text,
        hovertemplate="idx=%{x}<br>price=%{y}<br>ts=%{text}<extra></extra>",
        selectedpoints=[],  # important for some iframe cases
    ))

    # big hitbox (selection target)
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        showlegend=False,
        marker=dict(color="rgba(0,0,0,0)", size=28),
        customdata=custom,
        hoverinfo="none",
        name="Hitbox",
        selectedpoints=[],
    ))

    y_min = float(np.nanmin(np.asarray(y, dtype=float))) if len(y) else 0.0
    y_max = float(np.nanmax(np.asarray(y, dtype=float))) if len(y) else 1.0
    pad = max(0.01 * (y_max - y_min), 0.50)

    fig.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(t=10, b=10, l=10, r=10),
        dragmode="select",                 # box select default (still can lasso from modebar)
        clickmode="event+select",
        selectdirection="any",
        xaxis=dict(title="row index (selection axis)", showgrid=True),
        yaxis=dict(range=[y_min - pad, y_max + pad], autorange=False, side="right", showgrid=True),
        uirevision="overlay-ui",
        showlegend=False,
    )
    return fig


def events_to_indices(events) -> list[int]:
    """
    streamlit-plotly-events payloads differ by environment/version.

    We accept indices from:
      - e["customdata"]              (preferred; we set it)
      - e["pointIndex"] / e["pointNumber"]
      - e["x"]                       (our x is row_idx, so x is usable)
    """
    out: list[int] = []
    if not events:
        return out

    for e in events:
        if not isinstance(e, dict):
            continue

        # 1) customdata (most reliable if present)
        cd = e.get("customdata", None)
        if cd is not None:
            try:
                out.append(int(cd))
                continue
            except Exception:
                pass

        # 2) plotly point index keys (varies)
        for k in ("pointIndex", "pointNumber", "point_index", "point_number"):
            if k in e and e[k] is not None:
                try:
                    out.append(int(e[k]))
                    break
                except Exception:
                    pass
        else:
            # 3) x (we use row_idx on x-axis)
            x = e.get("x", None)
            if x is not None:
                try:
                    out.append(int(float(x)))
                except Exception:
                    pass

    return sorted(set(out))

# =========================
# Init state + load
# =========================
ensure_pivots_state(output_csv)

df = load_clean_ohlc(input_csv)
if df.empty:
    st.error(f"Failed to load/parse OHLC: {input_csv}")
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
    st.dataframe(df.head(10), width="stretch")

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
    st.write("Overlay y_col:", overlay_y)

fig_overlay = build_overlay_figure(overlay_valid, overlay_y)

# Backward-compatible plotly_events() (some versions don't accept config=)
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

selected_idx = events_to_indices(events)

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
            st.success(
                f"🎯 Auto-committed sweep: {candidate['type']} @ {candidate['timestamp']} ({candidate['price']:.2f})"
            )
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
        st.write("Selected idx:", selected_idx[:50])

# =========================
# Output
# =========================
st.subheader("📋 Labeled Pivots")
st.dataframe(st.session_state.pivots, width="stretch")

c1, c2 = st.columns([1, 1])
with c1:
    if st.button("🗑️ Clear pivots", use_container_width=True):
        st.session_state.pivots = pd.DataFrame(columns=["timestamp", "type", "strength", "price"])
        if "last_candidate" in st.session_state:
            del st.session_state["last_candidate"]
        st.rerun()

with c2:
    if st.button("💾 Export CSV", use_container_width=True):
        abs_path = os.path.abspath(output_csv)
        piv = st.session_state.pivots.copy()
        piv["timestamp"] = pd.to_datetime(piv["timestamp"], errors="coerce")
        piv = piv.dropna(subset=["timestamp", "type", "price"]).sort_values("timestamp").reset_index(drop=True)
        piv.to_csv(abs_path, index=False)
        st.success(f"Saved to {abs_path}")


st.caption(
    "Lightning tip: do all selecting in the overlay. "
    "The main candlestick chart is visual-only to avoid iframe drag-event issues."
)
