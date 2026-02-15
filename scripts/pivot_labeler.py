import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import calendar

# Streamlit App
st.set_page_config(layout="wide", page_title="CondorNet v4.2 Pivot Labeler")

st.title("🦅 CondorNet v4.2 Pivot Labeler")
st.markdown("""
**Instructions:**
1. Select a dataset and click **Load Data**.
2. Use the **Date Range** to narrow focus to < 5,000 bars for labeling.
3. Click on the chart to mark pivots.
4. Export the CSV when finished.
""")

# --- SIDEBAR: STATIC SETTINGS ---
st.sidebar.header("⚙️ Settings")
input_csv = st.sidebar.text_input("Input CSV Path", "data/Datasetv3/SPY_Barchart_Interactive_Chart_Range_1m_02_09_2026.csv")
output_csv = st.sidebar.text_input("Export Content Path", "pivots_manual.csv")
data_type = st.sidebar.selectbox("Data Format", ["Auto-Detect", "CondorNet V4.2", "Barchart Raw Spot"])
timeframe = st.sidebar.selectbox("View Timeframe", ["M1", "M5", "M15", "H1"], index=0)
pivot_type = st.sidebar.radio("Pivot Type to Mark", ["High", "Low"])
pivot_strength = st.sidebar.slider("Pivot Strength", 1, 3, 1)

st.sidebar.markdown("---")
view_mode = st.sidebar.selectbox("View Mode", ["Lines", "Candlesticks", "OHLC Bars"], index=0)
render_mode = st.sidebar.radio("Interaction Mode", ["Interactive (Labeling)", "Standard (Visual Only)"], index=1)
limit_bars = st.sidebar.checkbox("Focus: Last 1000 bars only", value=False)
auto_resample = st.sidebar.checkbox("Auto-Resample (Source must be M1)", value=False)

if st.sidebar.button("🧹 Clear App Cache"):
    st.cache_data.clear()
    st.rerun()

# TF Mapping
tf_map = {"M1": "1min", "M5": "5min", "M15": "15min", "H1": "1H"}

# --- DATA LOADING FUNCTION ---
@st.cache_data
def load_data(path, tf_str, fmt_choice, do_resample):
    if not os.path.exists(path):
        return pd.DataFrame()
        
    print(f"🎬 Loading dataset: {path} (TF: {tf_str if do_resample else 'Native'})...")
    
    # 0. Detect Format (using latin-1 to avoid trademark decode errors)
    with open(path, 'r', encoding='latin-1') as f:
        first_line = f.readline()
    
    is_barchart = "Symbol:" in first_line
    
    if fmt_choice == "Barchart Raw Spot" or (fmt_choice == "Auto-Detect" and is_barchart):
        df = pd.read_csv(path, skiprows=1, encoding='latin-1') 
    else:
        df = pd.read_csv(path, encoding='latin-1')

    # Aggressive Column Cleaning
    df.columns = [str(c).replace('"', '').replace("'", "").strip() for c in df.columns]

    # Robust Column Mapping
    col_map = {}
    for c in df.columns:
        c_low = c.lower()
        if 'date' in c_low or 'time' in c_low or 'timestamp' == c_low: col_map[c] = 'timestamp'
        elif 'open' == c_low: col_map[c] = 'open'
        elif 'high' == c_low: col_map[c] = 'high'
        elif 'low' == c_low: col_map[c] = 'low'
        elif 'close' == c_low or 'last' == c_low: col_map[c] = 'close'
        elif 'vol' in c_low and 'iv' in c_low: col_map[c] = 'iv'
        elif 'underlying' in c_low: col_map[c] = 'underlying_price'
    
    df = df.rename(columns=col_map)
    if 'underlying_price' not in df.columns and 'close' in df.columns:
        df['underlying_price'] = df['close']

    if 'timestamp' not in df.columns:
        return pd.DataFrame()

    # Parse timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    
    # Clean Numerics
    num_cols = ['open', 'high', 'low', 'close', 'underlying_price', 'iv']
    for col in num_cols:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace('"', '').str.replace(',', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Collapse duplicates
    spot_cols = ['open', 'high', 'low', 'close', 'underlying_price', 'iv']
    available = [c for c in spot_cols if c in df.columns]
    spots = df.groupby('timestamp')[available].first().reset_index() if df['timestamp'].duplicated().any() else df[['timestamp']+available].copy()
    
    # Resample
    if do_resample and tf_str != "1min":
        agg_dict = {c: 'mean' for c in available}
        if 'open' in available: agg_dict['open'] = 'first'
        if 'high' in available: agg_dict['high'] = 'max'
        if 'low' in available: agg_dict['low'] = 'min'
        if 'close' in available: agg_dict['close'] = 'last'
        if 'underlying_price' in available: agg_dict['underlying_price'] = 'last'
        spots = spots.set_index('timestamp').resample(tf_str).agg(agg_dict).dropna(subset=['underlying_price']).reset_index()
    
    if spots['timestamp'].dt.tz is not None:
        spots['timestamp'] = spots['timestamp'].dt.tz_localize(None)

    # Force float for OHLC
    ohlc = [c for c in ['open', 'high', 'low', 'close'] if c in spots.columns]
    for c in ohlc: spots[c] = spots[c].astype(float)
    
    return spots.sort_values('timestamp')

# --- MAIN APP FLOW ---
if os.path.exists(input_csv):
    st.sidebar.info(f"📁 Dataset Found: {os.path.basename(input_csv)}")
    load_btn = st.sidebar.button("🚀 Load / Refresh Data")
    
    if load_btn or 'data_loaded' in st.session_state:
        st.session_state.data_loaded = True
        data = load_data(input_csv, tf_map[timeframe], data_type, auto_resample)
        
        if data.empty:
            st.error("❌ Data failed to load or has no timestamp column.")
            st.stop()
            
        # --- SIDEBAR: DYNAMIC NAVIGATION ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("📅 Navigation")
        min_d, max_d = data['timestamp'].min().date(), data['timestamp'].max().date()
        date_range = st.sidebar.date_input("Select Date Range", [min_d, max_d])
        
        # Filtering
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            s_d, e_d = date_range
            filtered = data[(data['timestamp'].dt.date >= s_d) & (data['timestamp'].dt.date <= e_d)].copy()
        else:
            filtered = data.copy()
            
        if limit_bars:
            filtered = filtered.tail(1000)
            
        st.sidebar.write(f"Chart bars: **{len(filtered)}**")
        
        # Stats
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Stats")
        diff = (data['timestamp'].iloc[1] - data['timestamp'].iloc[0]).total_seconds() / 60 if len(data) > 1 else 0
        st.sidebar.write(f"Interval: **{int(diff)} min**")
        if 'close' in data.columns:
            st.sidebar.write(f"Price: **${data['close'].min():.2f}-${data['close'].max():.2f}**")

        if st.sidebar.button("🗑️ Clear All Pivots"):
            st.session_state.pivots = pd.DataFrame(columns=['timestamp', 'type', 'strength', 'price', 'timeframe'])
            st.rerun()

        # --- DATA PREVIEW ---
        with st.expander("📝 Data Preview"):
            st.dataframe(filtered.head(10))

        if 'pivots' not in st.session_state:
            if os.path.exists(output_csv):
                try:
                    df_piv = pd.read_csv(output_csv)
                    if 'timestamp' in df_piv.columns:
                        df_piv['timestamp'] = pd.to_datetime(df_piv['timestamp'], errors='coerce')
                    st.session_state.pivots = df_piv.dropna(subset=['timestamp'])
                except:
                    st.session_state.pivots = pd.DataFrame(columns=['timestamp', 'type', 'strength', 'price', 'timeframe'])
            else:
                st.session_state.pivots = pd.DataFrame(columns=['timestamp', 'type', 'strength', 'price', 'timeframe'])

        # --- CHARTING ---
        fig = make_subplots(rows=1, cols=1)
        
        if view_mode == "Candlesticks":
            fig.add_trace(go.Candlestick(
                x=filtered['timestamp'].tolist(), 
                open=filtered['open'].tolist(), 
                high=filtered['high'].tolist(), 
                low=filtered['low'].tolist(), 
                close=filtered['close'].tolist(), 
                name='Price'
            ))
        elif view_mode == "OHLC Bars":
            fig.add_trace(go.Ohlc(
                x=filtered['timestamp'].tolist(), 
                open=filtered['open'].tolist(), 
                high=filtered['high'].tolist(), 
                low=filtered['low'].tolist(), 
                close=filtered['close'].tolist(), 
                name='Price'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=filtered['timestamp'].tolist(), 
                y=filtered['close'].tolist(), 
                mode='lines', 
                line=dict(color='white', width=1), 
                name='Price'
            ))

        # Add existing pivots
        for tf_p in ["M1", "M5", "M15", "H1"]:
            p = st.session_state.pivots[st.session_state.pivots['timeframe'] == tf_p]
            if not p.empty:
                highs = p[p['type'] == 'High']
                lows = p[p['type'] == 'Low']
                fig.add_trace(go.Scatter(x=highs['timestamp'].tolist(), y=highs['price'].tolist(), mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name=f'{tf_p} Highs'))
                fig.add_trace(go.Scatter(x=lows['timestamp'].tolist(), y=lows['price'].tolist(), mode='markers', marker=dict(symbol='triangle-up', color='green', size=10), name=f'{tf_p} Lows'))

        fig.update_layout(
            height=800, 
            template="plotly_dark", 
            title=f"SPY {timeframe} - {len(filtered)} bars", 
            yaxis=dict(side="right"),
            xaxis=dict(type='date', rangeslider=dict(visible=False))
        )
        
        if render_mode == "Interactive (Labeling)":
            if len(filtered) > 5000:
                st.error("❌ **Too many bars for Interactive Mode!** Use Date Range to select < 5,000 bars.")
            else:
                from streamlit_plotly_events import plotly_events
                event_key = f"ev_{len(filtered)}_{timeframe}_{view_mode}"
                selected = plotly_events(fig, click_event=True, key=event_key)
                
                if selected:
                    p = selected[0]
                    idx = p.get('pointIndex')
                    if idx is not None and idx < len(filtered):
                        row = filtered.iloc[idx]
                        price = row['high'] if pivot_type == "High" else row['low']
                        new_p = pd.DataFrame([{'timestamp': row['timestamp'], 'type': pivot_type, 'strength': pivot_strength, 'price': price, 'timeframe': timeframe}])
                        st.session_state.pivots = pd.concat([st.session_state.pivots, new_p], ignore_index=True)
                        st.rerun()
        else:
            st.plotly_chart(fig, use_container_width=True)

        # --- EXPORT ---
        st.subheader("Labeled Pivots")
        st.dataframe(st.session_state.pivots)
        if st.button("💾 Export CSV"):
            st.session_state.pivots.to_csv(output_csv, index=False)
            st.success(f"Saved to {output_csv}")

    else:
        st.info("💡 **Ready to Start?** Click **'Load / Refresh Data'** in the sidebar.")
else:
    st.error(f"❌ File not found: {input_csv}")
