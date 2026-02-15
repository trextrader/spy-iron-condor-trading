import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import argparse

# Streamlit App
st.set_page_config(layout="wide", page_title="CondorNet v4.2 Pivot Labeler")

st.title("🦅 CondorNet v4.2 Pivot Labeler")
st.markdown("""
**Instructions:**
1. Select a dataset (M1 or M5 recommended).
2. Click on the chart to mark **Local Highs** or **Local Lows**.
3. Toggle the pivot type using the sidebar.
4. Export the resulting CSV for training.
""")

# Sidebar settings
st.sidebar.header("Settings")
input_csv = st.sidebar.text_input("Input CSV Path", "data/Datasetv3/SPY_Barchart_Interactive_Chart_Range_1m_02_09_2026.csv")
output_csv = st.sidebar.text_input("Export filename", "pivots_manual.csv")
data_type = st.sidebar.selectbox("Data Format", ["Auto-Detect", "CondorNet V4.2", "Barchart Raw Spot"])
timeframe = st.sidebar.selectbox("View Timeframe", ["M1", "M5", "M15", "H1"], index=0)
pivot_type = st.sidebar.radio("Pivot Type to Mark", ["High", "Low"])
pivot_strength = st.sidebar.slider("Pivot Strength", 1, 3, 1)

# TF Mapping
tf_map = {"M1": "1min", "M5": "5min", "M15": "15min", "H1": "1H"}

if os.path.exists(input_csv):
    st.sidebar.info(f"📁 Dataset: {os.path.basename(input_csv)}")
    
    @st.cache_data
    def load_data(path, tf_str, fmt_choice):
        # 0. Detect Format
        with open(path, 'r') as f:
            first_line = f.readline()
        
        is_barchart = "Symbol:" in first_line
        
        if fmt_choice == "Barchart Raw Spot" or (fmt_choice == "Auto-Detect" and is_barchart):
            # Skip 1st metadata row, use 2nd as headers
            df = pd.read_csv(path, skiprows=1) 
            
            # Standardize Barchart columns (including some studies)
            rename_map = {
                'Date Time': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Implied Volatility': 'iv',
                'FRAMA': 'frama_study',
                'Anchored VWAP': 'avwap_study',
                'McClellanOsc': 'mcclellan_osc'
            }
            df = df.rename(columns=rename_map)
            
            # Ensure underlying_price points to close for labeling
            if 'underlying_price' not in df.columns:
                df['underlying_price'] = df['close']
        else:
            df = pd.read_csv(path)
        
        # 1. Parse timestamps robustly
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        
        # 2. Extract spot-level columns
        spot_cols = [
            'open', 'high', 'low', 'close', 'underlying_price', 
            'rev_m5', 'rev_m15', 'rev_h1', 'iv',
            'frama_study', 'avwap_study', 'mcclellan_osc'
        ]
        available_spot_cols = [c for c in spot_cols if c in df.columns]
        
        # 3. Collapse multiple rows per timestamp (for Option files)
        # For spot files, this just confirms 1 row per min.
        spots = df.groupby('timestamp')[available_spot_cols].first().reset_index()
        
        # 4. Resample if requested timeframe is higher than M1
        if tf_str != "1min":
            agg_dict = {}
            if 'open' in spots.columns: agg_dict['open'] = 'first'
            if 'high' in spots.columns: agg_dict['high'] = 'max'
            if 'low' in spots.columns: agg_dict['low'] = 'min'
            if 'close' in spots.columns: agg_dict['close'] = 'last'
            if 'underlying_price' in spots.columns: agg_dict['underlying_price'] = 'last'
            
            # Weighted/averaged signals
            for c in ['rev_m5', 'rev_m15', 'rev_h1']:
                if c in spots.columns: agg_dict[c] = 'mean'
            
            if agg_dict:
                spots = spots.set_index('timestamp').resample(tf_str).agg(agg_dict).dropna(subset=['underlying_price']).reset_index()
        
        # 5. Final cleanup
        if spots['timestamp'].dt.tz is not None:
            spots['timestamp'] = spots['timestamp'].dt.tz_localize(None)
            
        return spots.sort_values('timestamp')

    data = load_data(input_csv, tf_map[timeframe], data_type)
    
    # --- DIAGNOSTICS SIDEBAR ---
    st.sidebar.subheader("📊 Data Stats")
    st.sidebar.write(f"Bars: {len(data)}")
    if not data.empty:
        st.sidebar.write("First bar timestamp Sample:")
        st.sidebar.code(str(data['timestamp'].iloc[0]))
        st.sidebar.write("Last bar timestamp Sample:")
        st.sidebar.code(str(data['timestamp'].iloc[-1]))
        
        # Check if we have H/M/S data
        has_time = (data['timestamp'].dt.hour.sum() + data['timestamp'].dt.minute.sum() > 0)
        if not has_time:
            st.sidebar.warning("⚠️ No intraday time detected! (Daily?)")
        else:
            st.sidebar.success("✅ Intraday data detected.")
    # ---------------------------
    
    # Session selector (optional filtering)
    date_range = st.sidebar.date_input("Date Range", [data['timestamp'].min(), data['timestamp'].max()])
    filtered_data = data[(data['timestamp'].dt.date >= date_range[0]) & (data['timestamp'].dt.date <= date_range[1])]

    # Manual labels state
    if 'pivots' not in st.session_state:
        if os.path.exists(output_csv):
            try:
                st.session_state.pivots = pd.read_csv(output_csv)
                # Ensure timeframe column exists for legacy compatibility
                if 'timeframe' not in st.session_state.pivots.columns:
                    st.session_state.pivots['timeframe'] = 'M1'
                st.sidebar.success(f"Loaded {len(st.session_state.pivots)} existing pivots.")
            except Exception as e:
                st.sidebar.error(f"Error loading existing pivots: {e}")
                st.session_state.pivots = pd.DataFrame(columns=['timestamp', 'type', 'strength', 'price', 'timeframe'])
        else:
            st.session_state.pivots = pd.DataFrame(columns=['timestamp', 'type', 'strength', 'price', 'timeframe'])

    # SCHEMA ENFORCEMENT: Ensure 'timeframe' exists if session persisted from older version
    if 'timeframe' not in st.session_state.pivots.columns:
        st.session_state.pivots['timeframe'] = 'M1'

    # Chart
    fig = make_subplots(rows=1, cols=1)
    
    # Candlestick Chart
    fig.add_trace(go.Candlestick(
        x=filtered_data['timestamp'],
        open=filtered_data['open'] if 'open' in filtered_data.columns else filtered_data['underlying_price'],
        high=filtered_data['high'] if 'high' in filtered_data.columns else filtered_data['underlying_price'],
        low=filtered_data['low'] if 'low' in filtered_data.columns else filtered_data['underlying_price'],
        close=filtered_data['close'] if 'close' in filtered_data.columns else filtered_data['underlying_price'],
        name='Market Price'
    ))
    
    # Overlays (Rev Signals)
    if 'rev_m5' in filtered_data.columns:
        fig.add_trace(go.Scatter(
            x=filtered_data['timestamp'], 
            y=filtered_data['underlying_price'] + filtered_data['rev_m5']*0.5, 
            name='Rev M5 Overlay',
            line=dict(color='cyan', dash='dot', width=0.5),
            visible='legendonly'
        ))

    # Existing Pivots
        # Display all pivots but highlight current timeframe
        for tf in ["M1", "M5", "M15", "H1"]:
            tf_data = st.session_state.pivots[st.session_state.pivots['timeframe'] == tf]
            if tf_data.empty: continue
            
            # Highs
            h = tf_data[tf_data['type'] == 'High']
            if not h.empty:
                fig.add_trace(go.Scatter(
                    x=h['timestamp'], y=h['price'], mode='markers',
                    marker=dict(symbol='triangle-down', color='red', size=12 if tf == timeframe else 8, 
                                opacity=1.0 if tf == timeframe else 0.4),
                    name=f'{tf} Highs'
                ))
            
            # Lows
            l = tf_data[tf_data['type'] == 'Low']
            if not l.empty:
                fig.add_trace(go.Scatter(
                    x=l['timestamp'], y=l['price'], mode='markers',
                    marker=dict(symbol='triangle-up', color='green', size=12 if tf == timeframe else 8,
                                opacity=1.0 if tf == timeframe else 0.4),
                    name=f'{tf} Lows'
                ))

    fig.update_layout(
        height=800, 
        template="plotly_dark", 
        xaxis=dict(
            rangeslider=dict(visible=True),
            type='date'
        ),
        yaxis=dict(
            fixedrange=False,
            title="Price ($)",
            side="right"
        ),
        clickmode='event+select',
        dragmode='zoom'  # Allows box zoom by default
    )

    # Display chart and capture clicks
    from streamlit_plotly_events import plotly_events
    selected_points = plotly_events(fig, click_event=True, hover_event=False, override_height=700)

    if selected_points:
        point = selected_points[0]
        ts = filtered_data.iloc[point['pointIndex']]['timestamp']
        price = filtered_data.iloc[point['pointIndex']]['underlying_price']
        
        # Add pivot
        new_pivot = pd.DataFrame({
            'timestamp': [ts],
            'type': [pivot_type],
            'strength': [pivot_strength],
            'price': [price],
            'timeframe': [timeframe]
        })
        
        # Check if already exists
        if ts not in st.session_state.pivots['timestamp'].values:
            st.session_state.pivots = pd.concat([st.session_state.pivots, new_pivot], ignore_index=True)
            st.rerun()

    # Table & Export
    st.subheader("Labeled Pivots")
    st.dataframe(st.session_state.pivots)
    
    if st.button("💾 Export to CSV"):
        st.session_state.pivots.to_csv(output_csv, index=False)
        st.success(f"Pivots exported to {output_csv}")

else:
    st.error(f"File not found: {input_csv}. Please check the sidebar path.")
