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
render_mode = st.sidebar.radio("Rendering Engine", ["Interactive (Labeling)", "Standard (Fast/Preview)"])
limit_bars = st.sidebar.checkbox("Focus: Last 500 bars only", value=False)

if st.sidebar.button("🧹 Clear App Cache"):
    st.cache_data.clear()
    st.rerun()

# TF Mapping
tf_map = {"M1": "1min", "M5": "5min", "M15": "15min", "H1": "1H"}

if os.path.exists(input_csv):
    st.sidebar.info(f"📁 Dataset: {os.path.basename(input_csv)}")
    load_btn = st.sidebar.button("🚀 Load / Refresh Data")
    
    @st.cache_data
    def load_data(path, tf_str, fmt_choice):
        print(f"🎬 Loading dataset: {path} ({tf_str}, {fmt_choice})...")
        # 0. Detect Format (using latin-1 to avoid trademark decode errors)
        with open(path, 'r', encoding='latin-1') as f:
            first_line = f.readline()
        
        is_barchart = "Symbol:" in first_line
        
        if fmt_choice == "Barchart Raw Spot" or (fmt_choice == "Auto-Detect" and is_barchart):
            print("Detected Barchart format. Skipping metadata header...")
            df = pd.read_csv(path, skiprows=1, encoding='latin-1') 
            # Aggressive column cleaning: strip quotes and whitespace
            df.columns = [str(c).replace('"', '').replace("'", "").strip() for c in df.columns]
            
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
            print("Assuming standard CondorNet V4.2 / options format...")
            df = pd.read_csv(path, encoding='latin-1')
            df.columns = [str(c).replace('"', '').replace("'", "").strip() for c in df.columns]
        
        # 1. Parse timestamps robustly
        print("Cleaning and parsing timestamps...")
        if df['timestamp'].dtype == object:
            df['timestamp'] = df['timestamp'].astype(str).str.replace('"', '').str.replace("'", "").str.strip()
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        
        # 1.5 Clean Numeric Columns (strip quotes and commas)
        print("Cleaning numeric columns...")
        num_cols = ['open', 'high', 'low', 'close', 'underlying_price', 'iv', 'rev_m5', 'rev_m15', 'rev_h1']
        for col in num_cols:
            if col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace('"', '').str.replace("'", "").str.replace(',', '').str.strip()
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 2. Extract spot-level columns
        spot_cols = [
            'open', 'high', 'low', 'close', 'underlying_price', 
            'rev_m5', 'rev_m15', 'rev_h1', 'iv',
            'frama_study', 'avwap_study', 'mcclellan_osc'
        ]
        available_spot_cols = [c for c in spot_cols if c in df.columns]
        
        # 3. Collapse multiple rows per timestamp (for Option files)
        print("Checking for duplicate bars...")
        if df['timestamp'].duplicated().any():
            spots = df.groupby('timestamp')[available_spot_cols].first().reset_index()
        else:
            spots = df[['timestamp'] + available_spot_cols].copy()
        
        # 4. Resample if requested timeframe is higher than M1
        if tf_str != "1min":
            print(f"Resampling to {tf_str}...")
            agg_dict = {}
            if 'open' in spots.columns: agg_dict['open'] = 'first'
            if 'high' in spots.columns: agg_dict['high'] = 'max'
            if 'low' in spots.columns: agg_dict['low'] = 'min'
            if 'close' in spots.columns: agg_dict['close'] = 'last'
            if 'underlying_price' in spots.columns: agg_dict['underlying_price'] = 'last'
            
            for c in ['rev_m5', 'rev_m15', 'rev_h1', 'iv']:
                if c in spots.columns: agg_dict[c] = 'mean'
            
            if agg_dict:
                spots = spots.set_index('timestamp').resample(tf_str).agg(agg_dict).dropna(subset=['underlying_price']).reset_index()
        
        # 5. Final cleanup
        if spots['timestamp'].dt.tz is not None:
            spots['timestamp'] = spots['timestamp'].dt.tz_localize(None)
            
        # 6. Aggressive OHLC check - Candlesticks NEED all 4 values to render
        ohlc_cols = [c for c in ['open', 'high', 'low', 'close'] if c in spots.columns]
        if ohlc_cols:
            before_drop = len(spots)
            spots = spots.dropna(subset=ohlc_cols)
            # FORCE FLOAT64
            for col in ohlc_cols:
                spots[col] = spots[col].astype(float)
            if len(spots) < before_drop:
                print(f"⚠️ Dropped {before_drop - len(spots)} bars with missing OHLC data.")
        
        print(f"✅ Data ready. Shape: {spots.shape}")
        return spots.sort_values('timestamp')

    data = pd.DataFrame()
    if load_btn or 'data_loaded' in st.session_state:
        st.session_state.data_loaded = True
        with st.spinner("📦 Reading and processing CSV... check terminal for progress logs."):
            data = load_data(input_csv, tf_map[timeframe], data_type)
    else:
        st.warning("Click 'Load Data' in the sidebar to begin.")
        st.stop()
    
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
            
        # Price Diagnostics
        if 'close' in data.columns:
            p_min = data['close'].min()
            p_max = data['close'].max()
            st.sidebar.write(f"Price Range: **${p_min:.2f} - ${p_max:.2f}**")
            if pd.isna(p_min):
                st.sidebar.error("❌ Price data is all NaN!")
    # ---------------------------
    
    # Session selector (optional filtering)
    date_range = st.sidebar.date_input("Date Range", [data['timestamp'].min().date(), data['timestamp'].max().date()])
    chart_style = st.sidebar.selectbox("Chart Style", ["Candlestick", "Line"])
    
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_data = data[(data['timestamp'].dt.date >= start_date) & (data['timestamp'].dt.date <= end_date)].copy()
    elif isinstance(date_range, (pd.Timestamp, pd.DatetimeIndex)):
        filtered_data = data.copy() # Fallback for single clicks
    else:
        filtered_data = data.copy()
    
    if limit_bars:
        filtered_data = filtered_data.sort_values('timestamp').tail(500)

    st.sidebar.write(f"Chart bars: {len(filtered_data)}")
    
    if filtered_data.empty:
        st.sidebar.error("⚠️ Filtered data is empty! Chart will be blank.")
        st.write("### 🔍 Debug: Filtered Data is Empty")
        st.write("Source data first 5 timestamps:")
        st.write(data['timestamp'].head())
    else:
        with st.expander("📝 View Chart Data Preview (verify columns)"):
            st.dataframe(filtered_data.head(10))

    # Manual labels state
    if 'pivots' not in st.session_state:
        if os.path.exists(output_csv):
            try:
                st.session_state.pivots = pd.read_csv(output_csv, encoding='latin-1')
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
    # Choose Trace Type (Candlesticks are heavy for interactive events)
    if render_mode == "Interactive (Labeling)":
        # USE ULTRA-LIGHTWEIGHT SCATTER FOR INTERACTIVE MODE
        # This fixes the 'blank sidebar / no click' issue on large datasets
        fig.add_trace(go.Scatter(
            x=filtered_data['timestamp'].tolist(),
            y=filtered_data['close'].tolist(),
            mode='lines',
            line=dict(color='white', width=1),
            name='Interactive Price'
        ))
    elif chart_style == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=filtered_data['timestamp'].tolist(),
            open=filtered_data['open'],
            high=filtered_data['high'],
            low=filtered_data['low'],
            close=filtered_data['close'],
            name='Candlesticks',
            increasing_line_color='green', 
            decreasing_line_color='red'
        ))
        # Add a faint line trace as a backup/baseline
        fig.add_trace(go.Scatter(
            x=filtered_data['timestamp'].tolist(), y=filtered_data['close'],
            mode='lines', line=dict(color='gray', width=0.5),
            name='Price Baseline (Faint)',
            opacity=0.3
        ))
    else:
        fig.add_trace(go.Scatter(
            x=filtered_data['timestamp'].tolist(),
            y=filtered_data['close'],
            name='Price (Line)',
            line=dict(color='white', width=1.5)
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
                    x=h['timestamp'].tolist(), y=h['price'].tolist(), mode='markers',
                    marker=dict(symbol='triangle-down', color='red', size=12 if tf == timeframe else 8, 
                                opacity=1.0 if tf == timeframe else 0.4),
                    name=f'{tf} Highs'
                ))
            
            # Lows
            l = tf_data[tf_data['type'] == 'Low']
            if not l.empty:
                fig.add_trace(go.Scatter(
                    x=l['timestamp'].tolist(), y=l['price'].tolist(), mode='markers',
                    marker=dict(symbol='triangle-up', color='green', size=12 if tf == timeframe else 8,
                                opacity=1.0 if tf == timeframe else 0.4),
                    name=f'{tf} Lows'
                ))

    # Explicitly calculate bounds for the current view
    y_min = filtered_data['low'].dropna().min() if 'low' in filtered_data.columns else filtered_data['underlying_price'].dropna().min()
    y_max = filtered_data['high'].dropna().max() if 'high' in filtered_data.columns else filtered_data['underlying_price'].dropna().max()

    # Convert ranges to ISO strings for Plotly stability
    x_range = [filtered_data['timestamp'].min().isoformat(), filtered_data['timestamp'].max().isoformat()] if not filtered_data.empty else None

    fig.update_layout(
        height=800, 
        template="plotly_dark", 
        xaxis=dict(
            rangeslider=dict(visible=(render_mode != "Interactive (Labeling)")),
            type='date',
            range=x_range
        ),
        yaxis=dict(
            fixedrange=False,
            title="Price ($)",
            side="right",
            range=[float(y_min) * 0.999, float(y_max) * 1.001] if not pd.isna(y_min) else None
        ),
        clickmode='event+select',
        dragmode='zoom'
    )

    # Display chart and capture clicks
    if render_mode == "Interactive (Labeling)":
        from streamlit_plotly_events import plotly_events
        # Force a slightly smaller subset if interactive to prevent hangs
        if len(filtered_data) > 2000:
            st.warning(f"⚠️ Displaying {len(filtered_data)} bars. Clicks may be slow.")
        selected_points = plotly_events(fig, click_event=True, hover_event=False, override_height=700)
    else:
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        selected_points = []

    if selected_points:
        point = selected_points[0]
        idx = point.get('pointIndex')
        curve = point.get('curveNumber')
        
        # DEBUG: Show click data in sidebar
        st.sidebar.markdown("---")
        st.sidebar.subheader("🖱️ DEBUG: Last Click")
        st.sidebar.json(point)
        
        # If we clicked a bar (usually curve 0 or low indices)
        if idx is not None and idx < len(filtered_data):
            row = filtered_data.iloc[idx]
            ts = row['timestamp']
            
            # SNAP TO PRICE: Highs snap to candle high, Lows snap to candle low
            if pivot_type == "High":
                price = row['high'] if pd.notna(row.get('high')) else row['underlying_price']
            else:
                price = row['low'] if pd.notna(row.get('low')) else row['underlying_price']
            
            # Add pivot
            new_p = pd.DataFrame([{
                'timestamp': ts,
                'type': pivot_type,
                'strength': pivot_strength,
                'price': price,
                'timeframe': timeframe
            }])
            
            # Lenient check for duplicates (just to get it working)
            already_exists = False
            if not st.session_state.pivots.empty:
                # Compare as strings to be safe with types
                ts_str = str(ts)
                existing_ts = st.session_state.pivots['timestamp'].astype(str).tolist()
                existing_types = st.session_state.pivots['type'].tolist()
                existing_tfs = st.session_state.pivots['timeframe'].tolist()
                
                for i in range(len(existing_ts)):
                    if existing_ts[i] == ts_str and existing_types[i] == pivot_type and existing_tfs[i] == timeframe:
                        already_exists = True
                        break
            
            if not already_exists:
                st.session_state.pivots = pd.concat([st.session_state.pivots, new_p], ignore_index=True)
                st.sidebar.success(f"✅ Marked {pivot_type} at {ts}")
                st.rerun()
            else:
                st.sidebar.info(f"💡 Marker already exists here.")
        else:
            st.sidebar.warning(f"Click missed the data bars. Try clicking precisely on the candle.")

    # Table & Export
    st.subheader("Labeled Pivots")
    st.dataframe(st.session_state.pivots)
    
    if st.button("💾 Export to CSV"):
        st.session_state.pivots.to_csv(output_csv, index=False)
        st.success(f"Pivots exported to {output_csv}")

else:
    st.error(f"File not found: {input_csv}. Please check the sidebar path.")
