import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import argparse
import calendar

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
if st.sidebar.button("🧹 Clear App Cache"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
view_mode = st.sidebar.selectbox("View Mode", ["Lines", "Candlesticks", "OHLC Bars"], index=0)
render_mode = st.sidebar.radio("Interaction Mode", ["Interactive (Labeling)", "Standard (Visual Only)"], index=1)
limit_bars = st.sidebar.checkbox("Focus: Last 1000 bars only", value=False)

# TF Mapping
tf_map = {"M1": "1min", "M5": "5min", "M15": "15min", "H1": "1H"}

if os.path.exists(input_csv):
    st.sidebar.info(f"📁 Dataset: {os.path.basename(input_csv)}")
    load_btn = st.sidebar.button("🚀 Load / Refresh Data")
    
    @st.cache_data
    def load_data(path, tf_str, fmt_choice, do_resample):
        print(f"🎬 Loading dataset: {path} (TF: {tf_str if do_resample else 'Native'}, Format: {fmt_choice})...")
        # 0. Detect Format (using latin-1 to avoid trademark decode errors)
        with open(path, 'r', encoding='latin-1') as f:
            first_line = f.readline()
        
        is_barchart = "Symbol:" in first_line
        
        if fmt_choice == "Barchart Raw Spot" or (fmt_choice == "Auto-Detect" and is_barchart):
            print("Detected Barchart format. Skipping metadata header...")
            df = pd.read_csv(path, skiprows=1, encoding='latin-1') 
        else:
            print("Assuming standard format...")
            df = pd.read_csv(path, encoding='latin-1')
        # 0.5 Aggressive Column Cleaning: strip quotes and whitespace
        df.columns = [str(c).replace('"', '').replace("'", "").strip() for c in df.columns]
        print(f"Loaded columns: {list(df.columns)}")

        # 0.6 Robust Column Mapping (Search for keywords instead of exact matches)
        col_map = {}
        for c in df.columns:
            c_low = c.lower()
            if 'date' in c_low or 'time' in c_low or 'timestamp' == c_low:
                col_map[c] = 'timestamp'
            elif 'open' == c_low: col_map[c] = 'open'
            elif 'high' == c_low: col_map[c] = 'high'
            elif 'low' == c_low: col_map[c] = 'low'
            elif 'close' == c_low: col_map[c] = 'close'
            elif 'last' == c_low: col_map[c] = 'close'
            elif 'vol' in c_low and 'iv' in c_low: col_map[c] = 'iv'
            elif 'underlying' in c_low: col_map[c] = 'underlying_price'
        
        df = df.rename(columns=col_map)
        
        # Ensure underlying_price points to close for labeling if missing
        if 'underlying_price' not in df.columns and 'close' in df.columns:
            df['underlying_price'] = df['close']
        elif 'underlying_price' not in df.columns:
            # Fallback for daily data / simple spot files
            price_col = [c for c in ['close', 'last', 'price'] if c in df.columns]
            if price_col:
                df['underlying_price'] = df[price_col[0]]

        # CHECK: If timestamp still missing, throw a descriptive error
        if 'timestamp' not in df.columns:
            st.error(f"❌ Could not find a 'timestamp' column. Available: {list(df.columns)}")
            st.info("Check if your CSV has a 'Date' or 'Time' header.")
            return pd.DataFrame()

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
        
        # 4. Resample if requested and requested timeframe is higher than M1
        if do_resample and tf_str != "1min":
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
        with st.spinner("📦 Processing dataset..."):
            # Passing params to load_data ensures the cache is unique per set of options
            data = load_data(input_csv, tf_map[timeframe], data_type, auto_resample)
        st.sidebar.success(f"✅ Data Ready: {len(data)} bars ({timeframe} {'Resampled' if auto_resample else 'Native'})")
    else:
        st.warning("Click 'Load Data' in the sidebar to begin.")
        st.stop()
    
    # --- DIAGNOSTICS SIDEBAR ---
    st.sidebar.subheader("📊 Data Stats")
    st.sidebar.write(f"Timeframe: **{timeframe}**")
    st.sidebar.write(f"Total Bars: **{len(data)}**")
    if not data.empty and len(data) > 1:
        st.sidebar.write(f"Start: {data['timestamp'].iloc[0].strftime('%Y-%m-%d')}")
        st.sidebar.write(f"End: {data['timestamp'].iloc[-1].strftime('%Y-%m-%d')}")
        
        # Detect Interval
        diff = (data['timestamp'].iloc[1] - data['timestamp'].iloc[0]).total_seconds() / 60
        st.sidebar.info(f"⏱️ Interval: **~{int(diff)} min**")
        
        # Price Diagnostics
        if 'close' in data.columns:
            p_min = data['close'].min()
            p_max = data['close'].max()
            st.sidebar.write(f"Price: **${p_min:.2f} - ${p_max:.2f}**")
    # ---------------------------
    
    # --- DATE FILTERING ---
    st.sidebar.subheader("📅 Navigation")
    try:
        min_date = data['timestamp'].min().date()
        max_date = data['timestamp'].max().date()
        date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])
    except:
        date_range = None
    
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_data = data[(data['timestamp'].dt.date >= start_date) & (data['timestamp'].dt.date <= end_date)].copy()
    else:
        filtered_data = data.copy()
    
    if limit_bars:
        filtered_data = filtered_data.sort_values('timestamp').tail(1000)
    
    gen_static = st.sidebar.button("📸 Generate Static Plot (Fallback)")
    auto_resample = st.sidebar.checkbox("Auto-Resample (Source must be M1)", value=False)

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
    
    # Unified Trace Logic
    if view_mode == "Candlesticks":
        fig.add_trace(go.Candlestick(
            x=filtered_data['timestamp'].tolist(),
            open=filtered_data['open'].tolist(),
            high=filtered_data['high'].tolist(),
            low=filtered_data['low'].tolist(),
            close=filtered_data['close'].tolist(),
            name='Candlesticks'
        ))
    elif view_mode == "OHLC Bars":
        fig.add_trace(go.Ohlc(
            x=filtered_data['timestamp'].tolist(),
            open=filtered_data['open'].tolist(),
            high=filtered_data['high'].tolist(),
            low=filtered_data['low'].tolist(),
            close=filtered_data['close'].tolist(),
            name='OHLC Bars'
        ))
    else:
        # Use standard Scatter for interaction (GL can be flaky with events)
        fig.add_trace(go.Scatter(
            x=filtered_data['timestamp'].tolist(),
            y=filtered_data['secondary_price'].tolist() if 'secondary_price' in filtered_data.columns else filtered_data['close'].tolist(),
            mode='lines+markers' if len(filtered_data) < 1000 else 'lines',
            marker=dict(size=4, opacity=0.5),
            line=dict(color='white', width=1),
            name='Price Line'
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
        title=f"SPY {timeframe} Chart ({len(filtered_data)} bars)",
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
        if len(filtered_data) > 5000:
            st.error(f"❌ **Dataset too large for Interactive Mode ({len(filtered_data)} bars)**")
            st.info("Interactive clicking only works reliably with < 5,000 bars. \n\n**Please:** \n1. Use the **'Date Range'** selector to pick a smaller window. \n2. Or check **'Focus: Last 500 bars'**. \n3. Or switch to **'Standard (Visual Only)'** mode.")
            selected_points = []
        else:
            from streamlit_plotly_events import plotly_events
            if len(filtered_data) > 2000:
                st.warning(f"⚠️ {len(filtered_data)} bars may cause click lag. Narrowing the Date Range is recommended.")
            
            # Use a unique key to force component refresh when data changes
            event_key = f"plotly_ev_{len(filtered_data)}_{timeframe}_{view_mode}"
            selected_points = plotly_events(fig, click_event=True, hover_event=False, override_height=700, key=event_key)
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

    if st.sidebar.button("🗑️ Clear All Pivots"):
        st.session_state.pivots = pd.DataFrame(columns=['timestamp', 'type', 'strength', 'price', 'timeframe'])
        st.rerun()

    # --- STATIC FALLBACK BLOCK ---
    if gen_static:
        st.subheader("📸 Static Matplotlib Plot (Fallback)")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        fig_plt, ax = plt.subplots(figsize=(12, 6))
        ax.plot(filtered_data['timestamp'], filtered_data['close'], color='blue', label='Price')
        
        # Plot existing pivots
        for tf_p in ["M1", "M5", "M15", "H1"]:
            pivs = st.session_state.pivots[st.session_state.pivots['timeframe'] == tf_p]
            if not pivs.empty:
                ax.scatter(pd.to_datetime(pivs['timestamp']), pivs['price'], 
                           marker='v' if tf_p == "H1" else '.', label=f'{tf_p} Pivots')
        
        ax.set_title(f"Static Plot ({timeframe}) - {len(filtered_data)} bars")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(fig_plt)

else:
    st.error(f"File not found: {input_csv}. Please check the sidebar path.")
