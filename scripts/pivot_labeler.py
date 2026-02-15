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
input_csv = st.sidebar.text_input("Input CSV Path", "data/Datasetv4/condornet_v41_FINAL.csv")
output_csv = st.sidebar.text_input("Export filename", "pivots_manual.csv")
timeframe = st.sidebar.selectbox("View Timeframe", ["M1", "M5", "M15", "H1"], index=0)
pivot_type = st.sidebar.radio("Pivot Type to Mark", ["High", "Low"])
pivot_strength = st.sidebar.slider("Pivot Strength", 1, 3, 1)

# TF Mapping
tf_map = {"M1": "1min", "M5": "5min", "M15": "15min", "H1": "1H"}

if os.path.exists(input_csv):
    @st.cache_data
    def load_data(path, tf_str):
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Unique spots for UI
        spots = df.groupby('timestamp').first().reset_index().sort_values('timestamp')
        
        if tf_str != "1min":
            spots = spots.set_index('timestamp').resample(tf_str).agg({
                'underlying_price': 'last', # Use last for price line
                'rev_m5': 'mean',
                'rev_m15': 'mean'
            }).dropna().reset_index()
        return spots

    data = load_data(input_csv, tf_map[timeframe])
    
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
    
    # Underlying Price
    fig.add_trace(go.Scatter(
        x=filtered_data['timestamp'], 
        y=filtered_data['underlying_price'], 
        name='Price (Spot)',
        line=dict(color='white', width=1)
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
        height=700, 
        template="plotly_dark", 
        xaxis_rangeslider_visible=True,
        clickmode='event+select'
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
