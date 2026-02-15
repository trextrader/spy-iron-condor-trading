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
pivot_type = st.sidebar.radio("Pivot Type to Mark", ["High", "Low"])
pivot_strength = st.sidebar.slider("Pivot Strength", 1, 3, 1)

if os.path.exists(input_csv):
    @st.cache_data
    def load_data(path):
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Unique spots for UI
        spots = df.groupby('timestamp').first().reset_index().sort_values('timestamp')
        return spots

    data = load_data(input_csv)
    
    # Session selector (optional filtering)
    date_range = st.sidebar.date_input("Date Range", [data['timestamp'].min(), data['timestamp'].max()])
    filtered_data = data[(data['timestamp'].dt.date >= date_range[0]) & (data['timestamp'].dt.date <= date_range[1])]

    # Manual labels state
    if 'pivots' not in st.session_state:
        st.session_state.pivots = pd.DataFrame(columns=['timestamp', 'type', 'strength', 'price'])

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
    if not st.session_state.pivots.empty:
        highs = st.session_state.pivots[st.session_state.pivots['type'] == 'High']
        lows = st.session_state.pivots[st.session_state.pivots['type'] == 'Low']
        
        fig.add_trace(go.Scatter(
            x=highs['timestamp'], y=highs['price'], mode='markers',
            marker=dict(symbol='triangle-down', color='red', size=12), name='Labled Highs'
        ))
        fig.add_trace(go.Scatter(
            x=lows['timestamp'], y=lows['price'], mode='markers',
            marker=dict(symbol='triangle-up', color='green', size=12), name='Labeled Lows'
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
            'price': [price]
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
