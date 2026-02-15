import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
import os

def create_pivot_tool(csv_path):
    print(f"📈 Loading data for pivot labeling: {csv_path}")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # We only need unique spots for the UI
    spots = df.groupby('timestamp')[['underlying_price', 'open', 'high', 'low', 'close']].first().reset_index()
    spots = spots.sort_values('timestamp')

    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.03)

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=spots['timestamp'],
        open=spots['open'],
        high=spots['high'],
        low=spots['low'],
        close=spots['close'],
        name='Underlying'
    ))

    fig.update_layout(
        title=f"CondorNet v4.2 Pivot Labeler - {os.path.basename(csv_path)}",
        xaxis_rangeslider_visible=False,
        height=800,
        template="plotly_dark",
        clickmode='event+select'
    )

    print("💡 INSTRUCTIONS:")
    print("1. Zoom into a local high/low.")
    print("2. Click the peak/trough.")
    print("3. (Note: This standalone script generates the visualization. For persistent labeling, run in Streamlit).")
    
    output_html = "pivot_labeler_session.html"
    fig.write_html(output_html)
    print(f"✅ Visualization saved to {output_html}. Open this in your browser to inspect pivots.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()
    create_pivot_tool(args.input)
