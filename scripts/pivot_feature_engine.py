import pandas as pd
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Merge manual pivots and compute structural features.")
    parser.add_argument("--input", type=str, required=True, help="Path to v4.2 training dataset (CSV)")
    parser.add_argument("--pivots", type=str, required=True, help="Path to manual pivots (CSV from labeler)")
    parser.add_argument("--output", type=str, required=True, help="Output path for v4.2_STRUCTURE dataset")
    args = parser.parse_args()

    print(f"📥 Loading dataset: {args.input}")
    df = pd.read_csv(args.input)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['timestamp', 'option_symbol'])

    print(f"📥 Loading manual pivots: {args.pivots}")
    pivots = pd.read_csv(args.pivots)
    pivots['timestamp'] = pd.to_datetime(pivots['timestamp'])
    
    # --------------------------------------------------------
    # STEP 1: Merge Clicks as Flags
    # --------------------------------------------------------
    # We broadcast the underlying pivot to all option rows at that timestamp
    pivots['pivot_high_flag'] = (pivots['type'] == 'High').astype(int)
    pivots['pivot_low_flag'] = (pivots['type'] == 'Low').astype(int)
    pivots.rename(columns={'strength': 'pivot_strength'}, inplace=True)

    # Timeframe-aware flags
    for tf in ["M1", "M5", "M15", "H1"]:
        pivots[f'pivot_{tf}_flag'] = (pivots['timeframe'] == tf).astype(int)

    pivot_merge_cols = ['timestamp', 'pivot_high_flag', 'pivot_low_flag', 'pivot_strength']
    for tf in ["M1", "M5", "M15", "H1"]:
        if f'pivot_{tf}_flag' in pivots.columns:
            pivot_merge_cols.append(f'pivot_{tf}_flag')

    pivot_merge = pivots[pivot_merge_cols].copy()

    df = df.merge(pivot_merge, on='timestamp', how='left')
    df['pivot_high_flag'] = df['pivot_high_flag'].fillna(0).astype(int)
    df['pivot_low_flag'] = df['pivot_low_flag'].fillna(0).astype(int)
    df['pivot_strength'] = df['pivot_strength'].fillna(0)
    for tf in ["M1", "M5", "M15", "H1"]:
        col = f'pivot_{tf}_flag'
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # --------------------------------------------------------
    # STEP 2: Structural Distances (Per-Option Series)
    # --------------------------------------------------------
    # These metrics describe the market "skeleton"
    print("🦴 Computing structural skeleton (Distance/Slope)...")
    
    def compute_structural_skeleton(group):
        group = group.sort_values('timestamp')
        
        # Binary signal for any pivot
        has_pivot = (group['pivot_high_flag'] == 1) | (group['pivot_low_flag'] == 1)
        
        if has_pivot.sum() < 2:
            group['p_dist_prev'] = 0
            group['p_slope_prev'] = 0
            return group

        # Distance from previous pivot
        group['p_dist_prev'] = group['timestamp'].diff().dt.total_seconds() / 60.0 # mins
        # Placeholder for complex cumulative sums
        # Note: In a real training set, we use pre-computed indices for faster lookup
        
        return group

    # Since pivots are shared by all options at a timestamp, 
    # we can compute this on the 'spot' series once and broadcast.
    spots = df.groupby('timestamp').first().reset_index().sort_values('timestamp')
    
    # Simple Distance-to-Last-Pivot implementation
    pivot_indices = spots.index[(spots['pivot_high_flag'] == 1) | (spots['pivot_low_flag'] == 1)].tolist()
    
    spots['p_dist_prev'] = 0.0
    spots['p_price_prev'] = 0.0
    
    for i in range(len(spots)):
        prev_pivots = [idx for idx in pivot_indices if idx < i]
        if prev_pivots:
            last_idx = prev_pivots[-1]
            dist = i - last_idx
            spots.loc[i, 'p_dist_prev'] = dist
            spots.loc[i, 'p_price_prev'] = spots.loc[last_idx, 'underlying_price']

    # Normalized distance and slope to last structure point
    spots['p_slope_prev'] = (spots['underlying_price'] - spots['p_price_prev']) / spots['p_dist_prev'].replace(0, 1)
    
    # --------------------------------------------------------
    # STEP 3: MTF Pivot Alignment
    # --------------------------------------------------------
    # alignment logic: If m5 indicator agrees with manual pivot, strength++
    # (In later versions, we merge M1, M5, M15 manual files here)
    print("📐 Computing MTF Structural Alignment...")
    
    spots['pivot_align_indic'] = ((spots['pivot_high_flag'] == 1) & (spots['rev_m5'] > 0)).astype(int) | \
                                 ((spots['pivot_low_flag'] == 1) & (spots['rev_m5'] < 0)).astype(int)

    # --------------------------------------------------------
    # Final Broadcast
    # --------------------------------------------------------
    print("📡 Re-broadcasting structure to options chain...")
    broadcast_cols = ['timestamp', 'p_dist_prev', 'p_slope_prev', 'pivot_align_indic']
    df = df.merge(spots[broadcast_cols], on='timestamp', how='left')

    print(f"💾 Saving v4.2 Structural set to: {args.output}")
    df.to_csv(args.output, index=False)
    print("🎉 Structural Feature Engineering complete.")

if __name__ == "__main__":
    main()
