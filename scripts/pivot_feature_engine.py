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

    print(f"[INFO] Loading dataset: {args.input}")
    df = pd.read_csv(args.input)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['timestamp', 'option_symbol'])

    print(f"[INFO] Loading manual pivots: {args.pivots}")
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
    print("[INFO] Computing structural skeleton (Distance/Slope)...")
    
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
    
    # Vectorized Distance-to-Last-Pivot implementation
    # 1. Identify pivot locations
    is_pivot = (spots['pivot_high_flag'] == 1) | (spots['pivot_low_flag'] == 1)
    
    # 2. Create columns that only exist at pivot points, then forward fill
    spots['last_pivot_idx'] = np.where(is_pivot, spots.index, np.nan)
    spots['last_pivot_price'] = np.where(is_pivot, spots['underlying_price'], np.nan)
    
    # ffill propagates the last valid observation forward
    spots['last_pivot_idx'] = spots['last_pivot_idx'].ffill()
    spots['p_price_prev'] = spots['last_pivot_price'].ffill().fillna(0.0)
    
    # 3. Calculate distance (current index - last pivot index)
    # Using index distance as proxy for time distance in minutes if 1m bars, 
    # but safer to use timestamp diff if needed. The original code used index diff.
    # Let's stick to index diff for speed if consistent, or timestamp diff for accuracy.
    # Original: dist = i - last_idx
    spots['p_dist_prev'] = spots.index - spots['last_pivot_idx']
    spots['p_dist_prev'] = spots['p_dist_prev'].fillna(0.0)

    # Normalized distance and slope to last structure point
    spots['p_slope_prev'] = (spots['underlying_price'] - spots['p_price_prev']) / spots['p_dist_prev'].replace(0, 1)
    
    # --------------------------------------------------------
    # STEP 3: Time & Session Features (V4.2)
    # --------------------------------------------------------
    print("[INFO] Computing Session Time features...")
    # Assume US Eastern Time for market hours (09:30 - 16:00)
    # We work with 'spots' to speed up processing
    spots['time'] = spots['timestamp'].dt.time
    
    # Helper for minutes from 09:30
    def get_mins_from_open(row):
        # returns float minutes
        t = row['timestamp']
        # Naive calculation assuming standard trading days
        open_time = t.replace(hour=9, minute=30, second=0, microsecond=0)
        close_time = t.replace(hour=16, minute=0, second=0, microsecond=0)
        
        mins_open = (t - open_time).total_seconds() / 60.0
        mins_close = (close_time - t).total_seconds() / 60.0
        return mins_open, mins_close

    # Scalar apply is slow but safe for now. Can vectorise if needed.
    # Vectorized approach:
    # Construct open/close timestamps for each day
    day_open = spots['timestamp'].dt.floor('D') + pd.Timedelta(hours=9, minutes=30)
    day_close = spots['timestamp'].dt.floor('D') + pd.Timedelta(hours=16)
    
    spots['minutes_from_open'] = (spots['timestamp'] - day_open).dt.total_seconds() / 60.0
    spots['minutes_to_close'] = (day_close - spots['timestamp']).dt.total_seconds() / 60.0
    
    # Normalize (390 mins in standard session)
    spots['norm_session_time'] = spots['minutes_from_open'] / 390.0
    spots['norm_session_time'] = spots['norm_session_time'].clip(0, 1) # Clip ext hours
    
    # Phases
    # 1: Open (< 30m), 2: Mid, 3: Close (< 30m to close), 4: Ext
    conditions = [
        (spots['minutes_from_open'] < 0) | (spots['minutes_to_close'] < 0), # Ext
        (spots['minutes_from_open'] <= 30), # Open
        (spots['minutes_to_close'] <= 30),  # Close
    ]
    choices = [4, 1, 3] # Else 2 (Mid)
    spots['session_phase'] = np.select(conditions, choices, default=2)

    # --------------------------------------------------------
    # STEP 4: Regime Features (Vol & Liq)
    # --------------------------------------------------------
    print("[INFO] Computing Regime features (Vol & Liq)...")
    
    # Volatility Regime
    # Use 'vol_ewma' if available, else 'atr_pct'
    vol_col = 'vol_ewma' if 'vol_ewma' in spots.columns else 'iv' 
    # Fallback to iv if vol_ewma missing (V3 has vol_ewma usually)
    
    if vol_col in spots.columns:
        # 10-day rolling quantile for context (assuming 390 bars/day)
        # We use a long window to establish "regime" relative to recent history
        window = 390 * 5 
        
        # Rank of current vol relative to window
        # Rolling rank is expensive. Approximation: (x - min) / (max - min)
        roll_min = spots[vol_col].rolling(window=window, min_periods=100).min()
        roll_max = spots[vol_col].rolling(window=window, min_periods=100).max()
        
        spots['vol_regime_score'] = (spots[vol_col] - roll_min) / (roll_max - roll_min + 1e-6)
        spots['vol_regime_score'] = spots['vol_regime_score'].clip(0, 1).fillna(0.5)
        
        # Label: 0=Low, 1=Normal, 2=High
        spots['vol_regime_label'] = pd.cut(spots['vol_regime_score'], 
                                           bins=[-0.1, 0.33, 0.66, 1.1], 
                                           labels=[0, 1, 2]).astype(float)
    else:
        spots['vol_regime_score'] = 0.5
        spots['vol_regime_label'] = 1.0

    # Liquidity Regime
    # Proxy: spread_ratio (lower is better) or volume (higher is better)
    # We use spread_ratio as primary
    liq_col = 'spread_ratio'
    if liq_col in spots.columns:
        # Inverse logic: Low spread = High Liq
        roll_min_l = spots[liq_col].rolling(window=window, min_periods=100).min()
        roll_max_l = spots[liq_col].rolling(window=window, min_periods=100).max()
        
        # Normalized spread position (0=tight/good, 1=wide/bad)
        spread_norm = (spots[liq_col] - roll_min_l) / (roll_max_l - roll_min_l + 1e-6)
        
        # Liq Score: 1 - spread_norm (0=bad, 1=good)
        spots['liq_score'] = 1.0 - spread_norm.clip(0, 1).fillna(0.5)
        
        spots['liq_regime'] = pd.cut(spots['liq_score'], 
                                     bins=[-0.1, 0.33, 0.66, 1.1], 
                                     labels=[0, 1, 2]).astype(float)
    else:
        spots['liq_score'] = 0.5
        spots['liq_regime'] = 1.0

    # --------------------------------------------------------
    # STEP 5: MTF Pivot Alignment
    # --------------------------------------------------------
    # alignment logic: If m5 indicator agrees with manual pivot, strength++
    # (In later versions, we merge M1, M5, M15 manual files here)
    print("[INFO] Computing MTF Structural Alignment...")
    
    if 'rev_m5' in spots.columns:
        spots['pivot_align_indic'] = ((spots['pivot_high_flag'] == 1) & (spots['rev_m5'] > 0)).astype(int) | \
                                     ((spots['pivot_low_flag'] == 1) & (spots['rev_m5'] < 0)).astype(int)
    else:
        spots['pivot_align_indic'] = 0.0

    # --------------------------------------------------------
    # Final Broadcast
    # --------------------------------------------------------
    print("[INFO] Re-broadcasting structure to options chain...")
    
    new_features = [
        'minutes_from_open', 'minutes_to_close', 'norm_session_time', 'session_phase',
        'vol_regime_label', 'vol_regime_score', 
        'liq_score', 'liq_regime',
        'p_dist_prev', 'p_slope_prev', 'pivot_align_indic'
    ]
    
    broadcast_cols = ['timestamp'] + new_features
    
    # Ensure all cols exist in spots before merge
    for c in new_features:
        if c not in spots.columns:
            spots[c] = 0.0

    df = df.merge(spots[broadcast_cols], on='timestamp', how='left')

    print(f"[INFO] Saving v4.2 Structural set to: {args.output}")
    df.to_csv(args.output, index=False)
    print("[SUCCESS] Structural Feature Engineering complete.")

if __name__ == "__main__":
    main()
