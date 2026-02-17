"""
microstructure.py — Microstructure Aggregate Features
Columns: aggregate_ask_count, aggregate_bid_count, mean_ask, mean_bid, quote_spread

These aggregate option-level microstructure data to bar-level metrics.
For spot-only datasets, these are derived from bid/ask columns if available.
"""
import numpy as np
import pandas as pd

def compute_microstructure(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    Compute microstructure aggregates per bar.
    
    For datasets with multiple rows per timestamp (option chains):
    - aggregate_bid/ask_count = number of active quotes
    - mean_bid/ask = average bid/ask across the chain
    - quote_spread = mean_ask - mean_bid
    
    For single-row-per-bar datasets:
    - Uses bid/ask columns directly
    """
    result = pd.DataFrame(index=df.index)
    
    if timestamp_col in df.columns:
        # Check if there are multiple rows per timestamp (option chain)
        rows_per_ts = df.groupby(timestamp_col).size()
        is_chain = rows_per_ts.median() > 1
        
        if is_chain and 'bid' in df.columns and 'ask' in df.columns:
            # Multi-row aggregation
            agg = df.groupby(timestamp_col).agg(
                aggregate_bid_count=('bid', lambda x: (x > 0).sum()),
                aggregate_ask_count=('ask', lambda x: (x > 0).sum()),
                mean_bid=('bid', 'mean'),
                mean_ask=('ask', 'mean'),
            ).reset_index()
            agg['quote_spread'] = agg['mean_ask'] - agg['mean_bid']
            
            df = df.merge(agg, on=timestamp_col, how='left', suffixes=('', '_agg'))
            for col in ['aggregate_bid_count', 'aggregate_ask_count', 'mean_bid', 'mean_ask', 'quote_spread']:
                agg_col = col + '_agg'
                if agg_col in df.columns:
                    result[col] = df[agg_col].astype(np.float32)
                elif col in df.columns:
                    result[col] = df[col].astype(np.float32)
            return result
    
    # Single-row per bar
    if 'bid' in df.columns and 'ask' in df.columns:
        result['mean_bid'] = df['bid'].astype(np.float32)
        result['mean_ask'] = df['ask'].astype(np.float32)
        result['quote_spread'] = (df['ask'] - df['bid']).astype(np.float32)
        result['aggregate_bid_count'] = (df['bid'] > 0).astype(np.float32)
        result['aggregate_ask_count'] = (df['ask'] > 0).astype(np.float32)
    elif 'bid_size' in df.columns and 'ask_size' in df.columns:
        result['aggregate_bid_count'] = df['bid_size'].astype(np.float32)
        result['aggregate_ask_count'] = df['ask_size'].astype(np.float32)
        result['mean_bid'] = df.get('bid', pd.Series(0, index=df.index)).astype(np.float32)
        result['mean_ask'] = df.get('ask', pd.Series(0, index=df.index)).astype(np.float32)
        result['quote_spread'] = (result['mean_ask'] - result['mean_bid']).astype(np.float32)
    else:
        # No microstructure data available — fill with safe defaults
        result['aggregate_bid_count'] = np.float32(0)
        result['aggregate_ask_count'] = np.float32(0)
        result['mean_bid'] = np.float32(0)
        result['mean_ask'] = np.float32(0)
        result['quote_spread'] = np.float32(0)
    
    return result

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    
    df = pd.read_csv(args.input)
    micro = compute_microstructure(df)
    for col in micro.columns:
        df[col] = micro[col]
    df.to_csv(args.output, index=False)
    print(f"[OK] aggregate_bid_count, aggregate_ask_count, mean_bid, mean_ask, quote_spread → {args.output}")
