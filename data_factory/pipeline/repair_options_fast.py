#!/usr/bin/env python3
"""
SPY Options Dataset Repair Tool - OPTIMIZED VERSION

Repairs structural issues in SPY options datasets using VECTORIZED operations:
1. Balances put/call ratio to 50/50 per timestamp
2. Uses vectorized Black-Scholes to interpolate missing options (10-100x faster)
3. Trims excess options by liquidity (if needed)

Usage:
    python repair_options_fast.py --file 2024|2025|both [--dry-run]
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Constants
TARGET_OPTIONS_PER_BAR = 100
TARGET_PUTS = 50
TARGET_CALLS = 50
RISK_FREE_RATE = 0.05

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                        'data', 'processed')


# ============================================================================
# VECTORIZED Black-Scholes Functions (NumPy-based, 10-100x faster)
# ============================================================================

def bs_d1_vec(S: np.ndarray, K: np.ndarray, T: np.ndarray, r: float, sigma: np.ndarray) -> np.ndarray:
    """Vectorized d1 calculation"""
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d1 = np.where((T <= 0) | (sigma <= 0) | (S <= 0) | (K <= 0), 0.0, d1)
    return d1


def bs_d2_vec(S: np.ndarray, K: np.ndarray, T: np.ndarray, r: float, sigma: np.ndarray) -> np.ndarray:
    """Vectorized d2 calculation"""
    return bs_d1_vec(S, K, T, r, sigma) - sigma * np.sqrt(np.maximum(T, 0.001))


def bs_call_price_vec(S: np.ndarray, K: np.ndarray, T: np.ndarray, r: float, sigma: np.ndarray) -> np.ndarray:
    """Vectorized Black-Scholes call price"""
    d1 = bs_d1_vec(S, K, T, r, sigma)
    d2 = bs_d2_vec(S, K, T, r, sigma)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    # Handle T <= 0 case
    intrinsic = np.maximum(0, S - K)
    return np.where(T <= 0, intrinsic, price)


def bs_put_price_vec(S: np.ndarray, K: np.ndarray, T: np.ndarray, r: float, sigma: np.ndarray) -> np.ndarray:
    """Vectorized Black-Scholes put price"""
    d1 = bs_d1_vec(S, K, T, r, sigma)
    d2 = bs_d2_vec(S, K, T, r, sigma)
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    # Handle T <= 0 case
    intrinsic = np.maximum(0, K - S)
    return np.where(T <= 0, intrinsic, price)


def bs_delta_vec(S: np.ndarray, K: np.ndarray, T: np.ndarray, r: float, sigma: np.ndarray, is_call: np.ndarray) -> np.ndarray:
    """Vectorized Black-Scholes Delta"""
    d1 = bs_d1_vec(S, K, T, r, sigma)
    call_delta = norm.cdf(d1)
    put_delta = norm.cdf(d1) - 1.0
    return np.where(is_call, call_delta, put_delta)


def bs_gamma_vec(S: np.ndarray, K: np.ndarray, T: np.ndarray, r: float, sigma: np.ndarray) -> np.ndarray:
    """Vectorized Black-Scholes Gamma"""
    d1 = bs_d1_vec(S, K, T, r, sigma)
    with np.errstate(divide='ignore', invalid='ignore'):
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        gamma = np.where((T <= 0) | (sigma <= 0) | (S <= 0), 0.0, gamma)
    return gamma


def bs_vega_vec(S: np.ndarray, K: np.ndarray, T: np.ndarray, r: float, sigma: np.ndarray) -> np.ndarray:
    """Vectorized Black-Scholes Vega (per 1% change)"""
    d1 = bs_d1_vec(S, K, T, r, sigma)
    vega = S * norm.pdf(d1) * np.sqrt(np.maximum(T, 0)) / 100
    return np.where((T <= 0) | (sigma <= 0) | (S <= 0), 0.0, vega)


def bs_theta_vec(S: np.ndarray, K: np.ndarray, T: np.ndarray, r: float, sigma: np.ndarray, is_call: np.ndarray) -> np.ndarray:
    """Vectorized Black-Scholes Theta (per day)"""
    d1 = bs_d1_vec(S, K, T, r, sigma)
    d2 = bs_d2_vec(S, K, T, r, sigma)
    
    term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(np.maximum(T, 0.001)))
    call_term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
    put_term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
    
    theta = (term1 + np.where(is_call, call_term2, put_term2)) / 365
    return np.where((T <= 0) | (sigma <= 0) | (S <= 0), 0.0, theta)


def bs_rho_vec(S: np.ndarray, K: np.ndarray, T: np.ndarray, r: float, sigma: np.ndarray, is_call: np.ndarray) -> np.ndarray:
    """Vectorized Black-Scholes Rho (per 1% change)"""
    d2 = bs_d2_vec(S, K, T, r, sigma)
    call_rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    put_rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    return np.where(is_call, call_rho, put_rho)


# ============================================================================
# Batch Synthetic Option Generation (FAST)
# ============================================================================

def generate_synthetic_options_batch(
    timestamps: np.ndarray,
    spots: np.ndarray,
    strikes: np.ndarray,
    is_calls: np.ndarray,
    expirations: np.ndarray,
    ivs: np.ndarray,
    ohlcv_data: dict
) -> pd.DataFrame:
    """
    Generate synthetic options in batch using vectorized Black-Scholes.
    
    This is 10-100x faster than row-by-row generation.
    """
    n = len(timestamps)
    if n == 0:
        return pd.DataFrame()
    
    # Ensure all inputs are numpy arrays
    S = np.asarray(spots, dtype=np.float64)
    K = np.asarray(strikes, dtype=np.float64)
    is_call = np.asarray(is_calls, dtype=bool)
    sigma = np.asarray(ivs, dtype=np.float64)
    sigma = np.where(sigma <= 0, 0.20, sigma)  # Default IV
    
    # Calculate time to expiration
    try:
        exp_dates = pd.to_datetime(expirations)
        ts_dates = pd.to_datetime(timestamps)
        dte = (exp_dates - ts_dates).dt.days.values
    except:
        dte = np.full(n, 30)  # Default 30 DTE
    
    T = np.maximum(dte / 365.0, 0.001)
    r = RISK_FREE_RATE
    
    # Vectorized price calculation
    call_prices = bs_call_price_vec(S, K, T, r, sigma)
    put_prices = bs_put_price_vec(S, K, T, r, sigma)
    prices = np.where(is_call, call_prices, put_prices)
    
    # Vectorized Greeks calculation
    deltas = bs_delta_vec(S, K, T, r, sigma, is_call)
    gammas = bs_gamma_vec(S, K, T, r, sigma)
    thetas = bs_theta_vec(S, K, T, r, sigma, is_call)
    vegas = bs_vega_vec(S, K, T, r, sigma)
    rhos = bs_rho_vec(S, K, T, r, sigma, is_call)
    
    # Generate option symbols
    option_symbols = []
    for i in range(n):
        try:
            exp = pd.to_datetime(expirations[i])
            exp_str = exp.strftime('%y%m%d')
        except:
            exp_str = '250131'  # Default
        cp = 'C' if is_calls[i] else 'P'
        strike_str = f"{int(strikes[i] * 1000):08d}"
        option_symbols.append(f"SPY   {exp_str}{cp}{strike_str}")
    
    # Build DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'symbol': 'SPY',
        'underlying_price': S,
        'open': ohlcv_data.get('open', S),
        'high': ohlcv_data.get('high', S),
        'low': ohlcv_data.get('low', S),
        'close': ohlcv_data.get('close', S),
        'option_symbol': option_symbols,
        'expiration': expirations,
        'strike': K,
        'call_put': np.where(is_call, 'C', 'P'),
        'bid': np.round(prices * 0.98, 2),
        'ask': np.round(prices * 1.02, 2),
        'iv': np.round(sigma, 4),
        'delta': np.round(deltas, 4),
        'gamma': np.round(gammas, 4),
        'theta': np.round(thetas, 4),
        'vega': np.round(vegas, 4),
        'rho': np.round(rhos, 4),
        'volume': 0,
        'open_interest': 0
    })
    
    return df


def select_strikes_atm(spot_price: float, num_strikes: int = 50, strike_width: float = 1.0) -> list:
    """Select strikes centered around ATM"""
    atm_strike = round(spot_price / strike_width) * strike_width
    half = num_strikes // 2
    strikes = [atm_strike + (i * strike_width) for i in range(-half, half + 1) if atm_strike + (i * strike_width) > 0]
    return sorted(strikes[:num_strikes])


# ============================================================================
# Optimized Repair Logic
# ============================================================================

def repair_timestamp_group_fast(
    group: pd.DataFrame,
    timestamp: str,
    target_puts: int = TARGET_PUTS,
    target_calls: int = TARGET_CALLS
) -> pd.DataFrame:
    """
    FAST repair of a single timestamp group using batch operations.
    """
    # Normalize call_put column
    if 'call_put' in group.columns:
        is_call = group['call_put'] == 'C'
    elif 'option_type' in group.columns:
        is_call = group['option_type'].str.upper().str.startswith('C')
    else:
        is_call = group['option_symbol'].str.contains('C')
    
    current_puts = group[~is_call].copy()
    current_calls = group[is_call].copy()
    
    n_puts = len(current_puts)
    n_calls = len(current_calls)
    
    # Get common data
    first_row = group.iloc[0]
    spot = first_row.get('underlying_price', first_row.get('close', 500))
    expiration = str(first_row.get('expiration', ''))
    avg_iv = group['iv'].mean() if 'iv' in group.columns and group['iv'].notna().any() else 0.20
    
    ohlcv = {
        'open': first_row.get('open', spot),
        'high': first_row.get('high', spot),
        'low': first_row.get('low', spot),
        'close': first_row.get('close', spot)
    }
    
    result_parts = []
    
    # Handle puts
    if n_puts > target_puts:
        # Keep most liquid puts
        if 'volume' in current_puts.columns and 'open_interest' in current_puts.columns:
            current_puts['_liq'] = current_puts['volume'].fillna(0) * current_puts['open_interest'].fillna(0)
            current_puts = current_puts.nlargest(target_puts, '_liq').drop(columns=['_liq'])
        else:
            current_puts = current_puts.head(target_puts)
        result_parts.append(current_puts)
    elif n_puts < target_puts:
        result_parts.append(current_puts)
        
        # Generate missing puts in BATCH
        n_missing = target_puts - n_puts
        existing_strikes = set(current_puts['strike'].unique())
        all_strikes = select_strikes_atm(spot, num_strikes=100)
        put_strikes = [s for s in all_strikes if s <= spot and s not in existing_strikes]
        put_strikes = sorted(put_strikes, key=lambda x: abs(x - spot))[:n_missing]
        
        if put_strikes:
            synthetic_puts = generate_synthetic_options_batch(
                timestamps=np.full(len(put_strikes), timestamp),
                spots=np.full(len(put_strikes), spot),
                strikes=np.array(put_strikes),
                is_calls=np.zeros(len(put_strikes), dtype=bool),
                expirations=np.full(len(put_strikes), expiration),
                ivs=np.full(len(put_strikes), avg_iv),
                ohlcv_data=ohlcv
            )
            result_parts.append(synthetic_puts)
    else:
        result_parts.append(current_puts)
    
    # Handle calls
    if n_calls > target_calls:
        # Keep most liquid calls
        if 'volume' in current_calls.columns and 'open_interest' in current_calls.columns:
            current_calls['_liq'] = current_calls['volume'].fillna(0) * current_calls['open_interest'].fillna(0)
            current_calls = current_calls.nlargest(target_calls, '_liq').drop(columns=['_liq'])
        else:
            current_calls = current_calls.head(target_calls)
        result_parts.append(current_calls)
    elif n_calls < target_calls:
        result_parts.append(current_calls)
        
        # Generate missing calls in BATCH
        n_missing = target_calls - n_calls
        existing_strikes = set(current_calls['strike'].unique())
        all_strikes = select_strikes_atm(spot, num_strikes=100)
        call_strikes = [s for s in all_strikes if s >= spot and s not in existing_strikes]
        call_strikes = sorted(call_strikes, key=lambda x: abs(x - spot))[:n_missing]
        
        if call_strikes:
            synthetic_calls = generate_synthetic_options_batch(
                timestamps=np.full(len(call_strikes), timestamp),
                spots=np.full(len(call_strikes), spot),
                strikes=np.array(call_strikes),
                is_calls=np.ones(len(call_strikes), dtype=bool),
                expirations=np.full(len(call_strikes), expiration),
                ivs=np.full(len(call_strikes), avg_iv),
                ohlcv_data=ohlcv
            )
            result_parts.append(synthetic_calls)
    else:
        result_parts.append(current_calls)
    
    if result_parts:
        return pd.concat(result_parts, ignore_index=True)
    else:
        return group


def repair_dataset_fast(filepath: str, output_path: str, dry_run: bool = False, chunk_size: int = 500000):
    """
    OPTIMIZED repair using larger chunks and vectorized operations.
    """
    filename = os.path.basename(filepath)
    print(f"\n{'='*60}", flush=True)
    print(f"REPAIRING (OPTIMIZED): {filename}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Mode: {'DRY RUN' if dry_run else 'FULL REPAIR'}", flush=True)
    print(f"Output: {output_path}", flush=True)
    print(f"Chunk size: {chunk_size:,} rows", flush=True)
    
    stats = {
        'total_rows_in': 0,
        'total_rows_out': 0,
        'timestamps_processed': 0,
        'timestamps_repaired': 0,
        'synthetic_puts_added': 0,
        'synthetic_calls_added': 0,
        'excess_puts_removed': 0,
        'excess_calls_removed': 0
    }
    
    first_chunk = True
    import time
    start_time = time.time()
    
    for chunk_num, chunk in enumerate(pd.read_csv(filepath, chunksize=chunk_size)):
        chunk_start = time.time()
        print(f"\n[Chunk {chunk_num + 1}] Processing {len(chunk):,} rows...", flush=True)
        stats['total_rows_in'] += len(chunk)
        
        if 'timestamp' not in chunk.columns:
            print("  ERROR: No timestamp column!", flush=True)
            continue
        
        chunk['ts_normalized'] = pd.to_datetime(chunk['timestamp']).dt.floor('min').astype(str)
        
        repaired_chunks = []
        
        for ts, group in chunk.groupby('ts_normalized'):
            stats['timestamps_processed'] += 1
            
            if 'call_put' in group.columns:
                n_puts = (group['call_put'] == 'P').sum()
                n_calls = (group['call_put'] == 'C').sum()
            else:
                n_puts = n_calls = 0
            
            needs_repair = (n_puts != TARGET_PUTS or n_calls != TARGET_CALLS or 
                           len(group) != TARGET_OPTIONS_PER_BAR)
            
            if needs_repair:
                stats['timestamps_repaired'] += 1
                
                if n_puts > TARGET_PUTS:
                    stats['excess_puts_removed'] += (n_puts - TARGET_PUTS)
                elif n_puts < TARGET_PUTS:
                    stats['synthetic_puts_added'] += (TARGET_PUTS - n_puts)
                
                if n_calls > TARGET_CALLS:
                    stats['excess_calls_removed'] += (n_calls - TARGET_CALLS)
                elif n_calls < TARGET_CALLS:
                    stats['synthetic_calls_added'] += (TARGET_CALLS - n_calls)
                
                if not dry_run:
                    repaired = repair_timestamp_group_fast(group.drop(columns=['ts_normalized']), ts)
                    repaired_chunks.append(repaired)
                else:
                    repaired_chunks.append(group.drop(columns=['ts_normalized']))
            else:
                repaired_chunks.append(group.drop(columns=['ts_normalized']))
        
        if repaired_chunks:
            repaired_chunk = pd.concat(repaired_chunks, ignore_index=True)
            stats['total_rows_out'] += len(repaired_chunk)
            
            if not dry_run:
                if first_chunk:
                    repaired_chunk.to_csv(output_path, index=False, mode='w')
                    first_chunk = False
                else:
                    repaired_chunk.to_csv(output_path, index=False, mode='a', header=False)
        
        chunk_time = time.time() - chunk_start
        elapsed = time.time() - start_time
        
        print(f"  ✓ Rows: {stats['total_rows_in']:,} → {stats['total_rows_out']:,}", flush=True)
        print(f"  ✓ Timestamps repaired: {stats['timestamps_repaired']:,} | Calls added: {stats['synthetic_calls_added']:,}", flush=True)
        print(f"  ⏱ Chunk time: {chunk_time:.1f}s | Total: {elapsed/60:.1f}min", flush=True)
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}", flush=True)
    print("REPAIR SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Total time:             {total_time/60:.1f} minutes", flush=True)
    print(f"Total rows in:          {stats['total_rows_in']:,}", flush=True)
    print(f"Total rows out:         {stats['total_rows_out']:,}", flush=True)
    print(f"Timestamps processed:   {stats['timestamps_processed']:,}", flush=True)
    print(f"Timestamps repaired:    {stats['timestamps_repaired']:,}", flush=True)
    print(f"Synthetic puts added:   {stats['synthetic_puts_added']:,}", flush=True)
    print(f"Synthetic calls added:  {stats['synthetic_calls_added']:,}", flush=True)
    print(f"Excess puts removed:    {stats['excess_puts_removed']:,}", flush=True)
    print(f"Excess calls removed:   {stats['excess_calls_removed']:,}", flush=True)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Repair SPY Options Datasets (OPTIMIZED)')
    parser.add_argument('--file', choices=['2024', '2025', 'both'], default='both',
                        help='Which dataset to repair')
    parser.add_argument('--dry-run', action='store_true',
                        help='Analyze only, do not write repaired files')
    parser.add_argument('--output-suffix', default='_cleaned',
                        help='Suffix for output files (default: _cleaned)')
    
    args = parser.parse_args()
    
    print("=" * 60, flush=True)
    print("SPY OPTIONS DATASET REPAIR TOOL (OPTIMIZED)", flush=True)
    print("=" * 60, flush=True)
    print(f"Mode: {'DRY RUN' if args.dry_run else 'FULL REPAIR'}", flush=True)
    print(f"Target: {args.file}", flush=True)
    print("Using vectorized Black-Scholes (10-100x faster)", flush=True)
    
    if args.file in ['2024', 'both']:
        filepath = os.path.join(DATA_DIR, 'Spy_Options_2024_1m.csv')
        output = os.path.join(DATA_DIR, f'Spy_Options_2024_1m{args.output_suffix}.csv')
        if os.path.exists(filepath):
            repair_dataset_fast(filepath, output, dry_run=args.dry_run)
        else:
            print(f"⚠️  File not found: {filepath}", flush=True)
    
    if args.file in ['2025', 'both']:
        filepath = os.path.join(DATA_DIR, 'Spy_Options_2025_1m.csv')
        output = os.path.join(DATA_DIR, f'Spy_Options_2025_1m{args.output_suffix}.csv')
        if os.path.exists(filepath):
            repair_dataset_fast(filepath, output, dry_run=args.dry_run)
        else:
            print(f"⚠️  File not found: {filepath}", flush=True)
    
    print("\n" + "=" * 60, flush=True)
    print("REPAIR COMPLETE", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
