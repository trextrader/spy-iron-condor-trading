#!/usr/bin/env python3
"""
SPY Options Dataset Repair Tool

Repairs structural issues in SPY options datasets:
1. Balances put/call ratio to 50/50 per timestamp
2. Uses Black-Scholes to interpolate missing option prices and Greeks
3. Trims excess options by liquidity (if needed)

Usage:
    python repair_options_datasets.py --file 2024|2025|both [--dry-run]
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
from scipy.stats import norm
from scipy.optimize import brentq
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Constants
TARGET_OPTIONS_PER_BAR = 100
TARGET_PUTS = 50
TARGET_CALLS = 50
RISK_FREE_RATE = 0.05  # 5% annual risk-free rate

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                        'data', 'processed')


# ============================================================================
# Black-Scholes Functions (copied from compute_historical_greeks.py)
# ============================================================================

def bs_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate d1 for Black-Scholes"""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def bs_d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate d2 for Black-Scholes"""
    if T <= 0 or sigma <= 0:
        return 0.0
    return bs_d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call price"""
    if T <= 0:
        return max(0, S - K)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = bs_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put price"""
    if T <= 0:
        return max(0, K - S)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = bs_d2(S, K, T, r, sigma)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    """Black-Scholes Delta"""
    if T <= 0 or sigma <= 0:
        if is_call:
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    d1 = bs_d1(S, K, T, r, sigma)
    if is_call:
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1.0


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes Gamma (same for calls and puts)"""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = bs_d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * math.sqrt(T))


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes Vega (per 1% change in volatility)"""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = bs_d1(S, K, T, r, sigma)
    return S * norm.pdf(d1) * math.sqrt(T) / 100


def bs_theta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    """Black-Scholes Theta (per day)"""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = bs_d2(S, K, T, r, sigma)
    
    term1 = -S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
    
    if is_call:
        term2 = -r * K * math.exp(-r * T) * norm.cdf(d2)
    else:
        term2 = r * K * math.exp(-r * T) * norm.cdf(-d2)
    
    return (term1 + term2) / 365


def bs_rho(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    """Black-Scholes Rho (per 1% change in rates)"""
    if T <= 0 or sigma <= 0:
        return 0.0
    d2 = bs_d2(S, K, T, r, sigma)
    
    if is_call:
        return K * T * math.exp(-r * T) * norm.cdf(d2) / 100
    else:
        return -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100


def implied_volatility(price: float, S: float, K: float, T: float, r: float, 
                        is_call: bool, max_iter: int = 50) -> float:
    """Calculate implied volatility using Brent's method"""
    if T <= 0 or price <= 0 or S <= 0 or K <= 0:
        return 0.20  # Default IV
    
    intrinsic = max(0, S - K) if is_call else max(0, K - S)
    if price < intrinsic:
        return 0.05  # Very low IV
    
    def objective(sigma):
        if is_call:
            return bs_call_price(S, K, T, r, sigma) - price
        else:
            return bs_put_price(S, K, T, r, sigma) - price
    
    try:
        iv = brentq(objective, 0.001, 3.0, maxiter=max_iter)
        return iv
    except (ValueError, RuntimeError):
        return 0.20  # Default


# ============================================================================
# Strike Selection and Option Generation
# ============================================================================

def generate_option_symbol(underlying: str, expiration: datetime, strike: float, is_call: bool) -> str:
    """Generate OCC option symbol format"""
    exp_str = expiration.strftime('%y%m%d')
    cp = 'C' if is_call else 'P'
    strike_str = f"{int(strike * 1000):08d}"
    return f"{underlying.ljust(6)}{exp_str}{cp}{strike_str}"


def select_strikes_atm(spot_price: float, num_strikes: int = 50, strike_width: float = 1.0) -> list:
    """
    Select strikes centered around ATM (at-the-money).
    
    Args:
        spot_price: Current underlying price
        num_strikes: Number of strikes to generate
        strike_width: Distance between strikes
        
    Returns:
        List of strike prices centered around ATM
    """
    # Round spot to nearest strike
    atm_strike = round(spot_price / strike_width) * strike_width
    
    # Generate strikes above and below ATM
    half = num_strikes // 2
    strikes = []
    
    for i in range(-half, half + 1):
        strike = atm_strike + (i * strike_width)
        if strike > 0:
            strikes.append(strike)
    
    # Ensure we have exactly num_strikes
    while len(strikes) < num_strikes:
        # Add more OTM strikes
        if strikes[-1] + strike_width not in strikes:
            strikes.append(strikes[-1] + strike_width)
        elif strikes[0] - strike_width > 0:
            strikes.insert(0, strikes[0] - strike_width)
    
    return sorted(strikes[:num_strikes])


def interpolate_option(
    timestamp: str,
    ohlcv: dict,
    strike: float,
    is_call: bool,
    expiration: str,
    avg_iv: float,
    underlying_symbol: str = 'SPY'
) -> dict:
    """
    Interpolate a synthetic option row using Black-Scholes.
    
    Args:
        timestamp: Timestamp string
        ohlcv: Dict with open, high, low, close, underlying_price
        strike: Strike price
        is_call: True for call, False for put
        expiration: Expiration date string
        avg_iv: Average IV from existing options at this timestamp
        underlying_symbol: Underlying symbol
        
    Returns:
        Dictionary representing a synthetic option row
    """
    S = ohlcv['underlying_price']
    K = strike
    
    # Parse expiration date first (needed for both T calculation and symbol generation)
    try:
        exp_date = pd.to_datetime(expiration)
    except:
        # Default to 30 days from timestamp if expiration is invalid
        try:
            exp_date = pd.to_datetime(timestamp) + pd.Timedelta(days=30)
        except:
            exp_date = pd.Timestamp.now() + pd.Timedelta(days=30)
    
    # Calculate time to expiration
    try:
        ts_date = pd.to_datetime(timestamp)
        dte = (exp_date - ts_date).days
        T = max(dte / 365.0, 0.001)
    except:
        T = 30 / 365.0  # Default 30 DTE
    
    r = RISK_FREE_RATE
    sigma = avg_iv if avg_iv > 0 else 0.20
    
    # Calculate option price
    if is_call:
        price = bs_call_price(S, K, T, r, sigma)
    else:
        price = bs_put_price(S, K, T, r, sigma)
    
    # Calculate Greeks
    delta = bs_delta(S, K, T, r, sigma, is_call)
    gamma = bs_gamma(S, K, T, r, sigma)
    theta = bs_theta(S, K, T, r, sigma, is_call)
    vega = bs_vega(S, K, T, r, sigma)
    rho = bs_rho(S, K, T, r, sigma, is_call)
    
    # Generate option symbol
    option_symbol = generate_option_symbol(underlying_symbol, exp_date, K, is_call)
    
    return {
        'timestamp': timestamp,
        'symbol': underlying_symbol,
        'underlying_price': S,
        'open': ohlcv.get('open', S),
        'high': ohlcv.get('high', S),
        'low': ohlcv.get('low', S),
        'close': ohlcv.get('close', S),
        'option_symbol': option_symbol,
        'expiration': expiration,
        'strike': K,
        'call_put': 'C' if is_call else 'P',
        'bid': round(price * 0.98, 2),  # Synthetic bid (2% below mid)
        'ask': round(price * 1.02, 2),  # Synthetic ask (2% above mid)
        'iv': round(sigma, 4),
        'delta': round(delta, 4),
        'gamma': round(gamma, 4),
        'theta': round(theta, 4),
        'vega': round(vega, 4),
        'rho': round(rho, 4),
        'volume': 0,  # Synthetic option - no volume
        'open_interest': 0
    }


# ============================================================================
# Main Repair Logic
# ============================================================================

def repair_timestamp_group(
    group: pd.DataFrame,
    timestamp: str,
    target_puts: int = TARGET_PUTS,
    target_calls: int = TARGET_CALLS
) -> pd.DataFrame:
    """
    Repair a single timestamp group to have exactly target_puts and target_calls.
    
    Strategy:
    1. Count existing puts and calls
    2. If excess puts/calls, remove least liquid (lowest volume * OI)
    3. If missing puts/calls, interpolate using Black-Scholes
    """
    # Normalize call_put column
    if 'call_put' in group.columns:
        group['_is_call'] = group['call_put'] == 'C'
    elif 'option_type' in group.columns:
        group['_is_call'] = group['option_type'].str.upper().str.startswith('C')
    else:
        # Try to infer from option symbol
        group['_is_call'] = group['option_symbol'].str.contains('C')
    
    current_puts = group[~group['_is_call']]
    current_calls = group[group['_is_call']]
    
    n_puts = len(current_puts)
    n_calls = len(current_calls)
    
    # Get OHLCV data (same for all rows in group)
    first_row = group.iloc[0]
    ohlcv = {
        'underlying_price': first_row.get('underlying_price', first_row.get('close', 0)),
        'open': first_row.get('open', 0),
        'high': first_row.get('high', 0),
        'low': first_row.get('low', 0),
        'close': first_row.get('close', 0)
    }
    
    # Get common expiration
    expiration = str(first_row.get('expiration', ''))
    
    # Calculate average IV from existing options
    avg_iv = group['iv'].mean() if 'iv' in group.columns and group['iv'].notna().any() else 0.20
    
    repaired_rows = []
    
    # Handle puts
    if n_puts > target_puts:
        # Remove least liquid puts
        if 'volume' in current_puts.columns and 'open_interest' in current_puts.columns:
            current_puts = current_puts.copy()
            current_puts['_liquidity'] = current_puts['volume'].fillna(0) * current_puts['open_interest'].fillna(0)
            current_puts = current_puts.nlargest(target_puts, '_liquidity')
            current_puts = current_puts.drop(columns=['_liquidity', '_is_call'])
        else:
            current_puts = current_puts.head(target_puts)
        repaired_rows.append(current_puts.drop(columns=['_is_call'], errors='ignore'))
    elif n_puts < target_puts:
        # Keep existing puts
        repaired_rows.append(current_puts.drop(columns=['_is_call'], errors='ignore'))
        
        # Need to add synthetic puts
        n_missing = target_puts - n_puts
        existing_put_strikes = set(current_puts['strike'].unique())
        spot = ohlcv['underlying_price']
        
        # Select strikes for new puts (below ATM, not already existing)
        all_strikes = select_strikes_atm(spot, num_strikes=100)
        put_strikes = [s for s in all_strikes if s <= spot and s not in existing_put_strikes]
        
        # Sort by distance from ATM (closest first)
        put_strikes = sorted(put_strikes, key=lambda x: abs(x - spot))[:n_missing]
        
        for strike in put_strikes:
            synthetic = interpolate_option(
                timestamp, ohlcv, strike, is_call=False,
                expiration=expiration, avg_iv=avg_iv
            )
            repaired_rows.append(pd.DataFrame([synthetic]))
    else:
        # Correct number of puts
        repaired_rows.append(current_puts.drop(columns=['_is_call'], errors='ignore'))
    
    # Handle calls
    if n_calls > target_calls:
        # Remove least liquid calls
        if 'volume' in current_calls.columns and 'open_interest' in current_calls.columns:
            current_calls = current_calls.copy()
            current_calls['_liquidity'] = current_calls['volume'].fillna(0) * current_calls['open_interest'].fillna(0)
            current_calls = current_calls.nlargest(target_calls, '_liquidity')
            current_calls = current_calls.drop(columns=['_liquidity', '_is_call'])
        else:
            current_calls = current_calls.head(target_calls)
        repaired_rows.append(current_calls.drop(columns=['_is_call'], errors='ignore'))
    elif n_calls < target_calls:
        # Keep existing calls
        repaired_rows.append(current_calls.drop(columns=['_is_call'], errors='ignore'))
        
        # Need to add synthetic calls
        n_missing = target_calls - n_calls
        existing_call_strikes = set(current_calls['strike'].unique())
        spot = ohlcv['underlying_price']
        
        # Select strikes for new calls (above ATM, not already existing)
        all_strikes = select_strikes_atm(spot, num_strikes=100)
        call_strikes = [s for s in all_strikes if s >= spot and s not in existing_call_strikes]
        
        # Sort by distance from ATM (closest first)
        call_strikes = sorted(call_strikes, key=lambda x: abs(x - spot))[:n_missing]
        
        for strike in call_strikes:
            synthetic = interpolate_option(
                timestamp, ohlcv, strike, is_call=True,
                expiration=expiration, avg_iv=avg_iv
            )
            repaired_rows.append(pd.DataFrame([synthetic]))
    else:
        # Correct number of calls
        repaired_rows.append(current_calls.drop(columns=['_is_call'], errors='ignore'))
    
    # Combine all rows
    if repaired_rows:
        result = pd.concat(repaired_rows, ignore_index=True)
        return result
    else:
        return group.drop(columns=['_is_call'], errors='ignore')


def repair_dataset(filepath: str, output_path: str, dry_run: bool = False, chunk_size: int = 100000):
    """
    Repair an entire dataset, processing in chunks.
    
    Args:
        filepath: Input CSV path
        output_path: Output CSV path
        dry_run: If True, only analyze without writing
        chunk_size: Rows to process at a time
    """
    filename = os.path.basename(filepath)
    print(f"\n{'='*60}", flush=True)
    print(f"REPAIRING: {filename}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Mode: {'DRY RUN' if dry_run else 'FULL REPAIR'}", flush=True)
    print(f"Output: {output_path}", flush=True)
    
    # Statistics
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
    
    # Process file in chunks
    first_chunk = True
    
    for chunk_num, chunk in enumerate(pd.read_csv(filepath, chunksize=chunk_size)):
        print(f"\n[Chunk {chunk_num + 1}] Loading {chunk_size:,} rows...", flush=True)
        stats['total_rows_in'] += len(chunk)
        
        # Normalize timestamp column
        if 'timestamp' in chunk.columns:
            chunk['ts_normalized'] = pd.to_datetime(chunk['timestamp']).dt.floor('min').astype(str)
        else:
            print("  ERROR: No timestamp column found!", flush=True)
            continue
        
        repaired_chunks = []
        
        # Group by timestamp and repair each group
        for ts, group in chunk.groupby('ts_normalized'):
            stats['timestamps_processed'] += 1
            
            # Check current balance
            if 'call_put' in group.columns:
                n_puts = (group['call_put'] == 'P').sum()
                n_calls = (group['call_put'] == 'C').sum()
            else:
                n_puts = n_calls = 0  # Will trigger repair
            
            needs_repair = (n_puts != TARGET_PUTS or n_calls != TARGET_CALLS or 
                           len(group) != TARGET_OPTIONS_PER_BAR)
            
            if needs_repair:
                stats['timestamps_repaired'] += 1
                
                # Track what we're changing
                if n_puts > TARGET_PUTS:
                    stats['excess_puts_removed'] += (n_puts - TARGET_PUTS)
                elif n_puts < TARGET_PUTS:
                    stats['synthetic_puts_added'] += (TARGET_PUTS - n_puts)
                
                if n_calls > TARGET_CALLS:
                    stats['excess_calls_removed'] += (n_calls - TARGET_CALLS)
                elif n_calls < TARGET_CALLS:
                    stats['synthetic_calls_added'] += (TARGET_CALLS - n_calls)
                
                if not dry_run:
                    repaired = repair_timestamp_group(group, ts)
                    repaired_chunks.append(repaired)
                else:
                    repaired_chunks.append(group.drop(columns=['ts_normalized'], errors='ignore'))
            else:
                repaired_chunks.append(group.drop(columns=['ts_normalized'], errors='ignore'))
        
        # Combine repaired chunks
        if repaired_chunks:
            repaired_chunk = pd.concat(repaired_chunks, ignore_index=True)
            stats['total_rows_out'] += len(repaired_chunk)
            
            # Remove helper column if present
            if 'ts_normalized' in repaired_chunk.columns:
                repaired_chunk = repaired_chunk.drop(columns=['ts_normalized'])
            
            # Write to output
            if not dry_run:
                if first_chunk:
                    repaired_chunk.to_csv(output_path, index=False, mode='w')
                    first_chunk = False
                else:
                    repaired_chunk.to_csv(output_path, index=False, mode='a', header=False)
        
        print(f"  ✓ Rows: {stats['total_rows_in']:,} in → {stats['total_rows_out']:,} out", flush=True)
        print(f"  ✓ Timestamps repaired: {stats['timestamps_repaired']:,} | Calls added: {stats['synthetic_calls_added']:,}", flush=True)
    
    # Print summary
    print(f"\n{'='*60}", flush=True)
    print("REPAIR SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
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
    parser = argparse.ArgumentParser(description='Repair SPY Options Datasets')
    parser.add_argument('--file', choices=['2024', '2025', 'both'], default='both',
                        help='Which dataset to repair')
    parser.add_argument('--dry-run', action='store_true',
                        help='Analyze only, do not write repaired files')
    parser.add_argument('--output-suffix', default='_cleaned',
                        help='Suffix for output files (default: _cleaned)')
    
    args = parser.parse_args()
    
    print("=" * 60, flush=True)
    print("SPY OPTIONS DATASET REPAIR TOOL", flush=True)
    print("=" * 60, flush=True)
    print(f"Mode: {'DRY RUN' if args.dry_run else 'FULL REPAIR'}", flush=True)
    print(f"Target: {args.file}", flush=True)
    
    if args.file in ['2024', 'both']:
        filepath = os.path.join(DATA_DIR, 'Spy_Options_2024_1m.csv')
        output = os.path.join(DATA_DIR, f'Spy_Options_2024_1m{args.output_suffix}.csv')
        if os.path.exists(filepath):
            repair_dataset(filepath, output, dry_run=args.dry_run)
        else:
            print(f"⚠️  File not found: {filepath}")
    
    if args.file in ['2025', 'both']:
        filepath = os.path.join(DATA_DIR, 'Spy_Options_2025_1m.csv')
        output = os.path.join(DATA_DIR, f'Spy_Options_2025_1m{args.output_suffix}.csv')
        if os.path.exists(filepath):
            repair_dataset(filepath, output, dry_run=args.dry_run)
        else:
            print(f"⚠️  File not found: {filepath}")
    
    print("\n" + "=" * 60)
    print("REPAIR COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
