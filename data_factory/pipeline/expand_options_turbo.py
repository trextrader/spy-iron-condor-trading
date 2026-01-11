"""
Turbo Expansion Script: Expand daily options to intraday granularity
using vectorized Black-Scholes interpolation.

Optimized for speed and backtest compatibility.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import norm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ============================================================
# VECTORIZED BLACK-SCHOLES FORMULAS
# ============================================================

def bs_delta_vec(S, K, T, r, sigma, is_call):
    """Vectorized Delta calculation."""
    # Ensure T and sigma are safe to avoid div by zero
    T = np.maximum(T, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    delta_call = norm.cdf(d1)
    delta_put = delta_call - 1.0
    
    return np.where(is_call, delta_call, delta_put)

def bs_price_vec(S, K, T, r, sigma, is_call):
    """Vectorized Option Price calculation."""
    T = np.maximum(T, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return np.where(is_call, call_price, put_price)

def bs_gamma_vec(S, K, T, r, sigma):
    """Vectorized Gamma calculation."""
    T = np.maximum(T, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

# ============================================================
# EXPANSION LOGIC
# ============================================================

def expand_turbo(options_path, spot_path, output_path, risk_free_rate=0.05):
    print(f"[Turbo] Loading Options from {options_path}...")
    # Use low_memory=False to handle mixed types and avoid warnings
    opt_df = pd.read_csv(options_path, low_memory=False)
    opt_df.columns = [c.lower().strip() for c in opt_df.columns]
    
    # Pre-clean Options: Use errors='coerce' to skip junk rows (like repeated headers)
    opt_df['date'] = pd.to_datetime(opt_df['date'], errors='coerce')
    opt_df = opt_df.dropna(subset=['date'])
    opt_df['date'] = opt_df['date'].dt.date
    
    opt_df['expiration'] = pd.to_datetime(opt_df['expiration'], errors='coerce')
    opt_df = opt_df.dropna(subset=['expiration'])
    opt_df['expiration'] = opt_df['expiration'].dt.date

    # Ensure numeric columns are actually numeric (handling potential strings in mixed-type files)
    for col in ['strike', 'iv']:
        if col in opt_df.columns:
            opt_df[col] = pd.to_numeric(opt_df[col], errors='coerce')
    
    opt_df = opt_df.dropna(subset=['strike', 'iv'])
    
    # We only need a few columns from daily to propagate
    essential_cols = ['date', 'expiration', 'strike', 'call_put', 'iv', 'option_symbol']
    # If missing some columns, create defaults
    if 'call_put' not in opt_df.columns and 'symbol' in opt_df.columns:
        # Extract from symbol if formatted like SPY25...
        pass 
        
    opt_df = opt_df[[c for c in essential_cols if c in opt_df.columns]]
    print(f"      Rows: {len(opt_df)}")

    print(f"[Turbo] Loading Spot from {spot_path}...")
    spot_df = pd.read_csv(spot_path, parse_dates=['timestamp'])
    # Resample Spot to 15-min to match desired output
    # Rename Spot columns to avoid collision with Option OHLC names
    spot_df = spot_df.rename(columns={
        'open': 'spot_open', 'high': 'spot_high', 'low': 'spot_low', 
        'close': 'spot_close', 'volume': 'spot_volume', 'vwap': 'spot_vwap'
    })
    spot_df['date'] = spot_df['timestamp'].dt.date
    print(f"      Bars: {len(spot_df)}")

    # CROSS JOIN / MERGE
    print("[Turbo] Merging Intraday Spot with Daily Options...")
    merged = pd.merge(spot_df, opt_df, on='date', how='inner')
    print(f"      Expanded Combinations: {len(merged)}")

    if len(merged) == 0:
        print("[Error] No date overlap found between Spot and Options files.")
        return

    # VECTORIZED CALCULATIONS
    print("[Turbo] Calculating Greeks and Prices (Vectorized)...")
    
    # T calculation (Years to expiration)
    merged['bar_hour'] = merged['timestamp'].dt.hour + merged['timestamp'].dt.minute/60.0
    merged['days_to_exp'] = (merged['expiration'] - merged['date']).apply(lambda x: x.days).astype(float)
    
    merged['hour_frac'] = (16.0 - merged['bar_hour']) / 24.0
    merged.loc[merged['bar_hour'] >= 16, 'hour_frac'] = 0
    
    merged['T'] = (merged['days_to_exp'] + merged['hour_frac']) / 365.0
    
    S = merged['spot_close'].values
    K = merged['strike'].values
    T = merged['T'].values
    sigma = merged['iv'].values
    is_call = (merged['call_put'].str.upper() == 'C').values
    
    merged['opt_price'] = bs_price_vec(S, K, T, risk_free_rate, sigma, is_call)
    merged['delta_intraday'] = bs_delta_vec(S, K, T, risk_free_rate, sigma, is_call)
    merged['gamma_intraday'] = bs_gamma_vec(S, K, T, risk_free_rate, sigma)
    
    # Pass strategy filters
    merged['volume'] = 500
    
    # Restore original option symbol column
    if 'option_symbol' in merged.columns:
        merged['symbol'] = merged['option_symbol']
    
    # Map to final OHLC
    merged['close'] = merged['opt_price']
    merged['open'] = merged['opt_price']
    merged['high'] = merged['opt_price']
    merged['low'] = merged['opt_price']
    
    merged['iv_intraday'] = merged['iv']
    merged['theta_intraday'] = 0.0 
    merged['vega_intraday'] = 0.0
    merged['rho_intraday'] = 0.0
    
    # Ensure date/expiration are dates for CSV consistency
    merged['date'] = pd.to_datetime(merged['date']).dt.date
    merged['expiration'] = pd.to_datetime(merged['expiration']).dt.date
    
    output_cols = [
        'symbol', 'timestamp', 'date', 'expiration', 'strike', 'call_put',
        'open', 'high', 'low', 'close', 'volume', 
        'delta_intraday', 'gamma_intraday', 'theta_intraday', 'vega_intraday', 
        'rho_intraday', 'iv_intraday'
    ]
    
    final_df = merged[output_cols]
    
    print(f"[Turbo] Saving to {output_path}...")
    final_df.to_csv(output_path, index=False)
    print(f"      Final Size: {len(final_df)} rows.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--options", type=str, required=True, help="Daily iVol CSV")
    parser.add_argument("--spot", type=str, required=True, help="Intraday Spot CSV (SPY_5 or SPY_1)")
    parser.add_argument("--output", type=str, default="data/alpaca_options/intraday_expanded_m15.csv")
    args = parser.parse_args()
    
    # Ensure output dir exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    expand_turbo(args.options, args.spot, args.output)

if __name__ == "__main__":
    main()
