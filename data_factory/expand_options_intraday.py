"""
Expand daily ivolatility options data to 1-minute and 5-minute granularity
using Black-Scholes interpolation with Alpaca intraday spot data.

Usage:
    python data_factory/expand_options_intraday.py --input data/ivolatility/spy_ivol_1month.csv
    
Outputs:
    data/ivolatility/spy_ivol_1month_1min.csv
    data/ivolatility/spy_ivol_1month_5min.csv
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

from core.config import RunConfig

# Alpaca imports
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
except ImportError:
    print("[Error] alpaca-py not installed. Run: pip install alpaca-py")
    sys.exit(1)


# ============================================================
# BLACK-SCHOLES FORMULAS
# ============================================================

def bs_d1(S, K, T, r, sigma):
    """Calculate d1 for Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def bs_d2(S, K, T, r, sigma):
    """Calculate d2 for Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return bs_d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call option price."""
    if T <= 0:
        return max(0, S - K)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = bs_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put_price(S, K, T, r, sigma):
    """Black-Scholes put option price."""
    if T <= 0:
        return max(0, K - S)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = bs_d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_delta(S, K, T, r, sigma, is_call):
    """Calculate delta."""
    if T <= 0:
        if is_call:
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    d1 = bs_d1(S, K, T, r, sigma)
    if is_call:
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1

def bs_gamma(S, K, T, r, sigma):
    """Calculate gamma."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = bs_d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def bs_vega(S, K, T, r, sigma):
    """Calculate vega (per 1% IV change)."""
    if T <= 0:
        return 0.0
    d1 = bs_d1(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) / 100

def bs_theta(S, K, T, r, sigma, is_call):
    """Calculate theta (per day)."""
    if T <= 0:
        return 0.0
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = bs_d2(S, K, T, r, sigma)
    
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    
    if is_call:
        term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
    
    return (term1 + term2) / 365  # Per day


# ============================================================
# DATA FETCHING
# ============================================================

def fetch_intraday_bars(api_key, api_secret, symbol, start_date, end_date, timeframe="1Min"):
    """Fetch intraday bars from Alpaca."""
    print(f"[Alpaca] Fetching {timeframe} bars for {symbol} from {start_date} to {end_date}...")
    
    client = StockHistoricalDataClient(api_key, api_secret)
    
    if timeframe == "1Min":
        tf = TimeFrame.Minute
    elif timeframe == "5Min":
        tf = TimeFrame(5, TimeFrameUnit.Minute)
    else:
        tf = TimeFrame.Minute
    
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=start_date,
        end=end_date
    )
    
    bars = client.get_stock_bars(req)
    df = bars.df
    
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)
    
    df = df.reset_index()
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
    
    print(f"[Alpaca] Fetched {len(df)} bars.")
    return df


# ============================================================
# EXPANSION LOGIC
# ============================================================

def expand_options_to_intraday(options_df, spot_df, risk_free_rate=0.05):
    """
    Expand daily options data to intraday using BS interpolation.
    
    Args:
        options_df: Daily ivolatility data with columns: date, strike, expiration, call_put, iv, etc.
        spot_df: Intraday SPY bars with columns: timestamp, close
        risk_free_rate: Risk-free rate for BS calculations
    
    Returns:
        DataFrame with intraday options data
    """
    print("[Expand] Starting Black-Scholes interpolation...")
    
    # Standardize column names
    options_df.columns = [c.lower().strip() for c in options_df.columns]
    
    # Ensure date columns are proper types
    options_df['date'] = pd.to_datetime(options_df['date']).dt.date
    options_df['expiration'] = pd.to_datetime(options_df['expiration']).dt.date
    
    spot_df['timestamp'] = pd.to_datetime(spot_df['timestamp'])
    spot_df['date'] = spot_df['timestamp'].dt.date
    
    # Get unique dates in options data
    option_dates = options_df['date'].unique()
    
    results = []
    
    for opt_date in option_dates:
        # Get options for this date
        day_options = options_df[options_df['date'] == opt_date]
        
        # Get intraday bars for this date
        day_bars = spot_df[spot_df['date'] == opt_date]
        
        if day_bars.empty:
            print(f"  [Skip] No intraday data for {opt_date}")
            continue
        
        print(f"  [Process] {opt_date}: {len(day_options)} options × {len(day_bars)} bars")
        
        # For each intraday bar
        for _, bar in day_bars.iterrows():
            bar_ts = bar['timestamp']
            spot_price = bar['close']
            
            # For each option on this day
            for _, opt in day_options.iterrows():
                strike = opt['strike']
                expiration = opt['expiration']
                is_call = str(opt['call_put']).upper() == 'C'
                iv = opt.get('iv', opt.get('iv_raw', 0.20))  # Default 20% if missing
                
                # Calculate time to expiration (in years)
                # Account for intraday time
                days_to_exp = (expiration - bar_ts.date()).days
                # Add fractional day based on time (market closes at 4pm ET = hour 16)
                hour_fraction = (16 - bar_ts.hour) / 24.0 if bar_ts.hour < 16 else 0
                T = max(0, (days_to_exp + hour_fraction) / 365.0)
                
                # Calculate BS values
                if is_call:
                    price = bs_call_price(spot_price, strike, T, risk_free_rate, iv)
                else:
                    price = bs_put_price(spot_price, strike, T, risk_free_rate, iv)
                
                delta = bs_delta(spot_price, strike, T, risk_free_rate, iv, is_call)
                gamma = bs_gamma(spot_price, strike, T, risk_free_rate, iv)
                vega = bs_vega(spot_price, strike, T, risk_free_rate, iv)
                theta = bs_theta(spot_price, strike, T, risk_free_rate, iv, is_call)
                
                # Create result row
                row = {
                    'timestamp': bar_ts,
                    'date': opt_date,
                    'symbol': opt.get('symbol', 'SPY'),
                    'option_symbol': opt.get('option_symbol', ''),
                    'underlying_price': spot_price,
                    'strike': strike,
                    'expiration': expiration,
                    'call_put': 'C' if is_call else 'P',
                    'iv': iv,
                    'mean_price': price,
                    'bid': price * 0.98,  # Estimate 2% spread
                    'ask': price * 1.02,
                    'delta': delta,
                    'gamma': gamma,
                    'vega': vega,
                    'theta': theta,
                    'dte': days_to_exp
                }
                results.append(row)
    
    result_df = pd.DataFrame(results)
    print(f"[Expand] Generated {len(result_df)} intraday option rows.")
    return result_df


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Expand daily options to intraday granularity")
    parser.add_argument("--input", type=str, required=True, help="Path to daily ivolatility CSV")
    parser.add_argument("--key", type=str, help="Alpaca API Key")
    parser.add_argument("--secret", type=str, help="Alpaca Secret Key")
    args = parser.parse_args()
    
    # Resolve API keys
    api_key = args.key
    api_secret = args.secret
    
    if not api_key:
        try:
            cfg = RunConfig()
            api_key = cfg.alpaca_key
            api_secret = cfg.alpaca_secret
            print("[Config] Using Alpaca keys from config.py")
        except Exception as e:
            print(f"[Error] Could not load config: {e}")
            return
    
    # Load options data
    print(f"[Load] Reading {args.input}...")
    options_df = pd.read_csv(args.input)
    print(f"[Load] Loaded {len(options_df)} option rows.")
    
    # Determine date range
    options_df['date'] = pd.to_datetime(options_df['date'])
    start_date = options_df['date'].min()
    end_date = options_df['date'].max() + timedelta(days=1)
    
    print(f"[Range] Options data: {start_date.date()} to {end_date.date()}")
    
    # Fetch 1-minute bars
    spot_1min = fetch_intraday_bars(api_key, api_secret, "SPY", start_date, end_date, "1Min")
    
    # Generate 1-minute options file
    print("\n" + "="*60)
    print(" GENERATING 1-MINUTE OPTIONS FILE")
    print("="*60)
    options_1min = expand_options_to_intraday(options_df.copy(), spot_1min)
    
    output_1min = args.input.replace('.csv', '_1min.csv')
    options_1min.to_csv(output_1min, index=False)
    print(f"[Saved] {output_1min} ({len(options_1min)} rows)")
    
    # Resample to 5-minute and generate 5-minute file
    print("\n" + "="*60)
    print(" GENERATING 5-MINUTE OPTIONS FILE")
    print("="*60)
    
    # Resample spot data to 5-min
    spot_1min['timestamp'] = pd.to_datetime(spot_1min['timestamp'])
    spot_1min = spot_1min.set_index('timestamp')
    spot_5min = spot_1min.resample('5T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()
    
    options_5min = expand_options_to_intraday(options_df.copy(), spot_5min)
    
    output_5min = args.input.replace('.csv', '_5min.csv')
    options_5min.to_csv(output_5min, index=False)
    print(f"[Saved] {output_5min} ({len(options_5min)} rows)")
    
    print("\n[Complete] Both files generated successfully!")
    print(f"  1-min: {output_1min}")
    print(f"  5-min: {output_5min}")


if __name__ == "__main__":
    main()
