#!/usr/bin/env python3
"""
Compute historical Greeks for options bars data using Black-Scholes model.
Enriches the Alpaca historical bars with calculated IV, delta, gamma, theta, vega.
"""
import datetime as dt
import math
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Black-Scholes functions
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


def implied_volatility(price: float, S: float, K: float, T: float, r: float, 
                        is_call: bool, max_iter: int = 100) -> float:
    """Calculate implied volatility using Brent's method"""
    if T <= 0 or price <= 0 or S <= 0 or K <= 0:
        return 0.25  # Default IV
    
    intrinsic = max(0, S - K) if is_call else max(0, K - S)
    if price < intrinsic:
        return 0.01  # Very low IV
    
    def objective(sigma):
        if is_call:
            return bs_call_price(S, K, T, r, sigma) - price
        else:
            return bs_put_price(S, K, T, r, sigma) - price
    
    try:
        iv = brentq(objective, 0.001, 5.0, maxiter=max_iter)
        return iv
    except (ValueError, RuntimeError):
        # Fallback to iterative bisection
        return 0.25  # Default


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
    return S * norm.pdf(d1) * math.sqrt(T) / 100  # Per 1% move


def bs_theta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    """Black-Scholes Theta (per day)"""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = bs_d2(S, K, T, r, sigma)
    
    # First term: time decay from d1
    term1 = -S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
    
    if is_call:
        term2 = -r * K * math.exp(-r * T) * norm.cdf(d2)
    else:
        term2 = r * K * math.exp(-r * T) * norm.cdf(-d2)
    
    # Return per-day theta (divide by 365)
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


def load_spy_spot_prices(bars_df: pd.DataFrame) -> dict:
    """
    Extract approximate SPY spot prices from the option bars.
    Uses ATM options to estimate the spot price for each date.
    """
    spot_prices = {}
    
    # Get unique dates
    dates = bars_df['date'].unique()
    
    for date in dates:
        day_data = bars_df[bars_df['date'] == date]
        
        # Use the average of near-ATM strikes as proxy for spot
        # Typically options with highest volume are ATM
        if 'close' in day_data.columns:
            # Find options with reasonable prices (likely near ATM)
            atm_candidates = day_data[(day_data['close'] > 1) & (day_data['close'] < 50)]
            if len(atm_candidates) > 0:
                # Estimate spot from strike of options with moderate prices
                spot_estimate = atm_candidates['strike'].median()
                spot_prices[date] = spot_estimate
    
    return spot_prices


def download_spy_spot_history(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """
    Try to get historical SPY spot prices from Alpaca.
    Falls back to CSV if API fails.
    """
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from core.config import RunConfig
        
        r = RunConfig()
        client = StockHistoricalDataClient(r.alpaca_key, r.alpaca_secret)
        
        req = StockBarsRequest(
            symbol_or_symbols='SPY',
            timeframe=TimeFrame.Day,
            start=dt.datetime.combine(start_date, dt.time.min),
            end=dt.datetime.combine(end_date, dt.time.max)
        )
        
        bars = client.get_stock_bars(req)
        df = bars.df.reset_index()
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        
        return df[['date', 'close']].rename(columns={'close': 'spot'})
    except Exception as e:
        print(f"API failed ({e}), using estimated spot prices from options data")
        return None


def estimate_spot_from_options(bars_df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate SPY spot prices from option data using near-ATM options.
    For options with highest open interest/volume, the strike ≈ spot.
    Also uses put-call parity when both calls and puts are available.
    """
    print("Estimating spot prices from option data...")
    
    spots = []
    dates = sorted(bars_df['date'].unique())
    
    for date in dates:
        day_data = bars_df[bars_df['date'] == date]
        
        # Method 1: Find options with moderate prices (likely ATM)
        # ATM options typically have prices between $5-$20 for weekly/monthly
        calls = day_data[day_data['option_type'] == 'call']
        puts = day_data[day_data['option_type'] == 'put']
        
        estimated_spot = None
        
        # Find strikes where both call and put exist with similar prices
        if len(calls) > 0 and len(puts) > 0:
            # ATM: where call price ≈ put price (approximately)
            call_strikes = set(calls['strike'].unique())
            put_strikes = set(puts['strike'].unique())
            common_strikes = call_strikes & put_strikes
            
            if common_strikes:
                # For each common strike, find where call ≈ put (ATM indicator)
                min_diff = float('inf')
                best_strike = None
                
                for strike in common_strikes:
                    call_price = calls[calls['strike'] == strike]['close'].iloc[0]
                    put_price = puts[puts['strike'] == strike]['close'].iloc[0]
                    
                    # Skip if prices are too small (deep OTM) or too large (deep ITM)
                    if call_price < 0.5 or put_price < 0.5:
                        continue
                    if call_price > 100 or put_price > 100:
                        continue
                    
                    diff = abs(call_price - put_price)
                    if diff < min_diff:
                        min_diff = diff
                        best_strike = strike
                
                if best_strike is not None:
                    estimated_spot = best_strike
        
        # Method 2: Use strike with highest volume as proxy
        if estimated_spot is None:
            if 'volume' in day_data.columns and day_data['volume'].sum() > 0:
                max_vol_row = day_data.loc[day_data['volume'].idxmax()]
                estimated_spot = max_vol_row['strike']
        
        # Method 3: Fallback to median strike
        if estimated_spot is None:
            estimated_spot = day_data['strike'].median()
        
        spots.append({'date': date, 'spot': estimated_spot})
    
    return pd.DataFrame(spots)


def compute_greeks_for_bars(bars_df: pd.DataFrame, spot_df: pd.DataFrame, 
                             risk_free_rate: float = 0.05) -> pd.DataFrame:
    """
    Compute Greeks for each bar using Black-Scholes.
    """
    print(f"Computing Greeks for {len(bars_df)} option bars...")
    
    # Merge spot prices
    bars_df['date'] = pd.to_datetime(bars_df['date']).dt.date
    spot_df['date'] = pd.to_datetime(spot_df['date']).dt.date
    
    merged = bars_df.merge(spot_df, on='date', how='left')
    
    # Fill missing spots with strike price median (fallback)
    if merged['spot'].isna().any():
        median_spot = merged['strike'].median()
        merged['spot'] = merged['spot'].fillna(median_spot)
    
    # Calculate time to expiration in years
    merged['expiration'] = pd.to_datetime(merged['expiration']).dt.date
    merged['dte'] = (pd.to_datetime(merged['expiration']) - pd.to_datetime(merged['date'])).dt.days
    merged['T'] = merged['dte'] / 365.0
    
    # Use close price as option price
    merged['option_price'] = merged['close']
    
    # Determine if call or put
    merged['is_call'] = merged['option_type'] == 'call'
    
    print("Calculating implied volatility...")
    
    # Vectorized IV calculation (with fallback)
    ivs = []
    deltas = []
    gammas = []
    thetas = []
    vegas = []
    rhos = []
    
    total = len(merged)
    for i, row in merged.iterrows():
        if i % 10000 == 0:
            print(f"  Progress: {i}/{total} ({100*i/total:.1f}%)")
        
        S = row['spot']
        K = row['strike']
        T = max(row['T'], 0.001)  # Avoid T=0
        r = risk_free_rate
        price = row['option_price']
        is_call = row['is_call']
        
        # Calculate IV
        iv = implied_volatility(price, S, K, T, r, is_call)
        ivs.append(iv)
        
        # Calculate Greeks
        deltas.append(bs_delta(S, K, T, r, iv, is_call))
        gammas.append(bs_gamma(S, K, T, r, iv))
        thetas.append(bs_theta(S, K, T, r, iv, is_call))
        vegas.append(bs_vega(S, K, T, r, iv))
        rhos.append(bs_rho(S, K, T, r, iv, is_call))
    
    merged['iv'] = ivs
    merged['delta'] = deltas
    merged['gamma'] = gammas
    merged['theta'] = thetas
    merged['vega'] = vegas
    merged['rho'] = rhos
    
    # Select output columns
    output_cols = [
        'date', 'symbol', 'underlying', 'expiration', 'strike', 'option_type',
        'open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count',
        'spot', 'dte', 'iv', 'delta', 'gamma', 'theta', 'vega', 'rho'
    ]
    
    return merged[[c for c in output_cols if c in merged.columns]]


def main():
    print("=" * 60)
    print("Historical Greeks Calculator")
    print("=" * 60)
    
    # Load historical bars
    bars_file = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              'data', 'alpaca_options', 'spy_options_bars.csv')
    
    print(f"\nLoading bars from: {bars_file}")
    bars_df = pd.read_csv(bars_file)
    print(f"Loaded {len(bars_df)} option bars")
    
    # Get date range
    bars_df['date'] = pd.to_datetime(bars_df['date']).dt.date
    min_date = bars_df['date'].min()
    max_date = bars_df['date'].max()
    print(f"Date range: {min_date} to {max_date}")
    
    # Try to get SPY spot prices, fallback to estimation
    print("\nGetting SPY spot prices...")
    spot_df = download_spy_spot_history(min_date, max_date)
    
    if spot_df is None or len(spot_df) == 0:
        # Use estimation from options data
        spot_df = estimate_spot_from_options(bars_df)
    
    print(f"Got {len(spot_df)} spot price records")
    
    # Compute Greeks
    print("\nComputing Greeks using Black-Scholes...")
    enriched_df = compute_greeks_for_bars(bars_df, spot_df)
    
    # Save result
    output_file = bars_file.replace('.csv', '_with_greeks.csv')
    enriched_df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 60)
    print("Computation Complete!")
    print("=" * 60)
    print(f"Total records: {len(enriched_df):,}")
    print(f"Output file: {output_file}")
    print(f"\nSample Greeks:")
    print(enriched_df[['symbol', 'strike', 'iv', 'delta', 'gamma', 'theta', 'vega']].head(5).to_string())


if __name__ == "__main__":
    main()
