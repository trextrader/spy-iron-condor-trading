import pandas as pd
import numpy as np
import pandas_ta as ta
import os
from scipy.stats import norm

# --- Vectorized Black-Scholes Engine ---
def black_scholes_vec(S, K, T, r, sigma, option_type='C'):
    # S: Spot, K: Strike, T: Time (years), r: Risk-free, sigma: Vol
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'C':
        delta = norm.cdf(d1)
        price = S * delta - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    # Greeks
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * (norm.cdf(d2 if option_type == 'C' else -d2))
    
    return price, delta, gamma, vega, theta

def extract_cp_from_symbol(symbol):
    s = str(symbol).replace(" ", "")
    if len(s) > 9:
        cp = s[9]
        if cp in ['C', 'P']: return cp
    if 'C' in s[3:]: return 'C'
    if 'P' in s[3:]: return 'P'
    return 'C'

def construct_foundation_1m():
    spot_path = r"C:\SPYOptionTrader_test\data\spot\SPY_1.csv"
    opts_path = r"C:\SPYOptionTrader_test\data\alpaca_options\spy_options_intraday_large_with_greeks_m1.csv"
    output_path = r"C:\SPYOptionTrader_test\data\processed\mamba_institutional_1m.csv"

    print(f"--- [1/4] Loading Spot Timeline (1m) ---")
    df_spot = pd.read_csv(spot_path)
    df_spot['dt'] = pd.to_datetime(df_spot['timestamp'], utc=True)
    
    # Pre-compute indicators on spot (before duplication)
    print("Computing 1m Indicators (Pd 12)...")
    df_spot['rsi'] = ta.rsi(df_spot['close'], length=12)
    df_spot['atr'] = ta.atr(df_spot['high'], df_spot['low'], df_spot['close'], length=12)
    adx_df = ta.adx(df_spot['high'], df_spot['low'], df_spot['close'], length=12)
    df_spot['adx'] = adx_df.iloc[:, 0] if adx_df is not None else np.nan
    bbands = ta.bbands(df_spot['close'], length=12)
    if bbands is not None:
        df_spot['bb_lower'] = bbands.iloc[:, 0]
        df_spot['bb_upper'] = bbands.iloc[:, 2]
    df_spot['stoch_k'] = ta.stoch(df_spot['high'], df_spot['low'], df_spot['close'], k=12, d=3).iloc[:, 0]
    df_spot['sma'] = ta.sma(df_spot['close'], length=12)
    psar = ta.psar(df_spot['high'], df_spot['low'], df_spot['close'])
    if psar is not None:
        df_spot['psar'] = psar.iloc[:, 0].fillna(psar.iloc[:, 1])
        df_spot['psar_mark'] = np.where(psar.iloc[:, 0].isna(), 1.0, -1.0)
    
    df_spot = df_spot.ffill().bfill()

    # 100% UTC Anchor
    df_spot = df_spot.set_index('dt').sort_index()
    
    # Calculate IV Rank (2-year rolling context)
    # Since we have the whole 2 year spot file here, we calculate it on RV
    # which we'll use for synthesis and as a feature.
    print("Calculating IV Rank (RV Context)...")
    df_spot['log_ret'] = np.log(df_spot['close'] / df_spot['close'].shift(1))
    df_spot['rv_12'] = df_spot['log_ret'].rolling(window=12).std() * np.sqrt(252 * 390) 
    # IVR = (current RV - min RV) / (max RV - min RV)
    window = 252 * 390 # Full 2 year context
    df_spot['ivr'] = (df_spot['rv_12'] - df_spot['rv_12'].rolling(window, min_periods=1).min()) / \
                     (df_spot['rv_12'].rolling(window, min_periods=1).max() - df_spot['rv_12'].rolling(window, min_periods=1).min())
    df_spot['ivr'] = df_spot['ivr'].fillna(0.5) * 100.0 # Scale to 0-100
    
    # --- MULTI-HORIZON RISK TARGET: Max Drawdown (Next 60m) ---
    print("Computing Multi-Horizon Risk: Max DD (Next 60m)...")
    # Max DD over next 60 periods: (Price_t - Min(Price_t+1..t+60)) / Price_t
    df_spot['future_min_60'] = df_spot['close'].shift(-60).rolling(window=60, min_periods=1).min()
    df_spot['max_dd_60m'] = (df_spot['close'] - df_spot['future_min_60']) / df_spot['close']
    df_spot['max_dd_60m'] = df_spot['max_dd_60m'].fillna(0).clip(lower=0) * 100.0 # Percentage

    # --- [2/4] Prepare Real Options (2025-2026) ---
    print(f"--- [2/4] Processing Real Options ---")
    if os.path.exists(opts_path):
        df_opts = pd.read_csv(opts_path)
        df_opts['dt'] = pd.to_datetime(df_opts['timestamp'], utc=True)
        # Extract CP if missing
        if 'call_put' not in df_opts.columns:
            df_opts['call_put'] = df_opts['symbol'].apply(extract_cp_from_symbol)
        
        # Cross-Sectional Selection (Top 50 C/P by Volume)
        print("Slicing Top 50 Liquid Contracts per Interval...")
        # Note: We group by DT and take top 50 C and top 50 P
        df_opts = df_opts.sort_values(['dt', 'volume'], ascending=[True, False])
        df_real_subset = df_opts.groupby(['dt', 'call_put']).head(50)
        print(f"Real Options processed: {len(df_real_subset):,} rows.")
    else:
        print("Real 1m Options file not found. Full synthesis will be used.")
        df_real_subset = pd.DataFrame()

    # --- [3/4] Synthetic Synthesis Engine (2024 Gap) ---
    print(f"--- [3/4] Synthesizing 2024 Surface ---")
    start_2024 = df_spot.index.min()
    end_2024 = pd.to_datetime("2025-01-01", utc=True)
    df_spot_2024 = df_spot[df_spot.index < end_2024].copy()
    
    # Calculate RV for synthesis scaling
    df_spot_2024['log_ret'] = np.log(df_spot_2024['close'] / df_spot_2024['close'].shift(1))
    df_spot_2024['rv'] = df_spot_2024['log_ret'].rolling(window=12).std() * np.sqrt(252 * 390) 
    
    synth_rows = []
    # Loop over 2024 minutes producing 100 rows each
    # For performance, we'll avoid a full loop and use vectorized BS soon if slow
    # but for 100k rows x 100 options, it's 10M rows.
    
    # Optimization: Chunked Synthesis
    # Generate 50 calls and 50 puts with DTE CYCLING (expiry decay)
    print(f"Synthesizing {len(df_spot_2024):,} intervals for 2024 with DTE Cycling...")
    
    # Vectorized Synthesis for 2024
    all_synth_dfs = []
    
    # We do 100 steps of strike offsets
    strike_offsets = np.linspace(-50, 50, 100)
    
    # DTE Cycling: Cycle through institutional DTE points [2, 7, 14, 30, 45]
    # This prevents the model from learning a static decay slope
    dtes = np.tile([2/365, 7/365, 14/365, 30/365, 45/365], (len(df_spot_2024) // 5) + 1)[:len(df_spot_2024)]
    df_spot_2024['te'] = dtes
    
    for offset in strike_offsets:
        chunk = df_spot_2024.copy()
        chunk['strike'] = chunk['close'].round(0) + offset
        chunk['call_put'] = 'C' if offset > 0 else 'P'
        chunk['iv'] = (chunk['rv_12'] * 1.18).fillna(0.15)
        
        # BS Vectorized
        prices, deltas, gammas, vegas, thetas = black_scholes_vec(
            chunk['close'].values, chunk['strike'].values, chunk['te'].values, 
            0.04, chunk['iv'].values, chunk['call_put'].values[0]
        )
        chunk['price'] = prices
        chunk['delta'] = deltas
        chunk['gamma'] = gammas
        chunk['vega'] = vegas
        chunk['theta'] = thetas
        chunk['volume'] = chunk['volume'] * 0.05 # Theoretical liquid volume
        chunk['spread_ratio'] = 0.001 + (chunk['iv'] * 0.005) # Synthetic friction (wider spreads in high vol)
        
        all_synth_dfs.append(chunk.reset_index())

    df_synth_2024 = pd.concat(all_synth_dfs).sort_values(['dt', 'volume'], ascending=[True, False])
    print(f"Synthetic 2024 Surface: {len(df_synth_2024):,} rows.")

    # --- [4/4] Final Merging & Export ---
    print(f"--- [4/4] Final Integration ---")
    # Merge Spot features into Real Options
    if not df_real_subset.empty:
        df_real_integrated = df_real_subset.merge(df_spot.reset_index(), on='dt', how='inner', suffixes=('', '_spot'))
    else:
        df_real_integrated = pd.DataFrame()

    # Combine 2024 Synth with 2025 Real
    df_final = pd.concat([df_synth_2024, df_real_integrated], ignore_index=True)
    
    # Add spread ratio for real data (Fallback since Alpaca m1 lacks bid/ask)
    df_final['spread_ratio'] = df_final['spread_ratio'].fillna(0.002) # Institutional baseline 0.2%

    # Final cleanup and target
    df_spot['next_close'] = df_spot['close'].shift(-1)
    df_final = df_final.merge(df_spot[['next_close']], on='dt', how='left')
    df_final['target_spot'] = np.log(df_final['next_close'] / df_final['close']) * 100.0

    print("Writing Institutional Foundation to Disk...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Write only essential columns to manage size (20M rows)
    cols = ['dt', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'strike', 'call_put',
            'delta', 'gamma', 'vega', 'theta', 'iv', 'ivr', 'spread_ratio', 'te',
            'rsi', 'atr', 'adx', 'bb_lower', 'bb_upper', 'stoch_k', 'sma', 'psar', 'psar_mark',
            'target_spot', 'max_dd_60m']
    
    # Save in chunks if needed, but for now direct 
    df_final[cols].to_csv(output_path, index=False)
    
    print("="*60)
    print(f"SUCCESS: Cross-Sectional Institutional 1m Generated")
    print(f"Final Count: {len(df_final):,} rows.")
    print(f"Foundation Parity: 1 Spot Min -> ~100 Options Rows")
    print("="*60)

if __name__ == "__main__":
    construct_foundation_1m()
