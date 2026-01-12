import pandas as pd
import numpy as np
import pandas_ta as ta
import os

def extract_cp(symbol):
    # Robust extraction of C or P from symbol
    # Compact: SPY250725C00613000 -> typically index 9 or 10
    # Spaced: SPY   250131C00455000 -> typically index 13
    s = str(symbol).replace(" ", "")
    # After removing spaces, format is usually SPY250131C...
    # Char at index 9: SPY (3) + 250131 (6) = 9
    if len(s) > 9:
        cp = s[9]
        if cp in ['C', 'P']:
            return cp
    # Fallback search
    if 'C' in s[3:]: return 'C'
    if 'P' in s[3:]: return 'P'
    return 'C'

def construct_foundation():
    spot_path = r"C:\SPYOptionTrader_test\data\spot\SPY_15.csv"
    # User might be referring to this one too
    opts_path = r"C:\SPYOptionTrader_test\data\alpaca_options\spy_options_intraday_large_with_greeks_m15.csv"
    output_path = r"C:\SPYOptionTrader_test\data\processed\mamba_foundation_15m.csv"

    print(f"Loading Master Spot Timeline: {spot_path}")
    df_spot = pd.read_csv(spot_path)
    df_spot['dt'] = pd.to_datetime(df_spot['timestamp'], utc=True)
    
    # 1. Prepare Real Options Metadata (2025-2026)
    print(f"Loading Real Options Metadata: {opts_path}")
    df_opts = pd.read_csv(opts_path)
    df_opts['dt'] = pd.to_datetime(df_opts['timestamp'], utc=True)
    df_opts['delta_abs'] = df_opts['delta_intraday'].abs()
    
    print("Extracting Call/Put flags from symbols...")
    df_opts['call_put'] = df_opts['symbol'].apply(extract_cp)

    print("Aggregating Real Options to 15m Intervals...")
    agg = df_opts.groupby('dt').agg(
        opt_iv=('iv_intraday', lambda x: x[(df_opts.loc[x.index, 'delta_abs'] >= 0.4) & (df_opts.loc[x.index, 'delta_abs'] <= 0.6)].mean()),
        opt_total_vol=('volume', 'sum'),
        opt_put_iv=('iv_intraday', lambda x: x[df_opts.loc[x.index, 'call_put'] == 'P'].mean()),
        opt_call_iv=('iv_intraday', lambda x: x[df_opts.loc[x.index, 'call_put'] == 'C'].mean())
    ).dropna(how='all')
    
    agg['opt_skew'] = agg['opt_put_iv'] - agg['opt_call_iv']
    agg['opt_vol'] = agg['opt_total_vol']
    agg = agg[['opt_iv', 'opt_skew', 'opt_vol']]
    
    print(f"Real Options Data contains {len(agg)} intervals.")

    # 2. Synchronize with Spot
    print("Merging Real Options into Spot Timeline...")
    df_final = df_spot.merge(agg, on='dt', how='left')

    # 3. Synthetic Synthesis for 2024 Gap
    print("Starting Synthetic Synthesis for 2024 data gap...")
    
    # Calculate Realized Volatility for IV substitution
    df_final['log_ret'] = np.log(df_final['close'] / df_final['close'].shift(1))
    df_final['rv'] = df_final['log_ret'].rolling(window=12).std() * np.sqrt(252 * 26) # Annualized 15m (approx 26 bars/day)
    
    # Identify overlaps for ratio matching
    common = df_final.dropna(subset=['opt_iv', 'rv'])
    if not common.empty:
        ratio = common['opt_iv'].mean() / common['rv'].mean()
        avg_skew = common['opt_skew'].mean()
        avg_vol_ratio = common['opt_vol'].mean() / common['volume'].mean()
        print(f"Sync Results: IV/RV Ratio={ratio:.4f}, Avg Skew={avg_skew:.4f}")
    else:
        ratio = 1.2
        avg_skew = 0.005
        avg_vol_ratio = 0.1
        print("Warning: No overlap found. Using fallback heuristics.")

    # Fill 2024 Synthetic Values
    mask_2024 = df_final['opt_iv'].isna()
    df_final.loc[mask_2024, 'opt_iv'] = (df_final.loc[mask_2024, 'rv'] * ratio).fillna(0.15)
    df_final.loc[mask_2024, 'opt_skew'] = avg_skew
    df_final.loc[mask_2024, 'opt_vol'] = (df_final.loc[mask_2024, 'volume'] * avg_vol_ratio).fillna(0)

    # --- 4. LAYER ON RAW REALITY INDICATORS (Period 12) ---
    print("Computing Raw Technical Indicators on the 15m Foundation...")
    main_close, main_open, main_high, main_low, main_vol = 'close', 'open', 'high', 'low', 'volume'
    
    # 1. RSI (12)
    df_final['rsi'] = ta.rsi(df_final[main_close], length=12)
    # 2. ATR (12)
    df_final['atr'] = ta.atr(df_final[main_high], df_final[main_low], df_final[main_close], length=12)
    # 3. ADX (12)
    adx_df = ta.adx(df_final[main_high], df_final[main_low], df_final[main_close], length=12)
    df_final['adx'] = adx_df.iloc[:, 0] if adx_df is not None else np.nan
    # 4. BBands (12)
    bbands = ta.bbands(df_final[main_close], length=12)
    if bbands is not None:
        df_final['bb_lower'] = bbands.iloc[:, 0]
        df_final['bb_upper'] = bbands.iloc[:, 2]
    # 5. Stochastic (12)
    stoch = ta.stoch(df_final[main_high], df_final[main_low], df_final[main_close], k=12, d=3)
    df_final['stoch_k'] = stoch.iloc[:, 0] if stoch is not None else np.nan
    # 6. SMA (12)
    df_final['sma'] = ta.sma(df_final[main_close], length=12)
    # 7. PSAR
    psar = ta.psar(df_final[main_high], df_final[main_low], df_final[main_close])
    if psar is not None:
        df_final['psar'] = psar.iloc[:, 0].fillna(psar.iloc[:, 1])
        df_final['psar_mark'] = np.where(psar.iloc[:, 0].isna(), 'A', 'B')
    
    # Target (Always next-bar log returns)
    df_final['log_ret_hidden'] = np.log(df_final[main_close] / df_final[main_close].shift(1)) * 100.0
    df_final['target'] = df_final['log_ret_hidden'].shift(-1)
    
    # Fill indicator warmups
    df_final.fillna(method='ffill', inplace=True)
    df_final.fillna(method='bfill', inplace=True)
    
    # Final CSV - Selecting user-approved columns
    final_cols = [
        'dt', 'symbol', 'open', 'high', 'low', 'close', 'volume',
        'rsi', 'atr', 'adx', 'bb_lower', 'bb_upper', 'stoch_k', 'sma', 'psar',
        'psar_mark', 'opt_iv', 'opt_skew', 'opt_vol', 'target'
    ]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final[final_cols].to_csv(output_path, index=False)
    
    print("="*60)
    print(f"SUCCESS: Mamba Foundation 15m Generated")
    print(f"Output: {output_path}")
    print(f"Spot Rows: {len(df_spot):,}")
    print(f"Final Foundation Rows: {len(df_final):,}")
    print(f"Indicator Period: 12 (Rule of 12 Sync)")
    print("="*60)

    # Verification Parity Check
    if len(df_spot) == len(df_final):
        print("✅ Parity Check: Row counts match perfectly (1-to-1).")
    else:
        print(f"❌ Parity Error: {len(df_spot)} != {len(df_final)}")

if __name__ == "__main__":
    construct_foundation()
