import pandas as pd
import numpy as np
import pandas_ta as ta
import os

def generate_mamba_table(spot_file, options_file, output_file):
    print(f"Loading Spot Data: {spot_file}")
    df_spot = pd.read_csv(spot_file)
    
    print(f"Loading Options Data: {options_file}")
    df_opts = pd.read_csv(options_file)
    
    # 1. Spot Feature Engineering
    df = df_spot.copy()
    df['dt'] = pd.to_datetime(df['timestamp'], utc=True)

    # OHLC columns
    main_close = 'close'
    main_open = 'open'
    main_high = 'high'
    main_low = 'low'
    main_vol = 'volume'

    print("Computing Raw Technical Indicators (via pandas-ta)...")
    
    # 1. RSI (Native 0-100)
    df['rsi'] = ta.rsi(df[main_close], length=12)
    
    # 2. ATR (Native Price Range)
    df['atr'] = ta.atr(df[main_high], df[main_low], df[main_close], length=12)
    
    # 3. ADX (Native 0-100)
    adx_df = ta.adx(df[main_high], df[main_low], df[main_close], length=12)
    if adx_df is not None:
        df['adx'] = adx_df.iloc[:, 0]
    else:
        df['adx'] = np.nan

    # 4. Bollinger Bands (Native Prices)
    bbands = ta.bbands(df[main_close], length=12, std=2)
    if bbands is not None:
        df['bb_lower'] = bbands.iloc[:, 0]
        df['bb_upper'] = bbands.iloc[:, 2]
    else:
        df['bb_lower'] = np.nan
        df['bb_upper'] = np.nan

    # 5. Stochastic Oscillator (Native 0-100)
    stoch = ta.stoch(df[main_high], df[main_low], df[main_close], k=12, d=3)
    if stoch is not None:
        df['stoch_k'] = stoch.iloc[:, 0]
    else:
        df['stoch_k'] = np.nan

    # 6. SMA (Native Price)
    df['sma'] = ta.sma(df[main_close], length=12)

    # 7. Parabolic SAR (Native Price + Mark)
    psar = ta.psar(df[main_high], df[main_low], df[main_close])
    if psar is not None:
        # Col 0: PSARl (Long/Below), Col 1: PSARs (Short/Above)
        df['psar'] = psar.iloc[:, 0].fillna(psar.iloc[:, 1])
        # Mark: 'A' if PSAR is above High, 'B' if PSAR is below Low
        # More generally: 'A' if it's the "Short" PSAR (Col 1), 'B' if it's the "Long" PSAR (Col 0)
        df['psar_mark'] = np.where(psar.iloc[:, 0].isna(), 'A', 'B')
    else:
        df['psar'] = np.nan
        df['psar_mark'] = 'N'

    # 2. Options Aggregation (Native Units)
    print("Aggregating Options Market State...")
    df_opts['dt'] = pd.to_datetime(df_opts['timestamp'], utc=True)
    df_opts['delta_abs'] = df_opts['delta_intraday'].abs()
    
    agg = df_opts.groupby('dt').agg(
        opt_iv=('iv_intraday', lambda x: x[(df_opts.loc[x.index, 'delta_abs'] >= 0.4) & (df_opts.loc[x.index, 'delta_abs'] <= 0.6)].mean()),
        opt_total_vol=('volume', 'sum'),
        opt_put_iv=('iv_intraday', lambda x: x[df_opts.loc[x.index, 'call_put'] == 'P'].mean()),
        opt_call_iv=('iv_intraday', lambda x: x[df_opts.loc[x.index, 'call_put'] == 'C'].mean())
    ).fillna(method='ffill')
    
    agg['opt_skew'] = agg['opt_put_iv'] - agg['opt_call_iv']
    agg['opt_vol'] = agg['opt_total_vol']
    
    # Merge
    print("Merging Data Streams...")
    df = df.merge(agg[['opt_iv', 'opt_skew', 'opt_vol']], on='dt', how='inner')
    
    # Target (Next bar log returns for training objective)
    # Keeping log_ret calculation for the target only
    df['log_ret_hidden'] = np.log(df[main_close] / df[main_close].shift(1)) * 100.0
    df['target'] = df['log_ret_hidden'].shift(-1)
    
    df.dropna(inplace=True)
    
    # Selection: Raw Reality Columns
    final_cols = [
        'dt', 'symbol', 'open', 'high', 'low', 'close', 'volume',
        'rsi', 'atr', 'adx', 'bb_lower', 'bb_upper', 'stoch_k', 'sma', 'psar',
        'psar_mark', 'opt_iv', 'opt_skew', 'opt_vol', 'target'
    ]
    
    output_df = df[final_cols]
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output_df.to_csv(output_file, index=False)
    
    print("="*40)
    print(f"SUCCESS: Raw Reality Feature Table Generated")
    print(f"Output: {output_file}")
    print(f"Final Count: {len(output_df):,} rows")
    print(f"Columns: {final_cols}")
    print("="*40)
    
    # Display first 5 rows
    print("\nSAMPLE (First 5 rows):")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(output_df.head(5))

if __name__ == "__main__":
    generate_mamba_table(
        "data/spot/SPY_5.csv",
        "data/alpaca_options/spy_options_intraday_large_with_greeks_m5.csv",
        "data/processed/mamba_homework_5m.csv"
    )
