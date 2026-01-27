import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime, timedelta
import scipy.stats as si

# --- CONFIG ---
SPOT_FILE = r"data/spot/SPY_1.csv"
IVOL_FILE = r"data/ivolatility/spy_options_ivol_large_clean.csv"
OUTPUT_DIR = r"data/processed/v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Risk-free rate assumption (approx 5% for 2024/25)
RISK_FREE_RATE = 0.045 

def black_scholes_price(S, K, T, r, sigma, option_type='C'):
    """
    Vectorized Black-Scholes pricing.
    S: Spot Price (array)
    K: Strike Price (array)
    T: Time to Maturity in Years (array)
    r: Risk-free rate (scalar)
    sigma: Implied Volatility (array)
    option_type: 'C' or 'P'
    """
    # handle small T
    T = np.maximum(T, 1e-6)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'C':
        price = (S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2))
    else:
        price = (K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1))
        
    return price

def calculate_greeks(S, K, T, r, sigma, option_type='C'):
    """Calculate Delta, Gamma, Theta, Vega, Rho"""
    T = np.maximum(T, 1e-6)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    N_d1 = si.norm.cdf(d1)
    N_d2 = si.norm.cdf(d2)
    n_d1 = si.norm.pdf(d1)
    
    if option_type == 'C':
        delta = N_d1
        theta = (- (S * sigma * n_d1) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * N_d2) / 365.0
        rho = (K * T * np.exp(-r * T) * N_d2) / 100.0
    else:
        delta = N_d1 - 1
        theta = (- (S * sigma * n_d1) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * si.norm.cdf(-d2)) / 365.0
        rho = (-K * T * np.exp(-r * T) * si.norm.cdf(-d2)) / 100.0
        
    gamma = n_d1 / (S * sigma * np.sqrt(T))
    vega = (S * np.sqrt(T) * n_d1) / 100.0
    
    return delta, gamma, theta, vega, rho

def load_data():
    print("⏳ Loading Data...")
    
    # 1. Spot Data (1-min)
    spot_df = pd.read_csv(SPOT_FILE)
    spot_df['timestamp'] = pd.to_datetime(spot_df['timestamp'], utc=True)
    # Convert to Eastern naive for easier math with daily dates? Or keep UTC? 
    # IVol file has dates like '1/2/2025'. We assume these are trade dates.
    spot_df['date'] = spot_df['timestamp'].dt.date
    print(f"   Spot Data: {len(spot_df):,} rows ({spot_df['timestamp'].min()} to {spot_df['timestamp'].max()})")
    
    # 2. Options Data (Daily) - ONLY FOR 2025
    if os.path.exists(IVOL_FILE):
        ivol_df = pd.read_csv(IVOL_FILE)
        # Parse 'trade_date_fetch' (e.g. 1/2/2025)
        # Note: CSV header has '`' as column 3?? "symbol,exchange,`,underlying_price..."
        # Let's inspect column names
        ivol_df.rename(columns={ivol_df.columns[2]: 'trade_date'}, inplace=True)
        ivol_df['trade_date'] = pd.to_datetime(ivol_df['trade_date']).dt.date
        ivol_df['expiration'] = pd.to_datetime(ivol_df['expiration']).dt.date
        print(f"   Options Data: {len(ivol_df):,} rows (Daily Snapshots)")
    else:
        print("⚠️ Options file not found. 2025 generation will fail.")
        ivol_df = pd.DataFrame()
        
    return spot_df, ivol_df

def generate_2024_synthetic(spot_df):
    """
    Vectorized Synthetic Generation for 2024.
    """
    print("\n🚀 Generating 2024 Synthetic Data (Pure Model - Vectorized)...")
    
    df_2024 = spot_df[spot_df['timestamp'].dt.year == 2024].copy()
    print(f"   2024 Spot Rows: {len(df_2024):,}")
    
    output_file = os.path.join(OUTPUT_DIR, "Spy_Options_2024_1m.csv")
    
    # Constants
    FIXED_IV = 0.15
    RISK_FREE = RISK_FREE_RATE
    DTO = 30 # Days to expiration
    T = DTO / 365.0
    
    # Prepare header
    # Added open,high,low,close (Underlying) before option symbol
    header = "timestamp,symbol,underlying_price,open,high,low,close,option_symbol,expiration,strike,call_put,bid,ask,iv,delta,gamma,theta,vega,rho,volume,open_interest\n"
    
    with open(output_file, 'w') as f:
        f.write(header)
        
        # Batch processing to manage memory
        BATCH_SIZE = 5000
        total_rows = len(df_2024)
        
        for i in tqdm(range(0, total_rows, BATCH_SIZE), desc="2024 Vectorized"):
            batch = df_2024.iloc[i : i+BATCH_SIZE]
            
            # Arrays for Spot and Ts
            S_arr = batch['close'].values  # Shape (N,)
            TS_arr = batch['timestamp'].values
            N = len(S_arr)
            
            # Generate Strikes Grid (Broadcasting)
            # Center on spot, +/- 25 strikes
            # We want shape (N, 50)
            offsets = np.arange(-25, 25, 1.0) # 50 offsets
            K_grid = np.round(S_arr[:, None]) + offsets # Shape (N, 50)
            
            # Replicate Spot to match grid
            S_grid = np.repeat(S_arr[:, None], 50, axis=1) # Shape (N, 50)
            
            # Calc Prices (Vectorized)
            # Calls
            d1 = (np.log(S_grid / K_grid) + (RISK_FREE + 0.5 * FIXED_IV ** 2) * T) / (FIXED_IV * np.sqrt(T))
            d2 = d1 - FIXED_IV * np.sqrt(T)
            
            nd1 = si.norm.cdf(d1)
            nd2 = si.norm.cdf(d2)
            n_d1_pdf = si.norm.pdf(d1)
            
            c_prices = S_grid * nd1 - K_grid * np.exp(-RISK_FREE * T) * nd2
            
            # Call Greeks
            c_delta = nd1
            c_gamma = n_d1_pdf / (S_grid * FIXED_IV * np.sqrt(T))
            c_theta = (- (S_grid * FIXED_IV * n_d1_pdf) / (2 * np.sqrt(T)) - RISK_FREE * K_grid * np.exp(-RISK_FREE * T) * nd2) / 365.0
            c_vega = (S_grid * np.sqrt(T) * n_d1_pdf) / 100.0
            c_rho = (K_grid * T * np.exp(-RISK_FREE * T) * nd2) / 100.0
            
            # Puts
            p_prices = K_grid * np.exp(-RISK_FREE * T) * si.norm.cdf(-d2) - S_grid * si.norm.cdf(-d1)
            
            # Put Greeks
            p_delta = nd1 - 1
            p_gamma = c_gamma # Same
            p_theta = (- (S_grid * FIXED_IV * n_d1_pdf) / (2 * np.sqrt(T)) + RISK_FREE * K_grid * np.exp(-RISK_FREE * T) * si.norm.cdf(-d2)) / 365.0
            p_vega = c_vega # Same
            p_rho = (-K_grid * T * np.exp(-RISK_FREE * T) * si.norm.cdf(-d2)) / 100.0
            
            # Formatting and Writing overhead
            # Can we speed this up? pandas.to_csv is slow for this.
            # Manual string formatting in loop or list comp
            
            # Expiry String
            # vectorizing date offset is tricky with numpy datetime64
            # Uniform expiry approx? 
            # We can just cache the expiry date string if it's relative?
            # Actually, `black_scholes_price` logic above assumes fixed T.
            # So the expiry date changes every day.
            
            # Let's iterate the batch natively now that math is done
            
            lines = []
            
            # Prepare constant parts
            exp_dates = list(pd.to_datetime(TS_arr) + timedelta(days=30))
            exp_strs = [e.strftime("%Y-%m-%d") for e in exp_dates]
            ticker_dates = [e.strftime("%y%m%d") for e in exp_dates]
            ts_strs = [str(ts) for ts in TS_arr]
            
            # Map full OHLC to strings or values
            # S_arr is 'close'
            O_arr = batch['open'].values
            H_arr = batch['high'].values
            L_arr = batch['low'].values
            # V_arr = batch['volume'].values? We used 100 as volume for options. Keeping option volume fixed.
            # But we need underlying OHLC columns named 'open','high','low','close' in output?
            # Or columns 'underlying_open', etc?
            # The precompute script looks for 'open', 'high', 'low', 'close'.
            # If we call them that, it implies Option OHLC?
            # Usually 'close' in an option file means Option Close.
            # But precompute extracts Spot from these.
            # Let's add them as 'open','high','low','close' (Underlying) but keep 'bid','ask' for Option.
            # Note: The schema in precompute expects 'close' to be underlying if it drops duplicates?
            # No. precompute: "Identify OHLCV columns (same for all strikes)... Spot extract... duplicate drop"
            # So yes, it expects these columns to represent the Spot.
            
            for r in range(N):
                ts_str = ts_strs[r]
                spot_val = S_arr[r]
                o_val = O_arr[r]
                h_val = H_arr[r]
                l_val = L_arr[r]
                exp_date_str = exp_strs[r]
                tick_date = ticker_dates[r]
                
                for k_idx in range(50):
                    strike_val = K_grid[r, k_idx]
                    
                    # CALL
                    line_c = f"{ts_str},SPY,{spot_val:.2f},{o_val:.2f},{h_val:.2f},{l_val:.2f},{spot_val:.2f},SPY{tick_date}C{int(strike_val*1000):08d},{exp_date_str},{strike_val:.1f},C,{c_prices[r, k_idx]:.2f},{c_prices[r, k_idx]:.2f},{FIXED_IV:.4f},{c_delta[r, k_idx]:.4f},{c_gamma[r, k_idx]:.4f},{c_theta[r, k_idx]:.4f},{c_vega[r, k_idx]:.4f},{c_rho[r, k_idx]:.4f},100,1000\n"
                    lines.append(line_c)
                    
                    # PUT
                    line_p = f"{ts_str},SPY,{spot_val:.2f},{o_val:.2f},{h_val:.2f},{l_val:.2f},{spot_val:.2f},SPY{tick_date}P{int(strike_val*1000):08d},{exp_date_str},{strike_val:.1f},P,{p_prices[r, k_idx]:.2f},{p_prices[r, k_idx]:.2f},{FIXED_IV:.4f},{p_delta[r, k_idx]:.4f},{p_gamma[r, k_idx]:.4f},{p_theta[r, k_idx]:.4f},{p_vega[r, k_idx]:.4f},{p_rho[r, k_idx]:.4f},100,1000\n"
                    lines.append(line_p)
            
            f.writelines(lines)

def generate_2025_hybrid(spot_df, ivol_df):
    """
    Hybrid Generation for 2025 (Vectorized by Day).
    """
    print("\n🚀 Generating 2025 Hybrid Data (Spot + Daily IV - Vectorized)...")
    
    output_file = os.path.join(OUTPUT_DIR, "Spy_Options_2025_1m.csv")
    
    # 1. Filter 2025 Spot
    df_2025 = spot_df[spot_df['timestamp'].dt.year == 2025].copy()
    unique_dates = df_2025['date'].unique() # sorted
    print(f"   2025 Spot Rows: {len(df_2025):,}")
    print(f"   Unique Trading Days: {len(unique_dates)}")
    
    # 2. Group Options by Date
    # Make sure we can lookup options by 'trade_date' (date object)
    # ivol_df['trade_date'] is already date object from load_data
    if ivol_df.empty:
        print("⚠️ No Options Data! Skipping 2025.")
        return

    options_by_date = dict(tuple(ivol_df.groupby('trade_date')))
    
    header = "timestamp,symbol,underlying_price,open,high,low,close,option_symbol,expiration,strike,call_put,bid,ask,iv,delta,gamma,theta,vega,rho,volume,open_interest\n"
    
    with open(output_file, 'w') as f:
        f.write(header)
        
        # Iterate by DAY
        for day in tqdm(unique_dates, desc="2025 Daily Batches"):
            # A. Get Spot for this day
            day_spot = df_2025[df_2025['date'] == day]
            if day_spot.empty: continue
            
            S_arr = day_spot['close'].values # (N_mins,)
            O_arr = day_spot['open'].values
            H_arr = day_spot['high'].values
            L_arr = day_spot['low'].values
            
            TS_arr = day_spot['timestamp'].values
            N_mins = len(S_arr)
            
            # B. Get Options for this day
            day_opts = options_by_date.get(day)
            if day_opts is None or day_opts.empty:
                # Fallback? No options for this day.
                continue
            
            # Filter Top 100 by OI
            day_opts = day_opts.sort_values('open_interest', ascending=False).head(100).copy()
            
            # Prepare vectors
            K_arr = day_opts['strike'].values # (M_opts,)
            Sigma_arr = day_opts['iv'].values
            Type_arr = day_opts['call_put'].values # 'C' or 'P'
            Sym_arr = day_opts['option_symbol'].values
            Exp_arr = day_opts['expiration'].values
            Vol_arr = day_opts['volume'].values
            OI_arr = day_opts['open_interest'].values
            
            # T Calculation (Time to Expiry)
            # Exp Date - Current Date. 
            # Technically T shrinks throughout the day, but usually Daily T is used.
            # Let's use Daily T for consistency with Daily IV.
            # T = (Expiry - TradeDate) / 365
            # We can compute T once per option (scalar per option)
            # Ensure day_opts['expiration'] is date object
            # day is date object
            # T Calculation (Time to Expiry)
            exp_series = pd.to_datetime(day_opts['expiration'])
            day_ts = pd.to_datetime(day)
            T_days = (exp_series - day_ts).dt.days
            T_arr = T_days.values / 365.0
            T_arr = np.maximum(T_arr, 1e-6) # prevent divide by zero
            
            M_opts = len(day_opts)
            
            # C. Broadcast (N_mins x M_opts)
            # We need to compute prices for every minute x every option
            
            # Spot Grid: (N, 1) -> (N, M)
            S_grid = S_arr[:, None]
            # K Grid: (1, M) -> (N, M)
            K_grid = K_arr[None, :]
            # T Grid: (1, M) -> (N, M)
            T_grid = T_arr[None, :]
            # Sigma Grid
            Sigma_grid = Sigma_arr[None, :]
            
            # Calculate d1, d2
            d1 = (np.log(S_grid / K_grid) + (RISK_FREE_RATE + 0.5 * Sigma_grid ** 2) * T_grid) / (Sigma_grid * np.sqrt(T_grid))
            d2 = d1 - Sigma_grid * np.sqrt(T_grid)
            
            # standard normal
            nd1 = si.norm.cdf(d1)
            nd2 = si.norm.cdf(d2)
            n_d1_pdf = si.norm.pdf(d1)
            
            # Pre-allocate output arrays
            Price_grid = np.zeros((N_mins, M_opts))
            Delta_grid = np.zeros((N_mins, M_opts))
            Gamma_grid = np.zeros((N_mins, M_opts))
            Theta_grid = np.zeros((N_mins, M_opts))
            Vega_grid = np.zeros((N_mins, M_opts))
            Rho_grid = np.zeros((N_mins, M_opts))
            
            # Masks
            is_call = (Type_arr == 'C') # (M,)
            is_put = ~is_call
            
            # --- CALLS ---
            if np.any(is_call):
                # subset columns where call
                c_mask = is_call[None, :] # broadcastable if needed, but easier to just index
                # Actually, simpler to Compute ALL as Calls then adjust Puts, OR just mask col-indices
                
                # We can just use the indices
                call_idxs = np.where(is_call)[0]
                
                # Prices
                Price_grid[:, is_call] = S_grid * nd1[:, is_call] - K_grid[:, is_call] * np.exp(-RISK_FREE_RATE * T_grid[:, is_call]) * nd2[:, is_call]
                
                # Delta
                Delta_grid[:, is_call] = nd1[:, is_call]
                
                # Gamma (Same for C/P)
                Gamma_grid[:, is_call] = n_d1_pdf[:, is_call] / (S_grid * Sigma_grid[:, is_call] * np.sqrt(T_grid[:, is_call]))
                
                # Theta
                term1 = (- (S_grid * Sigma_grid[:, is_call] * n_d1_pdf[:, is_call]) / (2 * np.sqrt(T_grid[:, is_call])))
                term2 = (- RISK_FREE_RATE * K_grid[:, is_call] * np.exp(-RISK_FREE_RATE * T_grid[:, is_call]) * nd2[:, is_call])
                Theta_grid[:, is_call] = (term1 + term2) / 365.0
                
                # Vega (Same for C/P)
                Vega_grid[:, is_call] = (S_grid * np.sqrt(T_grid[:, is_call]) * n_d1_pdf[:, is_call]) / 100.0
                
                # Rho
                Rho_grid[:, is_call] = (K_grid[:, is_call] * T_grid[:, is_call] * np.exp(-RISK_FREE_RATE * T_grid[:, is_call]) * nd2[:, is_call]) / 100.0

            # --- PUTS ---
            if np.any(is_put):
                # Puts
                Price_grid[:, is_put] = K_grid[:, is_put] * np.exp(-RISK_FREE_RATE * T_grid[:, is_put]) * si.norm.cdf(-d2[:, is_put]) - S_grid * si.norm.cdf(-d1[:, is_put])
                
                Delta_grid[:, is_put] = nd1[:, is_put] - 1
                
                Gamma_grid[:, is_put] = n_d1_pdf[:, is_put] / (S_grid * Sigma_grid[:, is_put] * np.sqrt(T_grid[:, is_put]))
                
                term1 = (- (S_grid * Sigma_grid[:, is_put] * n_d1_pdf[:, is_put]) / (2 * np.sqrt(T_grid[:, is_put])))
                term2 = (+ RISK_FREE_RATE * K_grid[:, is_put] * np.exp(-RISK_FREE_RATE * T_grid[:, is_put]) * si.norm.cdf(-d2[:, is_put]))
                Theta_grid[:, is_put] = (term1 + term2) / 365.0
                
                Vega_grid[:, is_put] = (S_grid * np.sqrt(T_grid[:, is_put]) * n_d1_pdf[:, is_put]) / 100.0
                
                Rho_grid[:, is_put] = (-K_grid[:, is_put] * T_grid[:, is_put] * np.exp(-RISK_FREE_RATE * T_grid[:, is_put]) * si.norm.cdf(-d2[:, is_put])) / 100.0
            
            # D. Writing to file locally
            # Construct a DataFrame for this day?
            # Or formatted strings?
            # Constructing a DF is cleaner but memory heavy if N*M is large (e.g. 400 * 100 = 40,000 rows).
            # 40k rows is tiny.
            
            # However, we need to flatten the grids.
            # Repeat Minute Metadata for each Option
            # Repeat Option Metadata for each Minute
            
            # Flatten strategy:
            # We want row 0 (min 0) with all M options, then row 1...
            # Or row 0 min 0 opt 0, row 0 min 0 opt 1...
            # reshape?
            
            # Let's create a DataFrame
            # M_opts rows repeated N_mins times? No, N_mins repeated M times?
            
            # "Long" format:
            # timestamp | symbol | underlying | opt_symbol | ...
            # We have N mins. We have M options. Result N*M rows.
            
            # Repeat timestamps N times for each option?
            # tiles:
            # timestamps: [t0, t0... M times, t1, t1... M times] -> np.repeat(TS_arr, M)
            # strikes: [k0, k1, .. kM, k0, k1...] -> np.tile(K_arr, N)
            
            # Ensure sorting order matches the flattened grids
            # We computed grids as (N, M). Flattening default is C-style (row by row).
            # Row 0 (all options), Row 1 (all options).
            # So timestamps should be repeated: t0, t0, t0...
            # Options should be tiled: opt1, opt2... opt1, opt2...
            
            flat_ts = np.repeat(TS_arr, M_opts)
            flat_spot = np.repeat(S_arr, M_opts)
            flat_o = np.repeat(O_arr, M_opts)
            flat_h = np.repeat(H_arr, M_opts)
            flat_l = np.repeat(L_arr, M_opts)
            
            flat_opt_sym = np.tile(Sym_arr, N_mins)
            flat_exp = np.tile(Exp_arr, N_mins)
            flat_strike = np.tile(K_arr, N_mins)
            flat_type = np.tile(Type_arr, N_mins)
            flat_vol = np.tile(Vol_arr, N_mins)
            flat_oi = np.tile(OI_arr, N_mins)
            flat_iv = np.tile(Sigma_arr, N_mins)
            
            flat_price = Price_grid.flatten() # (N*M,)
            flat_delta = Delta_grid.flatten()
            flat_gamma = Gamma_grid.flatten()
            flat_theta = Theta_grid.flatten()
            flat_vega = Vega_grid.flatten()
            flat_rho = Rho_grid.flatten()
            
            # Create DF
            batch_df = pd.DataFrame({
                'timestamp': flat_ts,
                'symbol': 'SPY',
                'underlying_price': flat_spot.round(2),
                'open': flat_o.round(2),
                'high': flat_h.round(2),
                'low': flat_l.round(2),
                'close': flat_spot.round(2),
                'option_symbol': flat_opt_sym,
                'expiration': flat_exp,
                'strike': flat_strike,
                'call_put': flat_type,
                'bid': flat_price.round(2),
                'ask': flat_price.round(2),
                'iv': flat_iv.round(4),
                'delta': flat_delta.round(4),
                'gamma': flat_gamma.round(4),
                'theta': flat_theta.round(4),
                'vega': flat_vega.round(4),
                'rho': flat_rho.round(4),
                'volume': flat_vol,
                'open_interest': flat_oi
            })
            
            # CSV Write
            # Header only on first write? No, we opened file in 'w' mode outside loop.
            # Wait, we opened file outside loop.
            # We need to write header only once.
            
            # to_csv append mode?
            # We are inside `with open(...) as f`.
            # Pandas can write to file handle? Yes.
            
            batch_df.to_csv(f, header=False, index=False)


if __name__ == "__main__":
    spot, ivol = load_data()
    # Phase 1: 2024 (Regenerate to update Schema)
    # f24 = os.path.join(OUTPUT_DIR, "Spy_Options_2024_1m.csv")
    generate_2024_synthetic(spot)
        
    # Phase 2: 2025
    generate_2025_hybrid(spot, ivol)
