from tabulate import tabulate
import mplfinance as mpf
import os
import datetime as dt
import time # Added for profiling
import csv
from matplotlib.backends.backend_pdf import PdfPages
from data_factory.sync_engine import MTFSyncEngine
from core.dto import OptionQuote
from strategies.options_strategy import (
    build_condor, calc_condor_credit, 
    PositionState, regime_wing_width
)
from intelligence.regime_filter import classify_regime, MarketRegime
from core.risk_manager import RiskManager, PortfolioGreeks
from intelligence.fuzzy_engine import (
    calculate_mtf_membership,
    calculate_iv_membership,
    calculate_regime_membership,
    calculate_rsi_membership,
    calculate_adx_membership,
    calculate_bbands_membership,
    calculate_stoch_membership,
    calculate_volume_membership,
    calculate_sma_distance_membership,
    calculate_psar_membership,
    calculate_atr_stop_multiplier
)
from intelligence.execution.cost_model import estimate_entry_cost
from qtmf.models import TradeIntent
from qtmf.facade import benchmark_and_size

# Safe import for Mamba
try:
    from intelligence.mamba_engine import HAS_MAMBA
except ImportError:
    HAS_MAMBA = False


# ==========================================================
# MARKET REALISM HELPERS (Stage 1)
# ==========================================================

def compute_realized_vol(prices: pd.Series, window: int = 20) -> float:
    """Compute annualized realized volatility from 5m bar returns.
    
    Args:
        prices: Series of close prices
        window: Lookback window for volatility calculation
        
    Returns:
        Annualized volatility as a decimal (e.g., 0.20 for 20%)
    """
    if len(prices) < window + 1:
        return 0.20  # Default 20% if insufficient data
    
    returns = prices.pct_change().dropna()
    if len(returns) < window:
        return 0.20
    
    std = returns.tail(window).std()
    # Annualize: sqrt(bars_per_year) where 5m bars = 78/day * 252 days = 19,656
    annualized = std * np.sqrt(19656)
    return float(annualized) if not np.isnan(annualized) else 0.20


def compute_iv_rank_proxy(current_vol: float, vol_history: list, lookback: int = 252) -> float:
    """Compute IV Rank as percentile of current vol vs trailing history.
    
    Args:
        current_vol: Current realized volatility
        vol_history: List of historical volatility values
        lookback: Number of periods to look back (default 252 ~= 1 year of daily)
        
    Returns:
        IV Rank as percentage 0-100
    """
    if len(vol_history) < 10:
        return 50.0  # Default mid-range
    
    history = vol_history[-lookback:] if len(vol_history) >= lookback else vol_history
    sorted_vols = sorted(history)
    rank = sum(1 for v in sorted_vols if v <= current_vol)
    ivr = (rank / len(sorted_vols)) * 100
    return round(ivr, 2)


def load_intraday_options(file_path: str, start_date=None, end_date=None) -> dict:
    """
    Load intraday options data with Greeks from CSV.
    Optimized for fast lookup by timestamp.
    
    Structure: {timestamp: {symbol: {data}}}
    """
    if not os.path.exists(file_path):
        print(f"[ERROR] Intraday options file not found: {file_path}")
        return {}

    print(f"[Data] Loading intraday options: {file_path}...")
    
    # Load specific columns to save memory
    use_cols = [
        'timestamp', 'symbol', 'expiration', 'strike', 'option_type', 
        'close', 'delta_intraday', 'theta_intraday', 'gamma_intraday', 
        'vega_intraday', 'iv_intraday'
    ]
    
    # Read CSV
    df = pd.read_csv(file_path, parse_dates=['timestamp', 'expiration'])
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Standardize 'call_put' or 'option_type' to 'option_type'
    if 'call_put' in df.columns:
        df.rename(columns={'call_put': 'option_type'}, inplace=True)
        
    # Normalize option_type (C/P -> call/put)
    if 'option_type' in df.columns:
        df['option_type'] = df['option_type'].replace({'C': 'call', 'P': 'put'})
    
    # Filter by date range
    if start_date:
        df = df[df['timestamp'] >= pd.Timestamp(start_date, tz='UTC')]
    if end_date:
        df = df[df['timestamp'] <= pd.Timestamp(end_date, tz='UTC')]
        
    print(f"[Data] Loaded {len(df):,} intraday option records.")
    
    # Index by timestamp for fast backtest lookup
    # Group by timestamp -> dict of symbols
    options_by_time = {}
    
    # Optimize grouping
    for timestamp, group in df.groupby('timestamp'):
        # Localize to None to match backtrader (if needed)
        ts_key = timestamp.replace(tzinfo=None) if timestamp.tzinfo else timestamp
        
        # Convert group to dict of dicts: {symbol: {fields}}
        group_dict = {}
        # Use to_dict('records') for safer column access
        for row in group.to_dict('records'):
            sym = row['symbol']
            # Expiration might be string or timestamp
            expiration = row['expiration']
            if hasattr(expiration, 'date'):
                expiration = expiration.date()
            
            group_dict[sym] = {
                'price': row['close'],
                'strike': row['strike'],
                'expiration': expiration,
                'type': row['option_type'],
                'delta': row['delta_intraday'],
                'theta': row['theta_intraday'],
                'gamma': row['gamma_intraday'],
                'vega': row['vega_intraday'],
                'iv': row['iv_intraday']
            }
        options_by_time[ts_key] = group_dict
        
    return options_by_time




def run_backtest_headless(s_cfg: StrategyConfig, r_cfg: RunConfig, preloaded_df=None, preloaded_options=None, preloaded_sync=None, preloaded_neural_forecasts=None, verbose=True):
    """
    Run backtest without generating reports, returning raw strategy object
    Used for optimization loops.
    
    Args:
        preloaded_neural_forecasts: Pre-computed neural forecasts to skip Mamba init (for optimizer)
    """
    class PandasData(bt.feeds.PandasData):
        params = (
            ('datetime', None),
            ('open', 'open'),
            ('high', 'high'),
            ('low', 'low'),
            ('close', 'close'),
            ('volume', 'volume'),
            ('openinterest', None),
        )

    # === Load Data ===
    if preloaded_df is not None:
        df = preloaded_df
    else:
        # Fallback to loading from disk (standard backtest behavior)
        csv_path = os.path.join("data", "spot", f"{s_cfg.underlying}_1.csv")
        if not os.path.exists(csv_path):
            print(f"[ERROR] Data not found: {csv_path}")
            return None
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        
        # Filter by date range if specified
        if hasattr(r_cfg, 'backtest_start') and r_cfg.backtest_start:
            start_dt = pd.Timestamp(r_cfg.backtest_start)
            df = df[df.index >= start_dt]
        if hasattr(r_cfg, 'backtest_end') and r_cfg.backtest_end:
            end_dt = pd.Timestamp(r_cfg.backtest_end) + pd.Timedelta(days=1)  # Include end day
            df = df[df.index < end_dt]
        
        # Optimization: Slice data if backtest_samples is set
        if r_cfg.backtest_samples and r_cfg.backtest_samples > 0:
            if len(df) > r_cfg.backtest_samples:
                df = df.iloc[-r_cfg.backtest_samples:]
        
        total_bars = len(df)
        if verbose: print(f"[Data] Loaded {total_bars:,} bars from {df.index.min().date()} to {df.index.max().date()}")
    
    # === Load Options Data (Intraday or Synthetic) ===
    options_df_raw = None # Keep for Mamba pre-compute
    if preloaded_options is not None:
        options_by_date = preloaded_options
        # ... detect intraday ...
        sample_keys = list(preloaded_options.keys())[:10]
        has_datetime_keys = any(isinstance(k, dt.datetime) for k in sample_keys)
        is_intraday = has_datetime_keys or getattr(r_cfg, 'use_intraday_data', False)
        total_bars = len(df) if df is not None else 0
    else:
        # Use explicit path if provided, otherwise fallback to defaults
        data_path = r_cfg.options_data_path if r_cfg.options_data_path and os.path.exists(r_cfg.options_data_path) else os.path.join("data", "alpaca_options", "spy_options_intraday_with_greeks.csv")
        synthetic_path = os.path.join("data", "synthetic_options", f"{s_cfg.underlying.lower()}_options_marks.csv")
        
        # Detect format
        is_synthetic_format = False
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                header = f.readline().lower()
                is_synthetic_format = 'option_symbol' in header and 'expiration' in header
        
        use_intraday = os.path.exists(data_path) and not is_synthetic_format and getattr(r_cfg, 'prefer_intraday', True)
        
        if use_intraday:
            if verbose: print(f"[Data] Using Intraday Options Data: {data_path}")
            options_df_raw = pd.read_csv(data_path, parse_dates=['timestamp', 'expiration'])
            options_by_date = load_intraday_options(data_path, r_cfg.backtest_start, r_cfg.backtest_end)
            is_intraday = True
        elif os.path.exists(data_path) and is_synthetic_format:
            if verbose: print(f"[Data] Using Synthetic Options Data: {data_path}")
            options_df_raw = pd.read_csv(data_path, parse_dates=["date", "expiration"])
            # Filter
            if hasattr(r_cfg, 'backtest_start') and r_cfg.backtest_start:
                options_df_raw = options_df_raw[options_df_raw['date'].dt.date >= pd.Timestamp(r_cfg.backtest_start).date()]
            if hasattr(r_cfg, 'backtest_end') and r_cfg.backtest_end:
                options_df_raw = options_df_raw[options_df_raw['date'].dt.date <= pd.Timestamp(r_cfg.backtest_end).date()]
            
            options_by_date = {date.date(): group.to_dict('records') for date, group in options_df_raw.groupby('date')}
            is_intraday = False
        else:
            # Synthetic Fallback
            options_df_raw = pd.read_csv(synthetic_path, parse_dates=["date", "expiration"]) if os.path.exists(synthetic_path) else None
            if options_df_raw is not None:
                options_by_date = {date.date(): group.to_dict('records') for date, group in options_df_raw.groupby('date')}
                is_intraday = False
            else:
                print(f"[ERROR] No options data found.")
                return None
    
    # === Pre-Calculate Indicators & Neural Forecasts (Batch Mode) ===
    if preloaded_neural_forecasts is not None:
        neural_forecasts = preloaded_neural_forecasts
    elif getattr(s_cfg, 'use_mamba_model', False) and HAS_MAMBA and not df.empty:
        neural_forecasts = None
        try:
            from intelligence.mamba_engine import MambaForecastEngine
            d_model = getattr(s_cfg, 'mamba_d_model', 1024)
            layers = getattr(s_cfg, 'mamba_layers', 32) 
            mamba_engine = MambaForecastEngine(d_model=d_model, layers=layers)
            
            # Pass options_df_raw for 13-feature sync
            neural_forecasts = mamba_engine.precompute_all(df, options_df=options_df_raw, batch_size=4096)
            print(f"[BacktestEngine] Neural Signals Pre-Computed: {len(neural_forecasts)} (Features: {mamba_engine.model.input_proj.in_features if hasattr(mamba_engine.model, 'input_proj') else '?'})")
        except Exception as e:
            print(f"[Warning] Failed to batch-compute Mamba signals: {e}")
    else:
        neural_forecasts = None

    # === Initialize MTF Sync Engine ===
    if preloaded_sync is not None:
        sync_engine = preloaded_sync
    else:
        sync_engine = MTFSyncEngine(s_cfg.underlying, r_cfg.mtf_timeframes) if r_cfg.use_mtf else None

    # === Strategy Definition ===
    class IronCondorStrategy(bt.Strategy):
        params = dict(s_cfg=None, r_cfg=None, sync_engine=None, options_data=None, 
                      is_intraday=False, neural_forecasts=None, verbose=True, total_bars=0)

        def __init__(self):
            self.s_cfg = self.params.s_cfg
            self.r_cfg = self.params.r_cfg
            self.sync = self.params.sync_engine
            self.options_data = self.params.options_data
            self.is_intraday = self.params.is_intraday
            self.neural_forecasts = self.params.neural_forecasts # Cached DF
            self.verbose = self.params.verbose
            self.total_bars = self.params.total_bars
            
            # Performance Tracking
            self.pnl = 0.0
            self.wins = 0
            self.losses = 0
            self.trades = 0
            self.peak = r_cfg.backtest_cash
            self.drawdowns = []
            self.equity_series = []
            self.trade_log = []
            
            # High-Fidelity Tracking
            self.active_position = None # Handle 1 position at a time for simplicity
            self.current_unrealized_pnl = 0.0 # Track latest open P&L for reporting

            self.bars_since_trade = 100 
            self.price_cache = {} # Persistent cache for leg prices: {symbol: last_price}
            self.bar_count = 0
            self.last_progress_pct = -1
            
            # Log throttling for spammy filter messages
            self._last_credit_reject_date = None
            self._last_credit_reject_value = None
            
            # Regime Indicators (Stage 2)
            self.sma = bt.ind.SMA(self.datas[0], period=200)
            self.adx = bt.ind.ADX(self.datas[0], period=14)
            
            # Risk Manager (Stage 3)
            self.risk_manager = RiskManager(self.s_cfg)
            
            # Mamba is handled via cached self.neural_forecasts now

        def close_position(self, reason, cost, dt, pnl):
            self.pnl += pnl
            self.bars_since_trade = 0 # Cooling Period
            self.trades += 1
            if pnl > 0: self.wins += 1
            else: self.losses += 1
            
            # Log Logic for Analyzer
            trade_record = {
                "start": self.active_position.open_time,
                "end": dt,
                "amount": pnl, # Net PnL
                "reason": reason,
                "result": "win" if pnl > 0 else "loss",
                "metadata": self.active_position.metadata
            }
            self.trade_log.append(trade_record)
            
            if self.verbose:
                print(f"[Trade] Closed | Reason: {reason} | PnL: ${pnl:.2f} | CloseCost: ${cost:.2f}")

            self.active_position = None
            self.current_unrealized_pnl = 0.0

        def next(self):
            dt_now = self.datas[0].datetime.datetime(0)
            date_now = dt_now.date()
            
            # Progress output
            self.bar_count += 1
            if self.total_bars > 0:
                progress_pct = int((self.bar_count / self.total_bars) * 100)
                if self.verbose and progress_pct % 10 == 0 and progress_pct != self.last_progress_pct:
                    self.last_progress_pct = progress_pct
                    equity = self.r_cfg.backtest_cash + self.pnl
                    pos_status = "IN POSITION" if self.active_position else "SCANNING"
                    print(f"[Progress] {progress_pct:3d}% | Bar {self.bar_count:,}/{self.total_bars:,} | {dt_now.strftime('%Y-%m-%d %H:%M')} | Equity: ${equity:,.2f} | Trades: {self.trades} | {pos_status}")
            
            # 1. Update Basic Accounting
            current_equity = self.r_cfg.backtest_cash + self.pnl
            unrealized_pnl = self.current_unrealized_pnl if self.active_position else 0.0
            
            # Update Equity series (Critical for Drawdown/Sharpe)
            total_equity = current_equity + unrealized_pnl
            self.peak = max(self.peak, total_equity)
            self.drawdowns.append(self.peak - total_equity)
            self.equity_series.append(total_equity)

            # Heartbeat (Once per 500 bars if verbose)
            if self.verbose and self.bar_count % 500 == 0:
                print(f"      [Heartbeat] Bar {self.bar_count}/{self.total_bars} | Price: ${self.data.close[0]:.2f} | Time: {dt_now}")

            # 3. Entry Logic
            chain_records = []
            t_load_start = time.time() # Start Profile
            if self.is_intraday:
                # Intraday Lookup: Exact timestamp match
                # Convert backtrader datetime to pandas Timestamp (matching loaded data keys)
                dt_key = pd.Timestamp(dt_now).tz_localize(None)
                chain_dict = self.options_data.get(dt_key, {})
                if chain_dict:
                    # Convert dict of dicts to list of OptionQuote objects directly
                    # Structure: {symbol: {price, strike, expiration, type, delta...}}
                    quote_chain = []
                    for sym, data in chain_dict.items():
                        # OPTIMIZATION: Skip useless strikes (Delta < 0.005)
                        if abs(data.get('delta', 0) or 0) < 0.005:
                            continue
                        quote_chain.append(OptionQuote(
                            symbol=sym,
                            expiration=data['expiration'],
                            strike=data['strike'],
                            is_call=(data['type'] == 'call'),
                            bid=data['price'], # Use close as bid/ask/mid proxy for intraday
                            ask=data['price'],
                            mid=data['price'],
                            delta=data['delta'],
                            iv=data['iv'],
                            gamma=data.get('gamma', 0.0),
                            vega=data.get('vega', 0.0),
                            theta=data.get('theta', 0.0)
                        ))
                    chain_records = quote_chain # Use directly
            else:
                # EOD/Synthetic Lookup: Date match
                chain_records = self.options_data.get(date_now, [])
            
            # Debug: Print once on first attempt after cooldown
            if self.bar_count == 101 and self.verbose:
                print(f"[DEBUG] Mode: {'Intraday' if self.is_intraday else 'Synthetic'}")
                print(f"[DEBUG] Looking for: {dt_now if self.is_intraday else date_now}")
                print(f"[DEBUG] Lookup Key Type: {type(dt_key)}")
                print(f"[DEBUG] Records found: {len(chain_records)}")
                # Show sample keys from loaded data
                if self.options_data:
                    sample_keys = list(self.options_data.keys())[:5]
                    print(f"[DEBUG] Sample Data Keys: {sample_keys}")
                    if sample_keys:
                        print(f"[DEBUG] Sample Key Type: {type(sample_keys[0])}")
            
            if chain_records is None or (isinstance(chain_records, list) and not chain_records) or (isinstance(chain_records, pd.DataFrame) and chain_records.empty):
                return
            
            if not self.is_intraday:
                # Standard conversion for Synthetic data (DataFrame or list of records)
                if isinstance(chain_records, pd.DataFrame):
                    # Iterate dataframe efficiently using itertuples
                    quote_chain = []
                    seen_contracts = set() # Deduplication
                    for r in chain_records.itertuples():
                        # OPTIMIZATION 1: Skip useless strikes (Delta < 0.005)
                        if abs(getattr(r, 'delta', 0) or 0) < 0.005:
                            continue
                        
                        # Extract Key Params
                        strike = getattr(r, 'strike')
                        expiration = getattr(r, 'expiration').date()
                        is_call = (getattr(r, 'contract_type') == 'call')
                        
                        # OPTIMIZATION 2: Deduplicate
                        k = (expiration, strike, is_call)
                        if k in seen_contracts:
                            continue
                        seen_contracts.add(k)

                        # Handle potential missing columns with getattr
                        quote_chain.append(OptionQuote(
                            symbol=getattr(r, 'option_symbol'),
                            expiration=expiration,
                            strike=strike,
                            is_call=is_call,
                            bid=getattr(r, 'bid'),
                            ask=getattr(r, 'ask'),
                            mid=getattr(r, 'last_price'),
                            delta=getattr(r, 'delta'),
                            iv=getattr(r, 'implied_volatility'),
                            # Stage 3: Greeks for Risk Management
                            gamma=getattr(r, 'gamma', 0.0),
                            vega=getattr(r, 'vega', 0.0),
                            theta=getattr(r, 'theta', 0.0)
                        ))
                else:
                    # Legacy: List of dicts
                    quote_chain = []
                    seen_contracts = set() # Deduplication
                    for r in chain_records:
                        # OPTIMIZATION 1: Skip useless strikes
                        if abs(r.get('delta', 0) or 0) < 0.005:
                            continue

                        # Extract Key Params
                        strike = r['strike']
                        expiration = r['expiration'].date()
                        is_call = (r['contract_type'] == 'call')

                        # OPTIMIZATION 2: Deduplicate
                        k = (expiration, strike, is_call)
                        if k in seen_contracts:
                            continue
                        seen_contracts.add(k)

                        quote_chain.append(OptionQuote(
                            symbol=r['option_symbol'],
                            expiration=expiration,
                            strike=strike,
                            is_call=is_call,
                            bid=r['bid'],
                            ask=r['ask'],
                            mid=r['last_price'],
                            delta=r['delta'],
                            iv=r['implied_volatility'],
                            # Stage 3: Greeks for Risk Management
                            gamma=r.get('gamma', 0.0) or 0.0,
                            vega=r.get('vega', 0.0) or 0.0,
                            theta=r.get('theta', 0.0) or 0.0
                        ))

            
            t_load_end = time.time()
            if t_load_end - t_load_start > 0.05:
                print(f"[Slow Load] {self.data.datetime.date(0)}: {t_load_end - t_load_start:.4f}s | Records: {len(chain_records)}")

            # === EXIT LOGIC (Restored) ===
            if self.active_position:
                legs = self.active_position.legs
                
                # 0. Expiration Check (Critical Fix)
                # If we are past expiration, we must exit immediately.
                # Settlement is handled by intrinsic value if data is missing.
                # Use .date() to ensure compatibility with datetime.date (fixes Timestamp error)
                short_exp = legs.short_call.expiration
                if hasattr(short_exp, 'date'):
                    short_exp = short_exp.date()
                
                if date_now >= short_exp:
                    # Calculate Settlement Value (Intrinsic)
                    spot = self.data.close[0]
                    
                    # Call side
                    call_val = max(0, spot - legs.short_call.strike) - max(0, spot - legs.long_call.strike)
                    # Put side
                    put_val = max(0, legs.short_put.strike - spot) - max(0, legs.long_put.strike - spot)
                    
                    settlement_cost = call_val + put_val
                    unrealized_pnl = (self.active_position.credit_received - settlement_cost) * self.active_position.quantity * 100
                    
                    self.close_position("expired", settlement_cost, dt_now, unrealized_pnl)
                    return

                # Map current prices for active legs
                current_prices = {q.symbol: q.mid for q in quote_chain}
                syms = [legs.short_call.symbol, legs.long_call.symbol, legs.short_put.symbol, legs.long_put.symbol]
                
                # Check mark-to-market if we have prices
                if all(s in current_prices for s in syms):
                    # Calculate Cost to Close (Mark-to-Market)
                    cost = (current_prices[legs.short_call.symbol] + current_prices[legs.short_put.symbol]) - \
                           (current_prices[legs.long_call.symbol] + current_prices[legs.long_put.symbol])
                    
                    unrealized_pnl = (self.active_position.credit_received - cost) * self.active_position.quantity * 100
                    self.current_unrealized_pnl = unrealized_pnl
                    
                    exit_reason = None
                    # 1. Profit Take
                    target = (self.active_position.credit_received * self.active_position.quantity * 100 * self.s_cfg.profit_take_pct)
                    if unrealized_pnl >= target:
                        exit_reason = "profit_take"
                    
                    # 2. Stop Loss
                    elif cost >= (self.active_position.credit_received * (1 + self.s_cfg.loss_close_multiple)):
                        exit_reason = "stop_loss"
                        
                    # 3. Max Hold
                    elif (dt_now.replace(tzinfo=None) - self.active_position.open_time.replace(tzinfo=None)).days >= self.s_cfg.max_hold_days:
                         exit_reason = "max_hold"
                    
                    if exit_reason:
                        self.close_position(exit_reason, cost, dt_now, unrealized_pnl)
                        return
                    
                # If we are in a position, we skip entry logic entirely
                return

            self.bars_since_trade += 1
            if self.bars_since_trade < 100:
                if self.verbose and self.bar_count % 500 == 0:
                    print(f"      [Cooling] Bar {self.bar_count} | Cooldown: {self.bars_since_trade}/100")
                return

            # === Market Realism (Stage 1) ===
            # Calculate Realized Volatility
            vol_window = 20
            # Get enough history for calculation (window + 1 for diff)
            price_history = self.data.close.get(ago=0, size=vol_window+1)
            # Backtrader returns array.array or list, convert to Series
            price_series = pd.Series(price_history) if price_history else pd.Series()
            
            realized_vol = compute_realized_vol(price_series, window=vol_window)
            
            # Update Volatility History for IV Rank
            if not hasattr(self, 'vol_history'):
                self.vol_history = []
            self.vol_history.append(realized_vol)
            
            # Calculate IV Rank
            ivr = compute_iv_rank_proxy(realized_vol, self.vol_history)
            
            # Use realized vol as VIX proxy (scaled to %)
            vix = realized_vol * 100 
            
            if ivr < self.s_cfg.iv_rank_min or vix > self.s_cfg.vix_threshold:
                if self.verbose:
                    print(f"  [Filter] IVR/VIX gate: ivr={ivr:.1f} < {self.s_cfg.iv_rank_min} or vix={vix:.1f} > {self.s_cfg.vix_threshold}, skip entry")
                return

            # === ENHANCED ENTRY LOGIC WITH ALL INDICATORS ===
            mtf_snapshot = self.sync.get_snapshot(dt_now) if self.sync else None

            # Import all membership functions
            from intelligence.fuzzy_engine import (
                calculate_mtf_membership, calculate_iv_membership, calculate_regime_membership,
                calculate_rsi_membership, calculate_adx_membership, calculate_bbands_membership,
                calculate_volume_membership, calculate_sma_distance_membership, calculate_stoch_membership
            )

            # Existing indicators
            mu_mtf = calculate_mtf_membership(mtf_snapshot) if self.s_cfg.use_mtf_filter else 0.5
            mu_iv = calculate_iv_membership(ivr)
            mu_regime = calculate_regime_membership(vix, self.s_cfg.vix_threshold)

            # Get 5-minute snapshot for new indicators
            snapshot_5m = mtf_snapshot.get('5', {}) if mtf_snapshot else {}
            
            # Extract indicators centrally (safe defaults)
            rsi_current = snapshot_5m.get('rsi_14', None)
            adx_current = snapshot_5m.get('adx_14', None)
            stoch_k = snapshot_5m.get('stoch_k', None)
            bb_position = snapshot_5m.get('bb_position', None)
            bb_width = snapshot_5m.get('bb_width', None)
            volume_ratio = snapshot_5m.get('volume_ratio', None)
            sma_distance = snapshot_5m.get('sma_distance', None)
            psar_pos = snapshot_5m.get('psar_position', None)

            # RSI Filter
            if self.s_cfg.use_rsi_filter:
                mu_rsi = calculate_rsi_membership(
                    rsi_current,
                    self.s_cfg.rsi_neutral_min,
                    self.s_cfg.rsi_neutral_max
                )
                if mu_rsi < 0.3:
                    if self.verbose:
                        print(f"  [Filter] RSI too extreme ({rsi_current}), skip entry")
                    return
            else:
                mu_rsi = 0.5

            # ADX Filter
            if self.s_cfg.use_adx_filter:
                mu_adx = calculate_adx_membership(
                    adx_current,
                    self.s_cfg.adx_threshold_low,
                    self.s_cfg.adx_threshold_high
                )
                if mu_adx < 0.3:
                    if self.verbose:
                        print(f"  [Filter] ADX too high ({adx_current}), trending market, skip entry")
                    return
            else:
                mu_adx = 0.5

            # Stochastic Filter
            if self.s_cfg.use_stoch_filter:
                mu_stoch = calculate_stoch_membership(
                    stoch_k,
                    self.s_cfg.stoch_neutral_min,
                    self.s_cfg.stoch_neutral_max
                )
                if mu_stoch < 0.3:
                    if self.verbose:
                        print(f"  [Filter] Stochastic extreme ({stoch_k}), skip entry")
                    return
            else:
                mu_stoch = 0.5

            # Bollinger Bands Filter
            if self.s_cfg.use_bbands_filter:
                mu_bbands = calculate_bbands_membership(
                    bb_position,
                    bb_width,
                    self.s_cfg.bbands_squeeze_threshold
                )
                if mu_bbands < 0.3:
                    if self.verbose:
                        print(f"  [Filter] BB position extreme ({bb_position}), skip entry")
                    return
            else:
                mu_bbands = 0.5

            # Volume Filter
            if self.s_cfg.use_volume_filter:
                mu_volume = calculate_volume_membership(
                    volume_ratio,
                    self.s_cfg.volume_min_ratio
                )
                if mu_volume < 0.3:
                    if self.verbose:
                        print(f"  [Filter] Low volume ({volume_ratio}), skip entry")
                    return
            else:
                mu_volume = 0.5

            # SMA Distance Filter
            if self.s_cfg.use_sma_filter:
                mu_sma = calculate_sma_distance_membership(
                    sma_distance,
                    self.s_cfg.sma_max_distance
                )
                if mu_sma < 0.3:
                    if self.verbose:
                        print(f"  [Filter] Price too far from SMA ({sma_distance}), skip entry")
                    return
            else:
                mu_sma = 0.5

            # PSAR Filter (10th Indicator)
            if self.s_cfg.use_psar_filter:
                mu_psar = calculate_psar_membership(psar_pos)
                if mu_psar < 0.3:
                    if self.verbose:
                        print(f"  [Filter] Strong PSAR trend ({psar_pos}), skip entry")
                    return
            else:
                mu_psar = 0.5

            # Existing MTF consensus check
            if self.s_cfg.use_mtf_filter:
                if mu_mtf < self.s_cfg.mtf_consensus_min or mu_mtf > self.s_cfg.mtf_consensus_max:
                    if self.verbose:
                        print(f"  [Filter] MTF consensus {mu_mtf:.2f} outside range, skip entry")
                    return

            # Warmup Rule: Require minimum bars since start (prevents early NaN contamination)
            if len(self.equity_series) < 60:
                if self.verbose:
                    print(f"  [Filter] Warmup period (bar {len(self.equity_series)}/60), skip entry")
                return

            # === Regime Classification (Stage 2) ===
            # Use Backtrader indicators
            current_regime = MarketRegime.LOW_VOL_RANGE 
            
            is_trending = self.adx[0] > 25.0
            is_bullish = self.data.close[0] > self.sma[0]
            
            if vix > 35.0:
                current_regime = MarketRegime.CRASH_MODE
            elif is_trending:
                current_regime = MarketRegime.BULL_TREND if is_bullish else MarketRegime.BEAR_TREND
            elif vix > 20.0:
                current_regime = MarketRegime.HIGH_VOL_RANGE
            
            # Dynamic Wing Width based on Regime
            base_width = self.s_cfg.wing_width_min
            if current_regime in [MarketRegime.HIGH_VOL_RANGE, MarketRegime.CRASH_MODE]:
                width = min(self.s_cfg.wing_width_max, base_width + self.s_cfg.wing_increment_high)
            elif current_regime in [MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND]:
                width = min(self.s_cfg.wing_width_max, base_width + self.s_cfg.wing_increment_med)
            else:
                width = base_width

            condor = build_condor(quote_chain, self.s_cfg, date_now, self.data.close[0], width)
            
            if not condor:
                if self.verbose:
                    print(f"  [Filter] Condor build failed (width={width:.1f}, close={self.data.close[0]:.2f})")
                return

            credit = calc_condor_credit(condor)
            
            # === Adaptive Credit Threshold (Regime Dependent) ===
            # If IV is low, premium is cheap -> lower the bar
            base_ratio = self.s_cfg.min_credit_to_width
            if ivr < 20.0:
                required_ratio = max(0.12, base_ratio * 0.7) # Allow 30% reduction or 0.12 floor
            elif ivr > 50.0:
                required_ratio = min(0.35, base_ratio * 1.2) # Demand more premium in high IV
            else:
                required_ratio = base_ratio
                
            min_credit = width * required_ratio
            
            if credit < min_credit:
                # Throttle: only log once per day or if credit changed by >$0.10
                should_log = False
                if self._last_credit_reject_date != date_now:
                    should_log = True
                elif self._last_credit_reject_value is None or abs(credit - self._last_credit_reject_value) >= 0.10:
                    should_log = True
                
                if should_log and self.verbose:
                    print(f"  [Filter] Insufficient credit: need ${min_credit:.2f} (ratio {required_ratio:.2f}), got ${credit:.2f}")
                    self._last_credit_reject_date = date_now
                    self._last_credit_reject_value = credit
                return

            # === Neural Forecasting (Mamba 2) ===
            neural_forecast_data = None
            if self.neural_forecasts is not None and not self.neural_forecasts.empty:
                # Fast Lookup from Cache
                # Backtrader 'len(self)' gives current length. self.bar_count tracks it too.
                # Safest to use iloc with self.bar_count which we manually increment or len(self)-1
                idx = len(self) - 1
                if 0 <= idx < len(self.neural_forecasts):
                    row = self.neural_forecasts.iloc[idx]
                    neural_forecast_data = {
                        'model_backend': 'Mamba2-Batch',
                        'confidence': float(row['mamba_conf']),
                        'prob_bull': float(row['mamba_bull']),
                        'prob_bear': float(row['mamba_bear']),
                        'prob_neutral': float(row['mamba_neutral']),
                        'regime_vol': 0 # Optional
                    }
                    
                    if self.verbose and self.bar_count % 500 == 0:
                         print(f"[Neural:GPU] Conf={neural_forecast_data['confidence']:.2f} | Bull={neural_forecast_data['prob_bull']:.2f}")

            # === Sizing via QTMF Facade (Neuro-Fuzzy) ===
            # from qtmf.models import TradeIntent # Moved to top
            # from qtmf.facade import benchmark_and_size # Moved to top

            # Assign Neural Confidence to gaussian_confidence (for Facade Fusion)
            # Default to 0.5 (neutral) if no forecast available
            mamba_conf = neural_forecast_data['confidence'] if (neural_forecast_data and 'confidence' in neural_forecast_data) else 0.5

            # Construct Trade Intent with all 10 Indicators + Neural Signal
            intent = TradeIntent(
                symbol=self.s_cfg.underlying,
                action="SELL_CONDOR",
                gaussian_confidence=mamba_conf, # Component for Stage 2 Fusion
                current_price=self.data.close[0],
                vix=vix,
                ivr=ivr,
                realized_vol=0.0,
                mtf_snapshot=mtf_snapshot,
                # Indicator State
                rsi=rsi_current,
                adx=adx_current,
                bb_position=bb_position,
                bb_width=bb_width,
                stoch_k=stoch_k,
                volume_ratio=volume_ratio,
                sma_distance=sma_distance,
                psar_position=psar_pos, # Pass the pos from earlier calc
                # Neural Forecast
                neural_forecast=neural_forecast_data,
                extras={
                    'equity': self.broker.get_cash(),
                    'max_loss_per_contract': width * 100.0,
                    'risk_fraction': self.s_cfg.max_account_risk_per_trade,
                    'min_gaussian_confidence': getattr(self.s_cfg, 'min_gaussian_confidence', 0.20),
                    'fallback_total_qty': getattr(self.s_cfg, 'min_total_qty_for_iron_condor', 2),
                    'min_total_qty_for_two_wings': getattr(self.s_cfg, 'min_total_qty_for_iron_condor', 2),
                    # Pass config weights to facade
                    'w_mtf': getattr(self.s_cfg, 'fuzzy_weight_mtf', 0.10),
                    'w_iv': getattr(self.s_cfg, 'fuzzy_weight_iv', 0.10),
                    'w_regime': getattr(self.s_cfg, 'fuzzy_weight_regime', 0.08),
                    'w_rsi': getattr(self.s_cfg, 'fuzzy_weight_rsi', 0.05),
                    'w_adx': getattr(self.s_cfg, 'fuzzy_weight_adx', 0.05),
                    'w_bbands': getattr(self.s_cfg, 'fuzzy_weight_bbands', 0.15),
                    'w_volume': getattr(self.s_cfg, 'fuzzy_weight_volume', 0.12),
                    'w_sma': getattr(self.s_cfg, 'fuzzy_weight_sma', 0.10),
                    'w_stoch': getattr(self.s_cfg, 'fuzzy_weight_stoch', 0.15),
                    'w_psar': getattr(self.s_cfg, 'fuzzy_weight_psar', 0.10),
                    'fuzzy_weight_neural': getattr(self.s_cfg, 'fuzzy_weight_neural', 0.10)
                }
            )
            
            # Get Sizing Plan from Facade
            plan = benchmark_and_size(intent)
            
            if not plan.approved:
                if self.verbose:
                    print(f"  [Filter] QTMF rejected: {plan.reason} (Conf={gaussian_confidence:.2f})")
                return
                
            quantity = plan.total_qty

            
            if quantity > 0:
                # === Stage 3: Risk Management Gate ===
                current_equity = self.r_cfg.backtest_cash + self.pnl
                risk_approved, risk_reason = self.risk_manager.check_new_trade(
                    legs=condor,
                    quantity=quantity,
                    current_equity=current_equity
                )
                
                if not risk_approved:
                    if self.verbose:
                        print(f"  [Risk] Trade rejected: {risk_reason}")
                    return
                
                # === Apply Market Realism (Stage 1) ===
                use_exec_cost = getattr(self.r_cfg, "use_execution_cost_model", False)
                slippage_rate = getattr(self.r_cfg, "slippage_per_contract", 0.02)
                commission_rate = getattr(self.r_cfg, "commission_per_contract", 0.65)

                if use_exec_cost:
                    cost_result = estimate_entry_cost(
                        condor,
                        quantity,
                        self.r_cfg,
                        min_half_spread=getattr(self.r_cfg, "exec_cost_min_half_spread", 0.01),
                        impact_coeff=getattr(self.r_cfg, "exec_cost_impact_coeff", 0.10),
                    )
                    entry_slippage = cost_result.entry_slippage
                    entry_commission = cost_result.entry_commission
                    exec_cost_details = cost_result.details
                else:
                    # Slippage on entry (4 legs * qty contracts)
                    entry_slippage = slippage_rate * quantity * 4
                    # Commission on entry
                    entry_commission = commission_rate * quantity * 4
                    exec_cost_details = {}

                # Net credit after slippage (we receive less due to adverse fills)
                net_credit = credit - entry_slippage
                
                print(f"[Trade] Opened | Qty: {quantity} | Credit: ${credit:.2f} (net: ${net_credit:.2f}) | Exp: {condor.short_call.expiration}")
                entry_ohlcv = {
                    "open": self.data.open[0],
                    "high": self.data.high[0],
                    "low": self.data.low[0],
                    "close": self.data.close[0],
                    "volume": self.data.volume[0]
                }
                self.active_position = PositionState(
                    id=f"BT-{len(self.trade_log)+1}",
                    legs=condor,
                    open_time=dt_now,
                    credit_received=net_credit,  # Use net credit after slippage
                    quantity=quantity,
                    adjustments_done={"call":0, "put":0},
                    mtf_consensus=mu_mtf
                )
                self.active_position.metadata = {
                    "ivr": ivr,
                    "vix": vix,
                    "width": width,
                    "entry_ohlcv": entry_ohlcv,
                    "strikes": {
                        "short_call": condor.short_call.strike,
                        "long_call": condor.long_call.strike,
                        "short_put": condor.short_put.strike,
                        "long_put": condor.long_put.strike
                    },
                    # Stage 1: Market realism tracking
                    "gross_credit": credit,
                    "entry_slippage": entry_slippage,
                    "entry_commission": entry_commission,
                    "slippage_rate": slippage_rate,
                    "commission_rate": commission_rate,
                    "use_execution_cost_model": use_exec_cost,
                    "exec_cost_details": exec_cost_details
                }

    # === Run Backtest ===
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(r_cfg.backtest_cash)
    cerebro.adddata(PandasData(dataname=df))
    cerebro.addstrategy(IronCondorStrategy, s_cfg=s_cfg, r_cfg=r_cfg, sync_engine=sync_engine, 
                        options_data=options_by_date, is_intraday=is_intraday, 
                        neural_forecasts=neural_forecasts, verbose=verbose, total_bars=total_bars)
    
    # Add Analyzers
    # Run quietly
    results = cerebro.run(maxcpus=1)
    if not results:
        return None
        
    return results[0]  # Return strategy instance

def run_backtest_and_report(s_cfg: StrategyConfig, r_cfg: RunConfig):
    """Execute backtest with comprehensive reporting"""
    
    # Run Headless
    strat = run_backtest_headless(s_cfg, r_cfg)
    
    if strat is None:
        print("[ERROR] Backtest failed to run")
        return

    # Calculate Metrics from Strategy Object
    realized_pnl = strat.pnl
    open_trade_pnl = 0.0
    open_trades_count = 0
    
    # Mark-to-Market Open Position
    if strat.active_position:
        open_trades_count = 1
        # Use the PnL tracked by the strategy itself
        open_trade_pnl = getattr(strat, 'current_unrealized_pnl', 0.0)
        
        # Final Equity = Initial Cash + Realized + Unrealized
        final_equity = r_cfg.backtest_cash + realized_pnl + open_trade_pnl
    else:
        final_equity = r_cfg.backtest_cash + realized_pnl

    net_profit = final_equity - r_cfg.backtest_cash
    pct_return = (net_profit / r_cfg.backtest_cash) * 100.0
    
    # Date calculations - Use actual data window (not trade dates)
    data_start_date = strat.data.datetime.date(-len(strat.data)+1)
    data_end_date = strat.data.datetime.date(0)

    # Track trade activity window separately
    if strat.trade_log:
        trade_start_date = strat.trade_log[0]["start"].date()
        trade_end_date = strat.trade_log[-1]["end"].date()
    else:
        trade_start_date = None
        trade_end_date = None

    days = (data_end_date - data_start_date).days
    years = days / 365.0 if days > 0 else 0.0
    cagr = ((final_equity / r_cfg.backtest_cash) ** (1 / years) - 1) * 100.0 if years > 0 and final_equity > 0 else 0.0

    wins = [t for t in strat.trade_log if t["result"] == "win"]
    losses = [t for t in strat.trade_log if t["result"] == "loss"]
    
    avg_win = sum(t["amount"] for t in wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(t["amount"] for t in losses)) / len(losses) if losses else 0.0
    win_rate = (len(wins) / len(strat.trade_log)) * 100.0 if strat.trade_log else 0.0
    expectancy = (avg_win * (win_rate/100)) - (avg_loss * (1 - win_rate/100))
    
    # --- Manual Metrics Calculation (Since Simulation mode bypasses Broker) ---
    
    # 1. Drawdown
    # strat.drawdowns tracks [peak - equity], so max(drawdowns) is max_dd money
    max_dd_money = max(strat.drawdowns) if strat.drawdowns else 0.0
    max_dd_pct = (max_dd_money / strat.peak * 100.0) if hasattr(strat, 'peak') and strat.peak > 0 else 0.0
    
    # 2. Sharpe Ratio (Manual)
    # Estimate from Equity Series (High frequency data)
    sharpe = 0.0
    if len(strat.equity_series) > 1:
        equity_curve = np.array(strat.equity_series)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        if np.std(returns) > 0:
            # Annualize (Assume 5min bars -> 78 bars/day * 252 days = 19656 bars/year)
            # Adjust N based on actual data frequency if known, but 5min is dominant
            bars_per_year = 252 * 78 
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(bars_per_year)

    gross_win = sum(t["amount"] for t in wins)
    gross_loss = abs(sum(t["amount"] for t in losses))

    # Profit Factor: INF when wins exist but no losses; 0.0 when no activity
    if gross_loss > 0:
        profit_factor = gross_win / gross_loss
    elif gross_win > 0:
        profit_factor = float('inf')
    else:
        profit_factor = 0.0
    
    largest_win = max([t["amount"] for t in wins]) if wins else 0.0
    largest_loss = min([t["amount"] for t in losses]) if losses else 0.0
    
    net_pnl_dd_ratio = net_profit / max_dd_money if max_dd_money > 0 else 0.0

    # 3. Total Time in Trade
    total_seconds = sum((t["end"] - t["start"]).total_seconds() for t in strat.trade_log)
    m, s = divmod(total_seconds, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    time_in_trade_str = f"{int(d)}d {int(h):02d}:{int(m):02d}:{int(s):02d}"

    data_start_str = data_start_date.strftime("%Y-%m-%d")
    data_end_str = data_end_date.strftime("%Y-%m-%d")

    # Count open positions (positions in strat.positions that are still active)
    open_positions = len(strat.positions) if hasattr(strat, 'positions') and strat.positions else 0

    print("\n" + "="*60)
    print(f"BACKTEST RESULTS: {s_cfg.underlying}")
    print("="*60)
    print(f"Data Window:      {data_start_str} to {data_end_str} ({days} days)")
    print(f"Final Equity:     ${final_equity:,.2f}")
    print(f"Net Profit:       ${net_profit:,.2f} ({pct_return:.2f}%)")
    print(f"CAGR:             {cagr:.2f}%")
    print(f"Max Drawdown:     ${max_dd_money:,.2f} ({max_dd_pct:.2f}%)")
    print(f"Sharpe Ratio:     {sharpe:.2f}")
    print(f"Profit Factor:    {'INF' if profit_factor == float('inf') else f'{profit_factor:.2f}'}")
    print(f"Net PnL / DD:     {net_pnl_dd_ratio:.2f}")
    print(f"Time in Trade:    {time_in_trade_str}")
    print(f"Total Trades:     {len(strat.trade_log) + open_trades_count}")
    print(f"Open Trades:      {open_trades_count}")
    print(f"Win Rate:         {win_rate:.2f}%")
    print(f"Avg Win:          ${avg_win:,.2f}")
    print(f"Avg Loss:         ${avg_loss:,.2f}")
    print(f"Expectancy:       ${expectancy:.2f}")
    print(f"Largest Win:      ${largest_win:,.2f}")
    print(f"Largest Loss:     ${largest_loss:,.2f}")
    print(f"MTF Filter:       {'ENABLED' if s_cfg.use_mtf_filter else 'DISABLED'}")
    print("="*60)

    # Export Trades
    if strat.trade_log:
        trades_df = pd.DataFrame(strat.trade_log)
        out_csv = os.path.join("reports", "trades.csv")
        trades_df.to_csv(out_csv, index=False)
        print(f"[Export] Trade log saved: {out_csv}")
    
    # Export Baseline Metrics JSON
    import json
    from datetime import datetime
    baseline_metrics = {
        "timestamp": datetime.now().isoformat(),
        "data_window": f"{data_start_str} to {data_end_str}",
        "data_days": days,
        "net_profit": round(net_profit, 2),
        "net_profit_pct": round(pct_return, 2),
        "max_drawdown": round(max_dd_money, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "sharpe_ratio": round(sharpe, 2),
        "profit_factor": "INF" if profit_factor == float('inf') else round(profit_factor, 2),
        "net_pnl_dd_ratio": round(net_pnl_dd_ratio, 2),
        "total_trades": len(strat.trade_log),
        "open_trades": open_positions,
        "win_rate": round(win_rate, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "expectancy": round(expectancy, 2),
        "largest_win": round(largest_win, 2),
        "largest_loss": round(largest_loss, 2),
    }
    baseline_json_path = os.path.join("reports", "baseline_metrics.json")
    with open(baseline_json_path, 'w') as f:
        json.dump(baseline_metrics, f, indent=2)
    print(f"[Export] Baseline metrics saved: {baseline_json_path}")
    
    # Save PDF Report
    if r_cfg.backtest_plot:
        # Re-load DF just for plotting (inefficient but cleaner for now)
        csv_path = os.path.join("data", "spot", f"{s_cfg.underlying}_5.csv")
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        if r_cfg.backtest_samples and r_cfg.backtest_samples > 0:
            if len(df) > r_cfg.backtest_samples:
                df = df.iloc[-r_cfg.backtest_samples:]
    
        out_pdf = os.path.join("reports", "backtest_report.pdf")
        with PdfPages(out_pdf) as pdf:
            # Page 1: Metrics
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            ax.text(0.5, 0.95, f"Iron Condor Backtest Report: {s_cfg.underlying}", ha='center', fontsize=20, weight='bold')
            
            metrics = [
                ["Metric", "Value"],
                ["Data Window", f"{data_start_str} to {data_end_str}"],
                ["Initial Capital", f"${r_cfg.backtest_cash:,.2f}"],
                ["Final Equity", f"${final_equity:,.2f}"],
                ["Net Profit", f"${net_profit:,.2f} ({pct_return:.2f}%)"],
                ["Max Drawdown", f"${max_dd_money:,.2f} ({max_dd_pct:.2f}%)"],
                ["Sharpe Ratio", f"{sharpe:.2f}"],
                ["Profit Factor", "INF" if profit_factor == float('inf') else f"{profit_factor:.2f}"],
                ["Net PnL / DD", f"{net_pnl_dd_ratio:.2f}"],
                ["Time in Trade", f"{time_in_trade_str}"],
                ["Total Trades", f"{len(strat.trade_log)}"],
                ["Open Trades", f"{open_positions}"],
                ["Win Rate", f"{win_rate:.2f}%"],
                ["Avg Win", f"${avg_win:,.2f}"],
                ["Avg Loss", f"${avg_loss:,.2f}"],
                ["Largest Win", f"${largest_win:,.2f}"],
                ["Largest Loss", f"${largest_loss:,.2f}"],
                ["MTF Filter", "ENABLED" if s_cfg.use_mtf_filter else "DISABLED"],
                ["Liquidity Gate", "ENABLED" if s_cfg.use_liquidity_gate else "DISABLED"],
            ]

            
            table = tabulate(metrics, headers="firstrow", tablefmt="grid")
            ax.text(0.1, 0.5, table, family='monospace', fontsize=12)
            pdf.savefig(fig)
            plt.close(fig)
            
            # Page 2: Equity Curve
            fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
            
            axes[0].plot(strat.equity_series, label="Equity", color="blue")
            axes[0].set_title("Equity Curve")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

            axes[1].plot(strat.drawdowns, label="Drawdown", color="red", alpha=0.6)
            axes[1].fill_between(range(len(strat.drawdowns)), strat.drawdowns, 0, color='red', alpha=0.1)
            axes[1].set_title("Drawdown")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Page 3: Candlestick Chart with Strike Overlay
            window_size = 500
            start_idx = max(0, len(df) - window_size)
            df_window = df.iloc[start_idx:]
            
            # Create trade markers and strike segments
            buy_markers = pd.Series(np.nan, index=df_window.index)
            sell_markers = pd.Series(np.nan, index=df_window.index)
            alines = [] # [( (date1, price1), (date2, price1) ), ...]
            
            for t in strat.trade_log:
                if t["start"] in df_window.index:
                    buy_markers.loc[t["start"]] = t["entry_ohlcv"]["close"]
                if t["end"] in df_window.index:
                    sell_markers.loc[t["end"]] = t["exit_ohlcv"]["close"]
                
                # Add strike lines if either start or end is in window
                if (t["start"] in df_window.index) or (t["end"] in df_window.index):
                    plot_start = max(t["start"], df_window.index[0])
                    plot_end = min(t["end"], df_window.index[-1])
                    if plot_end >= plot_start:
                        s = t.get("strikes", {})
                        for val in s.values():
                            alines.append([ (plot_start, val), (plot_end, val) ])

            apds = []
            if not buy_markers.isna().all():
                apds.append(mpf.make_addplot(buy_markers, type="scatter", markersize=100, marker="^", color="green", label="Entry"))
            if not sell_markers.isna().all():
                apds.append(mpf.make_addplot(sell_markers, type="scatter", markersize=100, marker="v", color="red", label="Exit"))
            
            plot_kwargs = {
                'type': "candle",
                'style': "charles",
                'title': f"{s_cfg.underlying} Price Action (Last {len(df_window)} Bars)",
                'ylabel': "Price ($)",
                'volume': True,
                'returnfig': True,
                'figsize': (11, 8.5)
            }
            if apds:
                plot_kwargs['addplot'] = apds
            if alines:
                plot_kwargs['alines'] = dict(alines=alines, colors='gray', alpha=0.3, linewidths=0.5)
            
            try:
                fig, axlist = mpf.plot(df_window, **plot_kwargs)
                pdf.savefig(fig)
                plt.close(fig)
            except Exception as e:
                print(f"[Warning] Failed to generate candlestick chart: {e}")

            # Page 4: Performance Distributions & Heatmaps
            if strat.trade_log:
                fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
                
                # 1. P&L Distribution Histogram
                amounts = [t["amount"] for t in strat.trade_log]
                axes[0].hist(amounts, bins=min(len(amounts), 20), color="skyblue", edgecolor="black", alpha=0.7)
                axes[0].axvline(0, color="red", linestyle="--", alpha=0.5)
                axes[0].set_title("Realized P&L Distribution")
                axes[0].set_xlabel("Profit/Loss ($)")
                axes[0].set_ylabel("Frequency")
                
                # 2. Monthly Performance Table
                trades_df = pd.DataFrame(strat.trade_log)
                trades_df['month'] = trades_df['end'].dt.to_period('M')
                monthly_pnl = trades_df.groupby('month')['amount'].sum()
                
                if not monthly_pnl.empty:
                    m_data = [["Month", "P&L", "Trades"]]
                    for month, pnl in monthly_pnl.items():
                        m_count = len(trades_df[trades_df['month'] == month])
                        m_data.append([str(month), f"${pnl:,.2f}", m_count])
                    
                    axes[1].axis('off')
                    m_table = axes[1].table(cellText=m_data, loc='center', cellLoc='center', colWidths=[0.2, 0.2, 0.2])
                    m_table.auto_set_font_size(False)
                    m_table.set_fontsize(12)
                    m_table.scale(1.2, 1.2)
                    axes[1].set_title("Monthly Performance Summary", pad=20)
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

                # Page 5: Detailed Recent Trades
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.axis('off')
                ax.text(0.5, 0.95, "Recent Trade History (Last 20)", ha='center', fontsize=16, weight='bold')
                
                recent_trades = strat.trade_log[-20:]
                t_data = [["Start Date", "End Date", "Result", "PnL", "Exit Reason"]]
                for t in recent_trades:
                    t_data.append([
                        t["start"].strftime("%Y-%m-%d %H:%M"),
                        t["end"].strftime("%Y-%m-%d %H:%M"),
                        t["result"].upper(),
                        f"${t['amount']:.2f}",
                        t["exit_reason"]
                    ])
                
                table = ax.table(cellText=t_data, loc='center', cellLoc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.0, 1.5)
                
                pdf.savefig(fig)
                plt.close(fig)

        print(f"[Export] PDF report saved: {out_pdf}")

    print("\n[Backtest] Complete!")
