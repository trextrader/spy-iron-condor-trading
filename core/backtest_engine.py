# core/backtest_engine.py - Iron Condor with MTF + Professional Reporting
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import StrategyConfig, RunConfig
import backtrader as bt
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
from tabulate import tabulate
import mplfinance as mpf
import os
import csv
from matplotlib.backends.backend_pdf import PdfPages
from data_factory.sync_engine import MTFSyncEngine
from strategies.options_strategy import (
    OptionQuote, build_condor, calc_condor_credit, 
    PositionState, regime_wing_width
)

def run_backtest_headless(s_cfg: StrategyConfig, r_cfg: RunConfig, preloaded_df=None, preloaded_options=None, preloaded_sync=None, verbose=True):
    """
    Run backtest without generating reports, returning raw strategy object
    Used for optimization loops.
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
        csv_path = os.path.join("reports", s_cfg.underlying, f"{s_cfg.underlying}_1.csv")
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
    
    # === Load Synthetic Options Data ===
    if preloaded_options is not None:
        options_by_date = preloaded_options
    else:
        options_path = os.path.join("data", "synthetic_options", f"{s_cfg.underlying.lower()}_options_marks.csv")
        if not os.path.exists(options_path):
            print(f"[ERROR] Synthetic options data not found: {options_path}")
            return None
        
        # Load and optimize for fast lookup
        if verbose: print(f"[Data] Loading synthetic options: {options_path}...")
        options_df = pd.read_csv(options_path, parse_dates=["date", "expiration"])
        options_by_date = {}
        for date, group in options_df.groupby('date'):
            options_by_date[date.date()] = group.to_dict('records')
    
    # === Initialize MTF Sync Engine ===
    if preloaded_sync is not None:
        sync_engine = preloaded_sync
    else:
        sync_engine = MTFSyncEngine(s_cfg.underlying, r_cfg.mtf_timeframes) if r_cfg.use_mtf else None

    # === Strategy Definition ===
    class IronCondorStrategy(bt.Strategy):
        params = dict(s_cfg=None, r_cfg=None, sync_engine=None, options_data=None, verbose=True)

        def __init__(self):
            self.s_cfg = self.params.s_cfg
            self.r_cfg = self.params.r_cfg
            self.sync = self.params.sync_engine
            self.options_data = self.params.options_data
            self.verbose = self.params.verbose
            
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
            self.bars_since_trade = 100 
            self.price_cache = {} # Persistent cache for leg prices: {symbol: last_price}
            
            # Progress tracking
            self.bar_count = 0
            self.total_bars = len(self.datas[0])
            self.last_progress_pct = -1
            
            # Initialize Mamba Neural Engine
            if getattr(self.s_cfg, 'use_mamba_model', False):
                try:
                    from intelligence.mamba_engine import MambaForecastEngine
                    self.mamba_engine = MambaForecastEngine(d_model=getattr(self.s_cfg, 'mamba_d_model', 64))
                    if self.verbose:
                        print(f"[{self.__class__.__name__}] Mamba Neural Engine Initialized")
                except Exception as e:
                    print(f"[Warning] Failed to init Mamba Engine: {e}")
                    self.mamba_engine = None

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
            
            # 1. Update Equity tracking for every bar
            current_equity = self.r_cfg.backtest_cash + self.pnl
            
            # If we provide an option chain for this date, calculate floating P&L
            unrealized_pnl = 0.0
            exit_triggered = False
            exit_reason = ""
            
            if self.active_position:
                chain_records = self.options_data.get(date_now, [])
                if chain_records:
                    leg_symbols = [
                        self.active_position.legs.short_call.symbol,
                        self.active_position.legs.long_call.symbol,
                        self.active_position.legs.short_put.symbol,
                        self.active_position.legs.long_put.symbol
                    ]
                    
                    # Update cache with current bar's prices
                    for r in chain_records:
                        if r['option_symbol'] in leg_symbols:
                            self.price_cache[r['option_symbol']] = r['last_price']
                    
                    # Try to get all prices (current or cached)
                    current_prices = {s: self.price_cache.get(s) for s in leg_symbols if s in self.price_cache}
                    
                    if len(current_prices) == 4:
                        # Current replacement cost (to close)
                        current_cost = (
                            current_prices[self.active_position.legs.short_call.symbol] - 
                            current_prices[self.active_position.legs.long_call.symbol] +
                            current_prices[self.active_position.legs.short_put.symbol] -
                            current_prices[self.active_position.legs.long_put.symbol]
                        )
                        
                        unrealized_pnl = (self.active_position.credit_received - current_cost) * self.active_position.quantity * 100
                        
                        # Exit Rule: Profit Take
                        target_profit = self.active_position.credit_received * self.s_cfg.profit_take_pct
                        
                        # Exit Rule: Dynamic Stop Loss (ATR-based)
                        base_multiplier = self.s_cfg.loss_close_multiple
                        use_atr_stops = getattr(self.s_cfg, 'use_atr_stops', False)
                        if use_atr_stops and self.sync:
                            # Get current ATR from primary timeframe
                            primary_tf = self.r_cfg.mtf_timeframes[0] if self.r_cfg.mtf_timeframes else '1'
                            tf_snapshot = self.sync.get_snapshot(dt_now)
                            atr_pct = tf_snapshot.get(primary_tf, {}).get('atr_pct', 0.01) if tf_snapshot and tf_snapshot.get(primary_tf) else 0.01
                            
                            # Import and calculate dynamic multiplier
                            from intelligence.fuzzy_engine import calculate_atr_stop_multiplier
                            atr_base = getattr(self.s_cfg, 'atr_stop_base_multiplier', 1.5)
                            dynamic_multiplier = calculate_atr_stop_multiplier(atr_pct, atr_base)
                            stop_loss = self.active_position.credit_received * (1 + dynamic_multiplier)
                        else:
                            stop_loss = self.active_position.credit_received * (1 + base_multiplier)
                        
                        # Exit Rule: Trailing Stop
                        use_trailing = getattr(self.s_cfg, 'use_trailing_stop', False)
                        if use_trailing and unrealized_pnl > 0:
                            # Check if trailing stop is activated
                            profit_ratio = unrealized_pnl / (self.active_position.credit_received * self.active_position.quantity * 100)
                            activation_pct = getattr(self.s_cfg, 'trailing_stop_activation_pct', 0.50)
                            distance_pct = getattr(self.s_cfg, 'trailing_stop_distance_pct', 0.25)
                            if profit_ratio >= activation_pct:
                                # Set trailing stop distance
                                trailing_floor = self.active_position.credit_received * (1 - activation_pct + distance_pct)
                                if current_cost <= trailing_floor:
                                    exit_triggered = True
                                    exit_reason = "Trailing Stop"
                        
                        if current_cost <= target_profit:
                            exit_triggered = True
                            exit_reason = "Profit Take"
                        elif current_cost >= stop_loss:
                            exit_triggered = True
                            exit_reason = "Stop Loss"
                    
                    # Exit Rule: Expiration (Date based, always checked)
                    dte = (self.active_position.legs.short_call.expiration - date_now).days
                    if dte <= 0:
                        exit_triggered = True
                        exit_reason = "Expiration"
                    
                    # Periodic Status
                    if self.verbose and len(self.equity_series) % 100 == 0:
                        cost_str = f"${current_cost:.2f}" if 'current_cost' in locals() else "N/A"
                        print(f"  [Position] PnL: ${unrealized_pnl:.2f} | Cost: {cost_str} | DTE: {dte}")
                            
                        if exit_triggered:
                            realized_pnl = unrealized_pnl
                            self.pnl += realized_pnl
                            self.trades += 1
                            if realized_pnl > 0: self.wins += 1
                            else: self.losses += 1
                            
                            print(f"[Trade] Closed: {exit_reason} | PnL: ${realized_pnl:.2f}")

                            # Market context at exit
                            exit_ohlcv = {
                                "open": self.data.open[0],
                                "high": self.data.high[0],
                                "low": self.data.low[0],
                                "close": self.data.close[0],
                                "volume": self.data.volume[0]
                            }
                            
                            self.trade_log.append({
                                "start": self.active_position.open_time,
                                "end": dt_now,
                                "result": "win" if realized_pnl > 0 else "loss",
                                "amount": realized_pnl,
                                "contracts": self.active_position.quantity,
                                "exit_reason": exit_reason,
                                "credit": self.active_position.credit_received,
                                "exit_cost": current_cost if 'current_cost' in locals() else 0.0,
                                "symbols": leg_symbols,
                                "ivr": self.active_position.metadata.get("ivr", 0),
                                "vix": self.active_position.metadata.get("vix", 0),
                                "mtf_consensus": self.active_position.mtf_consensus,
                                "wing_width": self.active_position.metadata.get("width", 0),
                                "strikes": self.active_position.metadata.get("strikes", {}),
                                "entry_ohlcv": self.active_position.metadata.get("entry_ohlcv", {}),
                                "exit_ohlcv": exit_ohlcv
                            })
                            self.active_position = None
                            unrealized_pnl = 0.0
                            self.bars_since_trade = 0 # Cooling Period

            # Track total equity including floating gain/loss
            total_equity = current_equity + unrealized_pnl
            self.peak = max(self.peak, total_equity)
            self.drawdowns.append(self.peak - total_equity)
            self.equity_series.append(total_equity)

            # 2. Skip if already in position or cooling down
            if self.active_position:
                return
                
            self.bars_since_trade += 1
            if self.bars_since_trade < 100:
                return

            # 3. Entry Logic
            chain_records = self.options_data.get(date_now, [])
            
            # Debug: Print once on first attempt after cooldown
            if self.bar_count == 101 and self.verbose:
                opt_keys = list(self.options_data.keys())[:5]
                print(f"[DEBUG] Looking for date: {date_now} (type: {type(date_now)})")
                print(f"[DEBUG] Options keys sample: {opt_keys} (type: {type(opt_keys[0]) if opt_keys else 'N/A'})")
                print(f"[DEBUG] Chain records found: {len(chain_records)}")
            
            if not chain_records:
                return
                
            quote_chain = [OptionQuote(
                symbol=r['option_symbol'],
                expiration=r['expiration'].date(),
                strike=r['strike'],
                is_call=(r['contract_type'] == 'call'),
                bid=r['bid'],
                ask=r['ask'],
                mid=r['last_price'],
                delta=r['delta'],
                iv=r['implied_volatility']
            ) for r in chain_records]

            # Market context (Simulated until we have real indices)
            ivr = 35.0 + np.random.uniform(-5, 5)
            vix = 18.0 + np.random.uniform(-3, 3) 
            
            if ivr < self.s_cfg.iv_rank_min or vix > self.s_cfg.vix_threshold:
                if self.bar_count == 101 and self.verbose:
                    print(f"[DEBUG] Entry blocked by IVR/VIX: ivr={ivr:.1f} (min={self.s_cfg.iv_rank_min}), vix={vix:.1f} (max={self.s_cfg.vix_threshold})")
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

            width = regime_wing_width(self.s_cfg, ivr, vix)
            condor = build_condor(quote_chain, self.s_cfg, date_now, self.data.close[0], width)
            
            if not condor:
                if self.bar_count == 101 and self.verbose:
                    print(f"[DEBUG] Entry blocked: condor build failed (width={width}, close={self.data.close[0]:.2f})")
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
                if self.bar_count == 101 and self.verbose:
                    print(f"[DEBUG] Entry blocked by credit. Needed {min_credit:.2f} (ratio {required_ratio:.2f}), Got {credit:.2f}")
                    # Diagnostic: Print Legs
                    if len(condor) == 4:
                        print(f"    Legs: ShortC={condor[0]['strike']}@{condor[0]['mid']:.2f} | LongC={condor[1]['strike']}@{condor[1]['mid']:.2f}")
                        print(f"          ShortP={condor[2]['strike']}@{condor[2]['mid']:.2f} | LongP={condor[3]['strike']}@{condor[3]['mid']:.2f}")
                        call_cr = condor[0]['mid'] - condor[1]['mid']
                        put_cr = condor[2]['mid'] - condor[3]['mid']
                        print(f"    Calc: CallCr={call_cr:.2f} + PutCr={put_cr:.2f} = {call_cr + put_cr:.2f}")
                return

            # === Neural Forecasting (Mamba 2) ===
            neural_forecast_data = None
            if getattr(self.s_cfg, 'use_mamba_model', False) and hasattr(self, 'mamba_engine'):
                # Pass recent data context to Mamba
                # Creating a small DF context for the mock/inference
                ctx_df = pd.DataFrame([{
                    'close': self.data.close[0],
                    'rsi_14': rsi_current,
                    'atr_pct': 0.01, # Simplified for mock
                    'volume_ratio': volume_ratio
                }])
                
                # Predict Market State
                forecast_state = self.mamba_engine.predict_state(ctx_df)
                neural_forecast_data = forecast_state.to_dict()
                
                if self.verbose and self.bar_count % 100 == 0:
                    print(f"[Neural:{forecast_state.model_backend}] Conf={forecast_state.confidence:.2f} | P(Bull)={forecast_state.prob_bull:.2f} | P(Bear)={forecast_state.prob_bear:.2f}")

            # === Sizing via QTMF Facade (Neuro-Fuzzy) ===
            from qtmf.models import TradeIntent
            from qtmf.facade import benchmark_and_size
            
            # Construct Trade Intent with all 9 indicators + Neural Signal
            intent = TradeIntent(
                symbol=self.s_cfg.underlying,
                action="SELL_CONDOR",
                gaussian_confidence=0.5, # Gaussian component placeholder for hybrid weighting
                current_price=self.data.close[0],
                vix=vix,
                ivr=ivr,
                realized_vol=0.0,
                mtf_snapshot=mtf_snapshot,
                # New Advanced Indicators
                rsi=rsi_current,
                adx=adx_current,
                bb_position=bb_position,
                bb_width=bb_width,
                stoch_k=stoch_k,
                volume_ratio=volume_ratio,
                sma_distance=sma_distance,
                # Neural Forecast
                neural_forecast=neural_forecast_data,
                extras={
                    'equity': self.broker.get_cash(),
                    'max_loss_per_contract': width * 100.0, # Approximate max loss as spread width
                    'risk_fraction': self.s_cfg.max_account_risk_per_trade,
                    # Pass config weights to facade
                    'w_mtf': getattr(self.s_cfg, 'fuzzy_weight_mtf', 0.25),
                    'w_iv': getattr(self.s_cfg, 'fuzzy_weight_iv', 0.20),
                    'w_regime': getattr(self.s_cfg, 'fuzzy_weight_regime', 0.15),
                    'w_rsi': getattr(self.s_cfg, 'fuzzy_weight_rsi', 0.10),
                    'w_adx': getattr(self.s_cfg, 'fuzzy_weight_adx', 0.10),
                    'w_bbands': getattr(self.s_cfg, 'fuzzy_weight_bbands', 0.10),
                    'w_volume': getattr(self.s_cfg, 'fuzzy_weight_volume', 0.05),
                    'w_sma': getattr(self.s_cfg, 'fuzzy_weight_sma', 0.05),
                    'w_stoch': getattr(self.s_cfg, 'fuzzy_weight_stoch', 0.07),
                    'fuzzy_weight_neural': getattr(self.s_cfg, 'fuzzy_weight_neural', 0.20)
                }
            )
            
            # Get Sizing Plan from Facade
            plan = benchmark_and_size(intent)
            
            if not plan.approved:
                # Silent return unless verbose debug needed
                # if self.verbose: print(f"[DEBUG] Entry blocked by QTMF: {plan.reason}")
                return
                
            quantity = plan.total_qty

            
            if quantity > 0:
                print(f"[Trade] Opened | Qty: {quantity} | Credit: ${credit:.2f} | Exp: {condor.short_call.expiration}")
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
                    credit_received=credit,
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
                    }
                }

    # === Run Backtest ===
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(r_cfg.backtest_cash)
    cerebro.adddata(PandasData(dataname=df))
    cerebro.addstrategy(IronCondorStrategy, s_cfg=s_cfg, r_cfg=r_cfg, sync_engine=sync_engine, options_data=options_by_date)
    
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
    final_equity = r_cfg.backtest_cash + strat.pnl
    net_profit = final_equity - r_cfg.backtest_cash
    pct_return = (net_profit / r_cfg.backtest_cash) * 100.0
    
    # Date calculations
    if strat.trade_log:
        start_date = strat.trade_log[0]["start"].date()
        end_date = strat.trade_log[-1]["end"].date()
    else:
        # Fallback to feed dates if no trades closed
        start_date = strat.data.datetime.date(-len(strat.data)+1)
        end_date = strat.data.datetime.date(0)
    
    days = (end_date - start_date).days
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
    profit_factor = gross_win / gross_loss if gross_loss > 0 else 0.0
    
    largest_win = max([t["amount"] for t in wins]) if wins else 0.0
    largest_loss = min([t["amount"] for t in losses]) if losses else 0.0
    
    net_pnl_dd_ratio = net_profit / max_dd_money if max_dd_money > 0 else 0.0

    # 3. Total Time in Trade
    total_seconds = sum((t["end"] - t["start"]).total_seconds() for t in strat.trade_log)
    m, s = divmod(total_seconds, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    time_in_trade_str = f"{int(d)}d {int(h):02d}:{int(m):02d}:{int(s):02d}"

    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    print("\n" + "="*60)
    print(f"BACKTEST RESULTS: {s_cfg.underlying}")
    print("="*60)
    print(f"Period:           {start_date_str} to {end_date_str} ({days} days)")
    print(f"Final Equity:     ${final_equity:,.2f}")
    print(f"Net Profit:       ${net_profit:,.2f} ({pct_return:.2f}%)")
    print(f"CAGR:             {cagr:.2f}%")
    print(f"Max Drawdown:     ${max_dd_money:,.2f} ({max_dd_pct:.2f}%)")
    print(f"Sharpe Ratio:     {sharpe:.2f}")
    print(f"Profit Factor:    {profit_factor:.2f}")
    print(f"Net PnL / DD:     {net_pnl_dd_ratio:.2f}")
    print(f"Time in Trade:    {time_in_trade_str}")
    print(f"Total Trades:     {len(strat.trade_log)}")
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
    
    # Save PDF Report
    if r_cfg.backtest_plot:
        # Re-load DF just for plotting (inefficient but cleaner for now)
        csv_path = os.path.join("reports", s_cfg.underlying, f"{s_cfg.underlying}_5.csv")
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
                ["Period", f"{start_date_str} to {end_date_str}"],
                ["Initial Capital", f"${r_cfg.backtest_cash:,.2f}"],
                ["Final Equity", f"${final_equity:,.2f}"],
                ["Net Profit", f"${net_profit:,.2f} ({pct_return:.2f}%)"],
                ["Max Drawdown", f"${max_dd_money:,.2f} ({max_dd_pct:.2f}%)"],
                ["Sharpe Ratio", f"{sharpe:.2f}"],
                ["Profit Factor", f"{profit_factor:.2f}"],
                ["Net PnL / DD", f"{net_pnl_dd_ratio:.2f}"],
                ["Time in Trade", f"{time_in_trade_str}"],
                ["Total Trades", f"{len(strat.trade_log)}"],
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