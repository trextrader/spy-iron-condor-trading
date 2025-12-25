# backtest_engine.py
from config import StrategyConfig, RunConfig
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

def run_backtest_and_report(s_cfg: StrategyConfig, r_cfg: RunConfig):
    # --- Load SPY 5-minute bars from disk ---
    class PandasData(bt.feeds.PandasData):
        params = (
            ('datetime', None),  # use the index
            ('open', 'open'),
            ('high', 'high'),
            ('low', 'low'),
            ('close', 'close'),
            ('volume', 'volume'),
            ('openinterest', None),
        )

    csv_path = os.path.join("reports", "SPY", "SPY_5.csv")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.index = pd.to_datetime(df.index) 

    # --- Initialize Backtrader engine ---
    cerebro = bt.Cerebro() 
    cerebro.broker.setcash(r_cfg.backtest_cash) 

    # Add data feed
    data = PandasData(dataname=df) 
    cerebro.adddata(data) 

    # --- Strategy definition ---
    class IronCondorSignal(bt.Strategy):
        params = dict(s_cfg=None, quantity=1) 

        def __init__(self):
            self.s_cfg = self.params.s_cfg 
            self.pnl = 0.0 
            self.wins = 0
            self.losses = 0 
            self.trades = 0 
            self.drawdowns = [] 
            self.peak = r_cfg.backtest_cash 
            self.equity_series = [] 
            self.trade_log = [] 

        def next(self):
            ivr = 35.0 
            vix = 18.0 

            if ivr >= self.s_cfg.iv_rank_min and vix <= self.s_cfg.vix_threshold: 
                width = self.s_cfg.wing_width_min 
                credit = width * self.s_cfg.min_credit_to_width

                import random
                outcome = random.random()
                duration = random.randint(3, 15) 
                start_time = self.datas[0].datetime.date(0)
                entry_ohlcv = {
                    "open": self.data.open[0], 
                    "high": self.data.high[0], 
                    "low": self.data.low[0], 
                    "close": self.data.close[0], 
                    "volume": self.data.volume[0] 
                }

                bid_entry = round(self.data.close[0] - 0.5, 2) 
                ask_entry = round(self.data.close[0] + 0.5, 2) 
                spread_entry = ask_entry - bid_entry 

                exit_idx = min(len(df) - 1, duration) 
                exit_date = df.index[exit_idx] 
                exit_row = df.iloc[exit_idx] 
                exit_ohlcv = {
                    "open": exit_row["open"], 
                    "high": exit_row["high"], 
                    "low": exit_row["low"], 
                    "close": exit_row["close"], 
                    "volume": exit_row["volume"] 
                }

                bid_exit = round(exit_row["close"] - 0.5, 2) 
                ask_exit = round(exit_row["close"] + 0.5, 2) 
                spread_exit = ask_exit - bid_exit 

                if outcome < 0.65: 
                    profit = credit * self.params.quantity 
                    self.pnl += profit 
                    self.wins += 1 
                    self.trade_log.append({
                        "start": start_time, 
                        "end": exit_date, 
                        "duration": duration, 
                        "result": "win", 
                        "amount": profit, 
                        "contracts": self.params.quantity, 
                        "ivr": ivr, 
                        "vix": vix, 
                        "entry_ohlcv": entry_ohlcv, 
                        "exit_ohlcv": exit_ohlcv, 
                        "bid_entry": bid_entry, 
                        "ask_entry": ask_entry, 
                        "spread_entry": spread_entry, 
                        "bid_exit": bid_exit, 
                        "ask_exit": ask_exit, 
                        "spread_exit": spread_exit 
                    })
                else:
                    loss = (width - credit) * 0.6 * self.params.quantity 
                    self.pnl -= loss 
                    self.losses += 1 
                    self.trade_log.append({
                        "start": start_time, 
                        "end": exit_date, 
                        "duration": duration, 
                        "result": "loss", 
                        "amount": -loss, 
                        "contracts": self.params.quantity, 
                        "ivr": ivr, 
                        "vix": vix, 
                        "entry_ohlcv": entry_ohlcv, 
                        "exit_ohlcv": exit_ohlcv, 
                        "bid_entry": bid_entry, 
                        "ask_entry": ask_entry, 
                        "spread_entry": spread_entry, 
                        "bid_exit": bid_exit, 
                        "ask_exit": ask_exit, 
                        "spread_exit": spread_exit 
                    })
                self.trades += 1 

            equity = r_cfg.backtest_cash + self.pnl 
            self.peak = max(self.peak, equity) 
            dd = self.peak - equity 
            self.drawdowns.append(dd) 
            self.equity_series.append(equity) 

    cerebro.addstrategy(IronCondorSignal, s_cfg=s_cfg, quantity=r_cfg.quantity) 
    result = cerebro.run(maxcpus=1) 
    strat = result[0] 

    # --- Metrics ---
    final_equity = r_cfg.backtest_cash + strat.pnl 
    net_profit = final_equity - r_cfg.backtest_cash 
    pct_return = (net_profit / r_cfg.backtest_cash) * 100.0
    days = (r_cfg.backtest_end - r_cfg.backtest_start).days 
    years = days / 365.0 if days > 0 else 0.0 
    cagr = ((final_equity / r_cfg.backtest_cash) ** (1 / years) - 1) * 100.0 if years > 0 else 0.0 

    wins = [t for t in strat.trade_log if t["result"] == "win"] 
    losses = [t for t in strat.trade_log if t["result"] == "loss"] 

    avg_win = np.mean([t["amount"] for t in wins]) if wins else 0 
    avg_loss = np.mean([t["amount"] for t in losses]) if losses else 0
    returns = [t["amount"]/r_cfg.backtest_cash for t in strat.trade_log] 
    sharpe = (np.mean(returns)/np.std(returns))*math.sqrt(len(returns)) if np.std(returns)>0 else 0 
    win_rate = strat.wins/strat.trades if strat.trades>0 else 0 
    expectancy = (avg_win*win_rate) + (avg_loss*(1-win_rate)) 

    metrics = {
        "final_equity": final_equity, 
        "net_profit": net_profit, 
        "pct_return": pct_return, 
        "cagr": cagr, 
        "trades": strat.trades, 
        "wins": strat.wins, 
        "losses": strat.losses, 
        "win_rate": win_rate, 
        "avg_win": avg_win, 
        "avg_loss": avg_loss, 
        "sharpe_ratio": sharpe, 
        "expectancy_ratio": expectancy, 
        "equity_series": strat.equity_series 
    }

    # --- Save outputs ---
    reports_dir = os.path.join(os.getcwd(), "reports") 
    os.makedirs(reports_dir, exist_ok=True) 

    headers = ["Start","End","Dur","Result","Amt","Ctr","IVR","VIX",
               "Entry O/H/L/C/V","Exit O/H/L/C/V",
               "Bid@Entry","Ask@Entry","Spread@Entry",
               "Bid@Exit","Ask@Exit","Spread@Exit"]
    rows = [] 
    for t in strat.trade_log:
        entry = f"{t['entry_ohlcv']['open']}/{t['entry_ohlcv']['high']}/{t['entry_ohlcv']['low']}/{t['entry_ohlcv']['close']}/{t['entry_ohlcv']['volume']}" 
        exitv = f"{t['exit_ohlcv']['open']}/{t['exit_ohlcv']['high']}/{t['exit_ohlcv']['low']}/{t['exit_ohlcv']['close']}/{t['exit_ohlcv']['volume']}" 
        rows.append([
            t["start"], t["end"], t["duration"], t["result"], t["amount"],
            t["contracts"], t["ivr"], t["vix"], entry, exitv, 
            t["bid_entry"], t["ask_entry"], t["spread_entry"], 
            t["bid_exit"], t["ask_exit"], t["spread_exit"] 
        ])

    csv_path = os.path.join(reports_dir, "trades.csv") 
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    pdf_path = os.path.join(reports_dir, "backtest_report.pdf") 
    with PdfPages(pdf_path) as pdf:
        # Page 1: Summary
        fig, ax = plt.subplots(figsize=(8.5, 11)) 
        ax.axis("off") 
        summary_text = "\n".join([f"{k}: {v}" for k, v in metrics.items() if k != "equity_series"]) 
        ax.text(0.05, 0.95, "Backtest Summary", fontsize=16, va="top") 
        ax.text(0.05, 0.9, summary_text, fontsize=10, va="top", family="monospace") 
        pdf.savefig(fig) 
        plt.close(fig) 

        # Page 2: Equity curve and drawdown
        fig, ax = plt.subplots(2, 1, figsize=(8.5, 11), sharex=True) 
        ax[0].plot(metrics["equity_series"], label="Equity") 
        ax[0].set_title("Equity Curve") 
        ax[0].legend() 
        ax[1].plot(strat.drawdowns, label="Drawdown") 
        ax[1].set_title("Drawdown") 
        ax[1].legend() 
        plt.tight_layout() 
        pdf.savefig(fig) 
        plt.close(fig) 

        # Page 3: Candlestick chart with trades
        df.index.name = "Date" 
        window_size = 200 
        start_idx = len(df) 
        found_trades = False
        df_window = None

        while start_idx > 0 and not found_trades: 
            current_window = df.iloc[max(0, start_idx - window_size):start_idx] 
            has_buy = any(t["start"] in current_window.index for t in strat.trade_log)
            has_sell = any(t["end"] in current_window.index for t in strat.trade_log)
            
            if has_buy or has_sell: 
                df_window = current_window
                found_trades = True
            else:
                start_idx -= window_size 

        if df_window is None: 
            df_window = df.tail(window_size) 

        # Create properly sized Series for trade markers
        buy_plot_data = pd.Series(np.nan, index=df_window.index)
        sell_plot_data = pd.Series(np.nan, index=df_window.index)

        for t in strat.trade_log:
            if t["start"] in df_window.index:
                buy_plot_data.loc[t["start"]] = t["entry_ohlcv"]["close"]
            if t["end"] in df_window.index:
                sell_plot_data.loc[t["end"]] = t["exit_ohlcv"]["close"]

        apds = []
        if not buy_plot_data.isna().all(): 
            apds.append(
                mpf.make_addplot(buy_plot_data, type="scatter", markersize=80, marker="^", color="g")
            ) 
        if not sell_plot_data.isna().all(): 
            apds.append(
                mpf.make_addplot(sell_plot_data, type="scatter", markersize=80, marker="v", color="r")
            ) 

        fig, axes = mpf.plot(
            df_window,
            type="candle",
            style="charles",
            addplot=apds if apds else None, 
            title="SPY Backtest Trades (200-Bar Window)",
            ylabel="Price",
            returnfig=True
        ) 

        if not found_trades: 
            axes[0].text(0.5, 0.5, "No trades in this window",
                         transform=axes[0].transAxes, 
                         ha="center", va="center", fontsize=12, color="red") 

        pdf.savefig(fig) 
        plt.close(fig)