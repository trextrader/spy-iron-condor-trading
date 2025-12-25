# main.py
import argparse
import datetime as dt
from config import StrategyConfig, RunConfig
from polygon_client import PolygonClient
from broker import PaperBroker
from options_strategy import run_strategy, select_and_enter_condor, manage_positions
from backtest_engine import run_backtest_and_report
from metrics import plot_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="SPY Iron Condor with Polygon data and Backtrader backtesting")
    # Mode
    parser.add_argument("--mode", choices=["live", "backtest"], default="backtest",
                        help="Run live (paper broker) or backtest mode.")
    # Polygon
    parser.add_argument("--polygon-key", type=str, default=None, help="Polygon.io API key.")
    # Strategy toggles
    parser.add_argument("--quantity", type=int, default=1, help="Contracts per condor (fixed sizing).")
    parser.add_argument("--wing-min", type=float, default=5.0, help="Min wing width.")
    parser.add_argument("--wing-max", type=float, default=10.0, help="Max wing width.")
    parser.add_argument("--min-credit-ratio", type=float, default=0.25, help="Min credit/width ratio.")
    parser.add_argument("--ivr-min", type=float, default=30.0, help="Minimum IV Rank to trade.")
    parser.add_argument("--vix-max", type=float, default=25.0, help="Maximum VIX to open new trades.")
    parser.add_argument("--profit-pct", type=float, default=0.50, help="Profit take percentage.")
    parser.add_argument("--loss-multiple", type=float, default=1.5, help="Loss close multiple of credit.")
    parser.add_argument("--max-hold-days", type=int, default=14, help="Days to expiration threshold for exit.")
    parser.add_argument("--max-positions", type=int, default=3, help="Max concurrent condors.")
    parser.add_argument("--max-alloc", type=float, default=0.15, help="Max portfolio allocation.")
    parser.add_argument("--dte-min", type=int, default=30, help="Min DTE.")
    parser.add_argument("--dte-max", type=int, default=45, help="Max DTE.")
    parser.add_argument("--delta-low", type=float, default=0.15, help="Short leg target delta (abs) low.")
    parser.add_argument("--delta-high", type=float, default=0.20, help="Short leg target delta (abs) high.")
    # Regime/delta hedge
    parser.add_argument("--enable-hedge", action="store_true", help="Enable dynamic delta hedging.")
    parser.add_argument("--hedge-threshold", type=float, default=0.10, help="Net position delta threshold.")
    parser.add_argument("--hedge-unit", type=int, default=10, help="Shares per 0.01 delta.")
    parser.add_argument("--regime-ivr-widen", type=float, default=40.0, help="IV Rank threshold to widen wings.")
    parser.add_argument("--regime-vix-widen", type=float, default=22.0, help="VIX threshold to widen wings.")
    parser.add_argument("--width-increment", type=float, default=5.0, help="Wing width increment in high vol regime.")
    # Backtest flags
    parser.add_argument("--bt-cash", type=float, default=25000.0,
                        help="Initial cash for backtest (default $25,000.00)")
    parser.add_argument("--bt-start", type=str, default="2024-01-01", help="Backtest start date YYYY-MM-DD.")
    parser.add_argument("--bt-end", type=str, default="2024-12-31", help="Backtest end date YYYY-MM-DD.")
    parser.add_argument("--bt-plot", action="store_true", help="Plot backtest metrics.")
    parser.add_argument("--bt-samples", type=int, default=500, help="Limit samples for speed.")
    # Optimizer (optional feed)
    parser.add_argument("--use-optimizer", action="store_true", help="Use width optimizer from historical curve.")
    # Dynamic sizing
    parser.add_argument("--dynamic-sizing", action="store_true",
                        help="Enable dynamic position sizing as % of equity.")
    parser.add_argument("--position-size-pct", type=float, default=0.02,
                        help="Fraction of equity risked per trade if dynamic sizing enabled (default 2%).")
    return parser.parse_args()

def build_configs(args):
    s_cfg = StrategyConfig(
        underlying="SPY",
        dte_min=args.dte_min,
        dte_max=args.dte_max,
        target_short_delta_low=args.delta_low,
        target_short_delta_high=args.delta_high,
        wing_width_min=args.wing_min,
        wing_width_max=args.wing_max,
        min_credit_to_width=args.min_credit_ratio,
        max_account_risk_per_trade=0.02,
        max_positions=args.max_positions,
        max_portfolio_alloc=args.max_alloc,
        iv_rank_min=args.ivr_min,
        vix_threshold=args.vix_max,
        profit_take_pct=args.profit_pct,
        loss_close_multiple=args.loss_multiple,
        max_hold_days=args.max_hold_days,
        delta_roll_threshold=0.30,
        allow_one_adjustment_per_side=True,
        dynamic_delta_hedge_threshold=args.hedge_threshold if args.enable_hedge else 999.0,
        dynamic_delta_hedge_unit=args.hedge_unit,
        regime_iv_rank_widen=args.regime_ivr_widen,
        regime_vix_widen=args.regime_vix_widen,
        width_widen_increment=args.width_increment
    )

    r_cfg = RunConfig(
        polygon_key=args.polygon_key,
        quantity=args.quantity,
        use_optimizer=args.use_optimizer,
        backtest_start=dt.datetime.strptime(args.bt_start, "%Y-%m-%d").date(),
        backtest_end=dt.datetime.strptime(args.bt_end, "%Y-%m-%d").date(),
        backtest_cash=args.bt_cash,
        backtest_plot=args.bt_plot,
        backtest_samples=args.bt_samples,
        dynamic_sizing=args.dynamic_sizing,
        position_size_pct=args.position_size_pct
    )
    return s_cfg, r_cfg

def main():
    args = parse_args()
    s_cfg, r_cfg = build_configs(args)

    print(f"Starting backtest with initial balance: ${r_cfg.backtest_cash:,.2f}")
    print(f"Sizing mode: {'dynamic %' if r_cfg.dynamic_sizing else 'fixed'}, "
          f"position_size_pct={r_cfg.position_size_pct}")

    if args.mode == "backtest":
        run_backtest_and_report(s_cfg, r_cfg)
        return

    # Live (paper) mode
    if not r_cfg.polygon_key:
        raise ValueError("Polygon API key required for live mode. Pass --polygon-key YOUR_KEY")

    poly = PolygonClient(r_cfg.polygon_key)
    broker = PaperBroker(poly)

    historical_credit_curve = {5.0: 0.28, 7.5: 0.30, 10.0: 0.27} if r_cfg.use_optimizer else None

    today = dt.date.today()
    pos = select_and_enter_condor(broker, s_cfg, today, historical_credit_curve, r_cfg.quantity)
    if pos:
        print(f"Entered position {pos.id} with credit {pos.credit_received:.2f}")
    manage_positions(broker, s_cfg, today)

    plot_metrics(broker.collect_trade_log(), title="Live Paper Metrics")

if __name__ == "__main__":
    main()
