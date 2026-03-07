"""
Find the per-strategy profit target that maximizes total net PnL.
Optimization: find threshold T where sum(pnl for all pnl >= T) is highest.
"""
import pandas as pd, numpy as np

df = pd.read_csv('reports/run18_Backtest/trades.csv')
opens  = df[df['action']=='OPEN'].reset_index(drop=True)
closes = df[df['action'].isna()].reset_index(drop=True)
merged = opens[['trade_id','strategy_class','template_id','entry_credit','max_loss']].copy()
merged['pnl']    = pd.to_numeric(closes['pnl'].values, errors='coerce')
merged['reason'] = closes['reason'].values
merged['win']    = merged['pnl'] > 0

order = ['custom_multi_leg','bear_put_spread','bull_call_spread','iron_condor']

OPTIMAL = {}

for strat in order:
    g = merged[merged['strategy_class']==strat].copy()
    g = g[g['pnl'].notna()].sort_values('pnl', ascending=False)
    pnl_vals = g['pnl'].values

    # Scan every unique pnl value as a candidate threshold
    best_total = -np.inf
    best_thresh = 0
    best_n = 0
    best_min = 0

    candidates = sorted(set(pnl_vals[pnl_vals > 0])) if any(pnl_vals > 0) else []
    # Also test $0 (all positive pnl trades)
    candidates = [0] + candidates

    rows = []
    for T in candidates:
        above = g[g['pnl'] >= T]
        total = above['pnl'].sum()
        rows.append((T, len(above), total, above['pnl'].min() if len(above) > 0 else 0))
        if total > best_total:
            best_total = total
            best_thresh = T
            best_n      = len(above)
            best_min    = above['pnl'].min() if len(above) > 0 else 0

    OPTIMAL[strat] = {
        'threshold': best_thresh,
        'n_trades':  best_n,
        'total_pnl': best_total,
        'min_pnl':   best_min,
    }

    print(f"=== {strat} ({len(g)} trades) ===")
    print(f"  {'Threshold':>12}  {'N trades':>9}  {'Total PnL':>12}  {'Min PnL':>10}  note")
    print(f"  {'-'*65}")
    for T, n, total, mn in sorted(rows, key=lambda x: -x[2])[:12]:
        mark = ' <-- OPTIMAL' if T == best_thresh else ''
        print(f"  ${T:>11,.0f}  {n:>9}  ${total:>11,.0f}  ${mn:>9,.0f}{mark}")
    print(f"  BEST: target >= ${best_thresh:,.0f} -> {best_n} trades, total=${best_total:,.0f}, min=${best_min:,.0f}")
    print()

print("=" * 60)
print("PER-STRATEGY PROFIT TARGET MAP (for backtest config)")
print("=" * 60)
for strat, opt in OPTIMAL.items():
    print(f"  '{strat}': {opt['threshold']:.0f},  # {opt['n_trades']} trades -> ${opt['total_pnl']:,.0f} total")
