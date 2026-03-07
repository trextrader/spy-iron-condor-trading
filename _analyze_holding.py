import pandas as pd, numpy as np

df = pd.read_csv('reports/run18_Backtest/trades.csv')
opens  = df[df['action']=='OPEN'].reset_index(drop=True)
closes = df[df['action'].isna()].reset_index(drop=True)
merged = opens[['trade_id','strategy_class','template_id','entry_credit','max_loss']].copy()
merged['pnl']    = pd.to_numeric(closes['pnl'].values, errors='coerce')
merged['reason'] = closes['reason'].values
merged['win']    = merged['pnl'] > 0

order = ['custom_multi_leg','bear_put_spread','bull_call_spread','iron_condor']

for strat in order:
    g = merged[merged['strategy_class']==strat].copy()
    wins   = g[g['win']]
    losses = g[~g['win']]
    n  = len(g)
    nw = len(wins)
    nl = len(losses)

    # Find lowest threshold above which ALL trades are wins
    sweet_thresh = None
    sweet_count  = 0
    sweet_min    = None
    if nw > 0:
        for thresh in np.arange(0, g['pnl'].max() + 100, 100):
            above = g[g['pnl'] >= thresh]
            if len(above) > 0 and above['win'].all():
                sweet_thresh = thresh
                sweet_count  = len(above)
                sweet_min    = above['pnl'].min()
                break

    tmpl = g['template_id'].mode()[0]
    print(f"Strategy : {strat}  (template={tmpl})")
    print(f"  Trades : {n}  |  Wins={nw} ({nw/n*100:.0f}%)  Losses={nl} ({nl/n*100:.0f}%)")
    print(f"  RANGE  :  Low=\${g['pnl'].min():>+9,.0f}   High=\${g['pnl'].max():>+9,.0f}")
    if nw > 0:
        print(f"  Wins   : \${wins['pnl'].min():>+9,.0f}  to  \${wins['pnl'].max():>+9,.0f}")
        print(f"  Losses : \${losses['pnl'].min():>+9,.0f}  to  \${losses['pnl'].max():>+9,.0f}")
        if sweet_thresh is not None:
            print(f"  SWEET SPOT: >= \${sweet_thresh:,.0f} -> {sweet_count} entries, ALL wins, min payout=\${sweet_min:,.0f}")
    else:
        print(f"  No winning trades")

    reasons = dict(g['reason'].value_counts())
    print(f"  Exits  : {reasons}")

    # PnL bucket breakdown
    pmin = g['pnl'].min()
    pmax = g['pnl'].max()
    step = max(500, round((pmax - pmin) / 8 / 500) * 500)
    bins = np.arange(pmin - step, pmax + step * 2, step)
    g['bucket'] = pd.cut(g['pnl'], bins=bins)
    print(f"  Bucket breakdown (step=${step:,.0f}):")
    for bkt, bg in g.groupby('bucket', observed=True):
        w = bg['win'].sum()
        bar = '#' * len(bg)
        mark = '  ALL WIN' if w == len(bg) and len(bg) > 0 else ('  ALL LOSS' if w == 0 else f'  {w}/{len(bg)}')
        print(f"    [{str(bkt):>30}]  n={len(bg):>3}  wins={w}  {bar}{mark}")
    print()
