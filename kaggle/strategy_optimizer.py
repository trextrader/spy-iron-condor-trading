"""
Strategy Optimizer — Grid search with interactive parameter prompts.
Runs the v45 backtester in a loop across parameter combinations,
ranks by NP/DD ratio, and lets user apply the best config.
"""
import itertools
import copy
import time
import os
import re
import sys
import io
import contextlib
import numpy as np
import pandas as pd


def run_optimizer(run_backtest_fn, args, df, rule_signals, model, feature_cols, device,
                  ruleset, model_path, data_path, norm_stats,
                  v43_outputs, bundle, dte_by_dow,
                  allowed_strategy_idxs, allowed_template_ids,
                  strategy_configs):
    """Interactive grid-search optimizer.

    Prompts for param ranges, runs backtest for each combination,
    ranks by NP/DD, lets user apply best params to strategy .py file.
    """

    # Resolve which template we are optimizing
    if allowed_template_ids and len(allowed_template_ids) == 1:
        template_id = list(allowed_template_ids)[0]
    elif allowed_template_ids:
        print("[optimizer] Multiple templates selected. Pick one for optimization.")
        for i, tid in enumerate(sorted(allowed_template_ids)):
            print(f"  {i}: {tid}")
        sel = input("Select index: ").strip()
        template_id = sorted(allowed_template_ids)[int(sel)]
    else:
        print("[optimizer] ERROR: --strategies must specify a template (e.g. --strategies short_call)")
        return

    base_cfg = strategy_configs.get(template_id)
    if not base_cfg:
        print(f"[optimizer] ERROR: No config found for template '{template_id}'")
        return

    print()
    print("=" * 70)
    print(f"  STRATEGY OPTIMIZER -- {template_id}")
    print("=" * 70)
    print("  Define parameter ranges:  current_value -> min max step")
    print("  Press Enter to skip (keeps current value)")
    print()

    # Optimizable params: (config_key, display_name, is_cli_param, type)
    PARAM_DEFS = [
        ("stop_loss_dollar",  "stop_loss_dollar",  False, int),
        ("profit_target",     "profit_target",     False, int),
        ("max_positions",     "max_positions",     True,  int),
        ("cooldown_bars",     "cooldown_bars",     False, int),
        ("stop_loss_mult",    "stop_loss_mult",    False, float),
        ("max_contracts",     "max_contracts",     False, int),
    ]

    param_grids = {}      # config key -> [values]
    cli_param_grids = {}  # CLI key -> [values]

    for key, display, is_cli, ptype in PARAM_DEFS:
        if is_cli:
            current = getattr(args, key, None) or base_cfg.get(key, "?")
        else:
            current = base_cfg.get(key, "?")

        prompt = f"  {display} ({current}): "
        raw = input(prompt).strip()

        if not raw:
            # Skip: use current value only
            if is_cli:
                cli_param_grids[key] = [current]
            else:
                param_grids[key] = [current]
        else:
            parts = raw.split()
            if len(parts) == 3:
                lo, hi, step = ptype(parts[0]), ptype(parts[1]), ptype(parts[2])
                if ptype == int:
                    vals = list(range(lo, hi + 1, step))
                else:
                    vals = []
                    v = lo
                    while v <= hi + 1e-9:
                        vals.append(round(v, 4))
                        v += step
            elif len(parts) == 1:
                vals = [ptype(parts[0])]
            else:
                vals = [ptype(p) for p in parts]

            if is_cli:
                cli_param_grids[key] = vals
            else:
                param_grids[key] = vals
            print(f"    -> {len(vals)} values: {vals[:10]}{'...' if len(vals) > 10 else ''}")

    # Build grid
    all_keys = list(param_grids.keys()) + list(cli_param_grids.keys())
    all_vals = list(param_grids.values()) + list(cli_param_grids.values())

    combos = list(itertools.product(*all_vals))
    n_combos = len(combos)

    print(f"\n  Grid: {' x '.join(str(len(v)) for v in all_vals)} = {n_combos} combinations")

    # Estimate time with a quick benchmark + cache bar data
    print("  Running benchmark (1 backtest + caching bar data)...")
    t0 = time.time()
    _bm_result = run_backtest_fn(
        df, rule_signals, model, feature_cols, device, ruleset,
        model_path=model_path, data_path=data_path, norm_stats=norm_stats,
        use_fuzzy_sizing=args.use_fuzzy_sizing, use_trade_rules=args.use_trade_rules,
        use_diffusion=args.use_diffusion, limit=args.limit,
        v43_outputs=v43_outputs, bundle=bundle,
        verbose_sim=False, max_positions=args.max_positions,
        allowed_strategy_idxs=allowed_strategy_idxs,
        allowed_template_ids=allowed_template_ids,
        use_profit_targets=getattr(args, 'profittargets', False),
        dte_by_dow=dte_by_dow,
        return_bar_cache=True,
    )
    _bm_eq, _bm_tr, _bar_cache = _bm_result
    print(f"  Bar cache: {len(_bar_cache)} bars cached")
    bench_sec = time.time() - t0
    est_min = (bench_sec * n_combos) / 60
    print(f"  Benchmark: {bench_sec:.1f}s per run")
    print(f"  Est. time: {est_min:.1f} minutes ({n_combos} x {bench_sec:.1f}s)")

    confirm = input(f"  Proceed with {n_combos} combinations? [y/N]: ").strip().lower()
    if confirm != 'y':
        print("  Aborted.")
        return

    # Execute grid
    results = []
    start_time = time.time()

    # Incremental CSV output (survives crashes / early stops)
    import datetime as _dt
    os.makedirs("reports", exist_ok=True)
    csv_path = os.path.join("reports",
                            f"optimizer_{template_id}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    def _save_csv():
        """Write top 100 results (sorted by NP/DD) to CSV."""
        sorted_res = sorted(results, key=lambda x: x['np_dd'], reverse=True)[:100]
        rows = []
        for ri, r in enumerate(sorted_res):
            row_data = {"rank": ri + 1}
            row_data.update(r['combo'])
            row_data.update({k: v for k, v in r.items() if k != 'combo'})
            rows.append(row_data)
        pd.DataFrame(rows).to_csv(csv_path, index=False)


    for ci, combo in enumerate(combos):
        combo_dict = dict(zip(all_keys, combo))

        # Apply config params to template config (in-memory)
        saved = {}
        for k in param_grids:
            saved[k] = base_cfg.get(k)
            base_cfg[k] = combo_dict[k]

        # Apply CLI params
        for k in cli_param_grids:
            if k not in saved:
                saved[k] = base_cfg.get(k)
            base_cfg[k] = combo_dict[k]

        mp = combo_dict.get('max_positions', args.max_positions)

        # Progress
        elapsed = time.time() - start_time
        eta = (elapsed / max(ci, 1)) * (n_combos - ci) / 60 if ci > 0 else est_min
        settings_str = " | ".join(f"{k}={v}" for k, v in combo_dict.items())
        pct = (ci + 1) / n_combos * 100
        bar_len = 30
        filled = int(bar_len * (ci + 1) / n_combos)
        bar = '#' * filled + '-' * (bar_len - filled)
        sys.stdout.write(f"\r  [{bar}] {ci+1}/{n_combos} ({pct:.0f}%)  {settings_str}  ETA {eta:.1f}m   ")
        sys.stdout.flush()

        try:
            # Suppress all stdout from run_backtest (strategy loader, tqdm, trade logs)
            with contextlib.redirect_stdout(io.StringIO()):
                eq_vals, cl_trades = run_backtest_fn(
                    df, rule_signals, model, feature_cols, device, ruleset,
                    model_path=model_path, data_path=data_path, norm_stats=norm_stats,
                    use_fuzzy_sizing=args.use_fuzzy_sizing, use_trade_rules=args.use_trade_rules,
                    use_diffusion=args.use_diffusion, limit=args.limit,
                    v43_outputs=v43_outputs, bundle=bundle,
                    verbose_sim=False, max_positions=mp,
                    allowed_strategy_idxs=allowed_strategy_idxs,
                    allowed_template_ids=allowed_template_ids,
                    use_profit_targets=getattr(args, 'profittargets', False),
                    dte_by_dow=dte_by_dow,
                    bar_cache=_bar_cache,
                )
        except Exception as e:
            print(f"  ERROR: {e}")
            for k, v in saved.items():
                base_cfg[k] = v
            continue

        # Restore config for next iteration
        for k, v in saved.items():
            base_cfg[k] = v

        # Compute metrics
        if not eq_vals or len(eq_vals) < 2:
            print("  -> no data")
            continue

        eq_arr = np.array(eq_vals)
        start_bal = eq_arr[0]
        net_pnl = eq_arr[-1] - start_bal
        net_pct = net_pnl / start_bal * 100
        running_max = np.maximum.accumulate(eq_arr)
        dd_pct = (eq_arr - running_max) / running_max * 100
        max_dd = abs(np.min(dd_pct))
        np_dd = net_pct / max_dd if max_dd > 0.01 else (net_pct * 100 if net_pct > 0 else 0)

        # Trade stats
        pnls = [t.get('pnl', 0) for t in cl_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        n_wins = len(wins)
        n_losses = len(losses)
        wr = n_wins / max(n_wins + n_losses, 1) * 100
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        gross_win = sum(wins)
        gross_loss = abs(sum(losses))
        pf = gross_win / gross_loss if gross_loss > 0 else (999 if gross_win > 0 else 0)
        expectancy = (avg_win * wr / 100) - (avg_loss * (1 - wr / 100))

        # Sharpe (from equity curve bar-to-bar returns)
        returns = np.diff(eq_arr) / eq_arr[:-1]
        sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252 * 78)) if np.std(returns) > 0 else 0

        # Max consecutive losses
        max_consec_l = 0
        cur_consec = 0
        for p in pnls:
            if p <= 0:
                cur_consec += 1
                max_consec_l = max(max_consec_l, cur_consec)
            else:
                cur_consec = 0

        print(f"\n  #{ci+1:>4}  NP=${net_pnl:+,.0f}  DD={max_dd:.2f}%  Ratio={np_dd:.1f}  W={n_wins} L={n_losses}")

        results.append({
            'combo': combo_dict,
            'net_pnl': net_pnl, 'net_pct': net_pct,
            'max_dd': max_dd, 'np_dd': np_dd,
            'wins': n_wins, 'losses': n_losses, 'wr': wr,
            'sharpe': sharpe, 'expectancy': expectancy,
            'pf': pf, 'max_consec_l': max_consec_l,
            'total_trades': n_wins + n_losses,
        })

        # Save after every iteration (crash-safe)
        _save_csv()

    total_time = (time.time() - start_time) / 60
    print(f"\n  Optimization complete: {len(results)} results in {total_time:.1f} minutes")

    if not results:
        print("  No valid results.")
        return

    # Sort by NP/DD ratio
    results.sort(key=lambda x: x['np_dd'], reverse=True)
    top = results[:100]

    # Display table
    print()
    print("=" * 140)
    print(f"  OPTIMIZER RESULTS -- {template_id} ({len(results)} combos, ranked by NP/DD)")
    print("=" * 140)

    # Build header from first combo's keys
    pk = list(top[0]['combo'].keys())
    hdr = f"  {'#':>3}  "
    for k in pk:
        w = max(len(k), 6)
        hdr += f"{k:>{w}}  "
    hdr += f" |  {'NP$':>10}  {'DD%':>6}  {'NP/DD':>6}  {'W':>3}  {'L':>3}  {'WR%':>5}  {'Sharpe':>6}  {'Expect':>7}  {'PF':>5}  {'MCL':>3}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for ri, r in enumerate(top):
        row = f"  {ri+1:>3}  "
        for k in pk:
            w = max(len(k), 6)
            v = r['combo'][k]
            if isinstance(v, float):
                row += f"{v:>{w}.2f}  "
            else:
                row += f"{v:>{w}}  "
        row += (f" |  ${r['net_pnl']:>9,.0f}  {r['max_dd']:>5.2f}%  {r['np_dd']:>6.1f}"
                f"  {r['wins']:>3}  {r['losses']:>3}  {r['wr']:>4.1f}%"
                f"  {r['sharpe']:>6.2f}  ${r['expectancy']:>6,.0f}  {r['pf']:>5.2f}"
                f"  {r['max_consec_l']:>3}")
        print(row)

    print("=" * 140)

    # Final CSV save (already saved incrementally)
    _save_csv()
    print(f"  Results saved to {csv_path}")

    # Prompt for selection
    print()
    sel = input("  Select row number to apply (or 'q' to quit): ").strip()
    if sel.lower() == 'q' or not sel.isdigit():
        print("  No changes applied.")
        return

    sel_idx = int(sel) - 1
    if sel_idx < 0 or sel_idx >= len(top):
        print(f"  Invalid selection: {sel}")
        return

    chosen = top[sel_idx]
    print(f"\n  Applying row {sel_idx + 1}: {chosen['combo']}")

    # Write to strategy .py file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    strat_file = os.path.join(script_dir, "strategies", f"{template_id}.py")

    if not os.path.exists(strat_file):
        print(f"  ERROR: Strategy file not found: {strat_file}")
        return

    with open(strat_file, 'r', encoding='utf-8') as f:
        slines = f.read()

    for key, val in chosen['combo'].items():
        if key == 'max_positions':
            continue  # CLI param, not in config file
        # Match: "key":  old_value,  or  "key":  old_value
        pattern = rf'"({re.escape(key)}":\s*)[^,\n]+'
        if isinstance(val, float):
            replacement = rf'"{key}":     {val}'
        else:
            replacement = rf'"{key}":      {val}'
        slines_new = re.sub(pattern, replacement, slines, count=1)
        if slines_new != slines:
            print(f"    Updated {key} = {val}")
            slines = slines_new
        else:
            print(f"    WARN: Could not find {key} in {strat_file}")

    with open(strat_file, 'w', encoding='utf-8') as f:
        f.write(slines)

    print(f"\n  Saved to {strat_file}")
    if 'max_positions' in chosen['combo']:
        print(f"  NOTE: max_positions={chosen['combo']['max_positions']} is a CLI arg."
              f" Use: --max-positions {chosen['combo']['max_positions']}")
    print(f"  Run: python kaggle/condor_brain_backtest_v45.py --use-v43"
          f" --strategies {template_id} --limit {args.limit or 500}"
          f" --profittargets --verbose")
