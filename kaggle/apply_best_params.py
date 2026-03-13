"""
apply_best_params.py
====================
Reads all optimizer sweep report files, picks the best objective per strategy
across all sweeps, then applies those best params directly to the strategy .py files.

Usage:
    python kaggle/apply_best_params.py

Report files expected in: kaggle/reports/
Strategy files expected in: kaggle/strategies/
"""

import os
import re
import sys

# Force UTF-8 output on Windows so box-drawing chars (in source files) don't crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
STRATEGIES_DIR = os.path.join(os.path.dirname(__file__), "strategies")

SWEEP_FILES = {
    "min":     "GPU_Optimization_Sweep_min.txt",
    "min_v2":  "GPU_Optimization_Sweep_min_v2.txt",
    "min-med": "GPU_Optimization_Sweep_min-med.txt",
    "med":     "GPU_Optimization_Sweep_med.txt",
    "med-max": "GPU_Optimization_Sweep_med-max.txt",
}

# Params that must be written as int
INT_PARAMS = {"stop_loss_dollar", "spread_width", "target_dte", "hold_days", "max_dte_exit"}
# Params that are int but named "profit_target" in the files
INT_PARAMS.add("profit_target")
# short_delta is float (4 decimals)

# ---------------------------------------------------------------------------
# Parser helpers
# ---------------------------------------------------------------------------

def _parse_summary_section(lines):
    """
    Parse the summary table that looks like:
      STRATEGY                               obj     net%     dd%   trades  status
      bear_call_ladder                     0.501     2.06    2.11       24  applied
      ...
    Returns dict: strategy_name -> {"obj": float, "net_pct": float, "dd_pct": float}
    Only includes strategies with numeric obj (skips SKIPPED / no results rows).
    """
    result = {}
    in_table = False
    header_found = False
    for line in lines:
        stripped = line.strip()
        # Detect header
        if re.match(r'STRATEGY\s+obj\s+net%', stripped):
            in_table = True
            header_found = True
            continue
        if not header_found:
            continue
        if not in_table:
            continue
        # Separator line
        if stripped.startswith("──") or stripped.startswith("══") or stripped == "":
            if in_table and header_found:
                # Could be end of table or just separator - continue
                continue
        # Stop if we hit the next major section header
        if stripped.startswith("[autoall]") or stripped.startswith("OPTIMIZED PARAMETERS"):
            break
        # Parse data row: try to match strategy + numbers
        # Format: "  strategy_name   <float>   <float>   <float>  <int>  <status>"
        # Skip SKIPPED lines and "no results" lines
        if "SKIPPED" in stripped or "no results" in stripped:
            continue
        # Match: name  obj  net%  dd%  trades  status
        m = re.match(
            r'\s+(\S+)\s+([-\d.]+)\s+([-+\d.]+)\s+([\d.]+)\s+(\d+)\s+(\S+)',
            line
        )
        if m:
            name = m.group(1)
            try:
                obj = float(m.group(2))
                net_pct = float(m.group(3))
                dd_pct = float(m.group(4))
                result[name] = {"obj": obj, "net_pct": net_pct, "dd_pct": dd_pct}
            except ValueError:
                pass
    return result


def _parse_params_section(lines):
    """
    Parse the OPTIMIZED PARAMETERS BY STRATEGY table:
      STRATEGY                          stop_loss_d  profit_targ  spread_widt   target_dte    hold_days  max_dte_exi  short_delta    wins  losses     net_pnl     net%     dd%
    Columns are fixed-width — we parse by position after locating the header.

    Returns dict: strategy_name -> param_dict
    N/A rows are returned with None values.
    """
    result = {}
    header_line = None
    header_idx = None

    for i, line in enumerate(lines):
        if re.match(r'\s+STRATEGY\s+stop_loss_d\s+profit_targ', line):
            header_line = line
            header_idx = i
            break

    if header_line is None:
        return result

    # Parse column start positions from the header line
    # Columns: STRATEGY, stop_loss_d, profit_targ, spread_widt, target_dte, hold_days, max_dte_exi, short_delta, wins, losses, net_pnl, net%, dd%
    # We only care about the first 8 data columns (after STRATEGY)
    col_names = [
        "stop_loss_dollar",  # stop_loss_d
        "profit_target",     # profit_targ
        "spread_width",      # spread_widt
        "target_dte",        # target_dte
        "hold_days",         # hold_days
        "max_dte_exit",      # max_dte_exi
        "short_delta",       # short_delta
    ]

    # Find column positions by locating header tokens
    tokens_in_header = [
        "stop_loss_d", "profit_targ", "spread_widt",
        "target_dte", "hold_days", "max_dte_exi", "short_delta",
        "wins"
    ]
    col_positions = []
    for tok in tokens_in_header:
        pos = header_line.find(tok)
        col_positions.append(pos)

    # Data rows follow the header + separator line
    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if stripped.startswith("──"):
            continue
        if stripped == "" or stripped.startswith("[autoall]") or stripped.startswith("COMPLETE"):
            continue
        # Stop at next major section
        if stripped.startswith("GPU ") or stripped.startswith("⚡"):
            break

        # Extract strategy name: everything up to the first data column
        strat_end = col_positions[0]
        if len(line) < strat_end:
            continue
        strat_name = line[:strat_end].strip()
        if not strat_name:
            continue

        # Check for N/A row
        if "N/A" in line:
            result[strat_name] = {k: None for k in col_names}
            continue

        # Extract values by column positions
        params = {}
        for i, col_name in enumerate(col_names):
            start = col_positions[i]
            end = col_positions[i + 1] if i + 1 < len(col_positions) else len(line)
            raw = line[start:end].strip() if len(line) > start else ""
            # Remove trailing +/- from wins etc — only care about the 7 param cols
            raw = raw.split()[0] if raw.split() else ""
            if raw in ("None", "", "N/A"):
                params[col_name] = None
            else:
                try:
                    params[col_name] = float(raw)
                except ValueError:
                    params[col_name] = None

        result[strat_name] = params

    return result


def parse_sweep_file(filepath, sweep_name):
    """
    Parse one sweep file. Returns:
      summary: dict  strategy -> {obj, net_pct, dd_pct}
      params:  dict  strategy -> {stop_loss_dollar, profit_target, spread_width,
                                   target_dte, hold_days, max_dte_exit, short_delta}
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    summary = _parse_summary_section(lines)
    params = _parse_params_section(lines)
    return summary, params


# ---------------------------------------------------------------------------
# Apply params to .py file
# ---------------------------------------------------------------------------

def _apply_params(strategy_name, params, obj, sweep_name):
    """Apply best params to the strategy .py file using regex replacement."""
    strat_file = os.path.join(STRATEGIES_DIR, f"{strategy_name}.py")
    if not os.path.exists(strat_file):
        return False, f"file not found: {strat_file}"

    with open(strat_file, "r", encoding="utf-8") as f:
        src = f.read()

    modified = False
    for param_name, value in params.items():
        if value is None:
            # Skip None params — leave the strategy file's existing None in place
            continue

        pattern = rf'("{param_name}"\s*:\s*)([^,\n]+)'

        if param_name in INT_PARAMS:
            repl = rf'\g<1>{int(round(value))}'
        elif param_name == "short_delta":
            repl = rf'\g<1>{float(value):.4f}'
        else:
            repl = rf'\g<1>{float(value):.4f}'

        new_src = re.sub(pattern, repl, src)
        if new_src != src:
            src = new_src
            modified = True

    if modified:
        with open(strat_file, "w", encoding="utf-8") as f:
            f.write(src)
        return True, "applied"
    else:
        return False, "no changes matched"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ── Step 1: Parse all 4 sweep files ──────────────────────────────────────
    all_sweeps = {}  # sweep_name -> (summary_dict, params_dict)

    for sweep_name, filename in SWEEP_FILES.items():
        filepath = os.path.join(REPORTS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  [warn] Missing report file: {filepath}")
            continue
        summary, params = parse_sweep_file(filepath, sweep_name)
        all_sweeps[sweep_name] = (summary, params)
        print(f"  [parse] {sweep_name}: {len(summary)} strategies in summary, "
              f"{len(params)} in params table")

    # ── Step 2: Collect all strategy names ───────────────────────────────────
    all_strategies = set()
    for sweep_name, (summary, params) in all_sweeps.items():
        all_strategies.update(summary.keys())
        all_strategies.update(params.keys())

    # ── Step 3: For each strategy, pick the sweep with highest obj ────────────
    best = {}  # strategy -> {sweep, obj, net_pct, dd_pct, params}

    for strat in sorted(all_strategies):
        best_obj = None
        best_entry = None

        for sweep_name, (summary, params) in all_sweeps.items():
            if strat not in summary:
                continue
            info = summary[strat]
            obj = info["obj"]
            # Skip negative-100 sentinel (broken wing dead strategies in med sweep)
            if obj <= -100:
                continue
            strat_params = params.get(strat, {})
            # Skip N/A rows (all params None)
            if strat_params and all(v is None for v in strat_params.values()):
                continue

            if best_obj is None or obj > best_obj:
                best_obj = obj
                best_entry = {
                    "sweep": sweep_name,
                    "obj": obj,
                    "net_pct": info["net_pct"],
                    "dd_pct": info["dd_pct"],
                    "params": strat_params,
                }

        if best_entry is not None:
            best[strat] = best_entry

    # ── Step 4: Apply params to strategy .py files ───────────────────────────
    results = []  # (strategy, sweep, obj, net_pct, dd_pct, status)

    for strat in sorted(best.keys()):
        entry = best[strat]
        obj = entry["obj"]

        if obj <= 0:
            results.append((strat, entry["sweep"], obj, entry["net_pct"], entry["dd_pct"], "skipped (obj<=0)"))
            continue

        ok, msg = _apply_params(strat, entry["params"], obj, entry["sweep"])
        results.append((strat, entry["sweep"], obj, entry["net_pct"], entry["dd_pct"], msg))

    # ── Step 5: Print summary table ──────────────────────────────────────────
    print()
    sep = "-" * 93
    print("=" * 95)
    print(f"  {'STRATEGY':<36}  {'BEST_SWEEP':<10}  {'obj':>7}  {'net%':>7}  {'dd%':>6}  {'STATUS'}")
    print("  " + sep)
    applied_count = 0
    skipped_count = 0
    for strat, sweep, obj, net_pct, dd_pct, status in results:
        net_str = f"{net_pct:+.2f}%"
        dd_str  = f"{dd_pct:.2f}%"
        print(f"  {strat:<36}  {sweep:<10}  {obj:7.3f}  {net_str:>7}  {dd_str:>6}  {status}")
        if status == "applied":
            applied_count += 1
        else:
            skipped_count += 1
    print("  " + sep)
    print(f"  Applied: {applied_count}   Skipped: {skipped_count}")
    print("=" * 95)


if __name__ == "__main__":
    main()
