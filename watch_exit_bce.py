"""
watch_exit_bce.py — Parse CondorNet v4.3 training log to track exit_bce and all
                    loss components across epochs.

Two data sources extracted from the log:
  1. [EPOCH COMPONENTS] lines  — full per-epoch averages (present after the
                                  condor_train_net_v43.py patch is pulled)
  2. [DEBUG BATCH 0] block     — all component values at batch 0, epoch 1 only
  3. [B####] verbose lines     — 5 visible components + total loss (gap proxy)

Usage:
    python watch_exit_bce.py train_v43_t4_17th_test.log
    python watch_exit_bce.py          # auto-finds newest .log in current dir
"""
import sys
import re
import glob
import os
from collections import defaultdict

# Loss weights from configs/loss_weights_v43.json — needed for gap proxy
WEIGHTS = {
    "strategy_ce":           1.0,
    "pop_bce":               1.0,
    "ev_mse":                0.5,
    "risk_mse":              0.5,
    "size_mse":              0.4,
    "spot_mse":              0.0,
    "fuzzy_var":             0.2,
    "pattern_ent":           0.1,
    "robust":                0.2,
    "consensus_consistency": 0.05,
    "exit_bce":              0.5,
}
VISIBLE_5 = ["strategy_ce", "pop_bce", "ev_mse", "risk_mse", "size_mse"]


def find_log():
    candidates = glob.glob("*.log") + glob.glob("logs/*.log")
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def parse_log(log_path):
    print(f"Reading: {log_path}\n")
    with open(log_path, "r", errors="replace") as f:
        lines = f.readlines()

    epochs = {}          # ep -> dict of collected data
    current_epoch = None
    in_debug = False
    debug_epoch = None
    debug_comps = {}

    for line in lines:
        line = line.rstrip()

        # ── Epoch summary line ──────────────────────────────────────────────
        m = re.search(r"Epoch\s+(\d+)\s*\|\s*Train:\s*([\d.]+)\s*\|\s*Val:\s*([\d.]+)", line)
        if m:
            ep = int(m.group(1))
            current_epoch = ep
            if ep not in epochs:
                epochs[ep] = {"batch_data": [], "epoch_components": None, "debug_comps": None}
            epochs[ep]["train_loss"] = float(m.group(2))
            epochs[ep]["val_loss"]   = float(m.group(3))
            continue

        # ── [EPOCH COMPONENTS] line (after patch) ──────────────────────────
        if "[EPOCH COMPONENTS]" in line and current_epoch is not None:
            comps = {}
            for m2 in re.finditer(r"(\w+)=([\d.]+)", line):
                comps[m2.group(1)] = float(m2.group(2))
            if comps:
                epochs[current_epoch]["epoch_components"] = comps
            continue

        # ── DEBUG BATCH 0 block (epoch 1 only) ─────────────────────────────
        if "[DEBUG BATCH 0] System Inventory" in line:
            in_debug = True
            debug_epoch = current_epoch if current_epoch is not None else 1
            debug_comps = {}
            continue

        if in_debug:
            m2 = re.match(r"\s+-\s+(\w+):\s+([\d.]+)", line)
            if m2:
                debug_comps[m2.group(1)] = float(m2.group(2))
            else:
                # Block ended on any non-component line
                in_debug = False
                if debug_epoch is not None:
                    if debug_epoch not in epochs:
                        epochs[debug_epoch] = {"batch_data": [], "epoch_components": None, "debug_comps": None}
                    epochs[debug_epoch]["debug_comps"] = dict(debug_comps)
                debug_epoch = None
                debug_comps = {}
            continue

        # ── [B####] verbose batch lines ─────────────────────────────────────
        m3 = re.search(r"\[B(\d+)\]\s+loss=([\d.]+)\s*\|(.*)", line)
        if m3 and current_epoch is not None:
            batch_loss = float(m3.group(2))
            comps = {"loss": batch_loss}
            for m4 in re.finditer(r"(\w+):([\d.]+)", m3.group(3)):
                comps[m4.group(1)] = float(m4.group(2))
            epochs[current_epoch]["batch_data"].append(comps)

    return epochs


def print_report(epochs):
    if not epochs:
        print("No epoch data found in log.")
        return

    sorted_eps = sorted(epochs.keys())
    best_val = float("inf")

    # ── Header ─────────────────────────────────────────────────────────────
    print("=" * 72)
    print("  CONDORNET v4.3 — LOSS COMPONENT TRACKER")
    print("=" * 72)

    # ── Epoch summary ──────────────────────────────────────────────────────
    print(f"\n{'Ep':>4}  {'Train':>8}  {'Val':>8}  {'ΔVal':>8}  {'Best':>5}")
    print("-" * 42)
    prev_val = None
    best_epochs = []   # (epoch, val_loss) for each new best
    for ep in sorted_eps:
        d = epochs[ep]
        train = d.get("train_loss", float("nan"))
        val   = d.get("val_loss",   float("nan"))
        delta = (val - prev_val) if prev_val is not None else 0.0
        flag  = " <-- BEST" if val < best_val else ""
        if val < best_val:
            best_val = val
            best_epochs.append((ep, val))
        print(f"  {ep:3d}  {train:8.4f}  {val:8.4f}  {delta:+8.4f}  {flag}")
        prev_val = val

    # ── Best-epoch cadence table ────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  IMPROVEMENT CADENCE — epochs waited between each new best")
    print("=" * 72)
    print(f"  {'Best#':>5}  {'Epoch':>6}  {'Val':>8}  {'Wait':>6}  {'ΔVal':>9}  bar")
    print("-" * 55)
    for i, (ep, val) in enumerate(best_epochs):
        wait  = ep - best_epochs[i-1][0] if i > 0 else 0
        dval  = val - best_epochs[i-1][1] if i > 0 else 0.0
        bar   = "█" * min(wait, 20)
        print(f"  {i+1:5d}  {ep:6d}  {val:8.4f}  {wait:6d}  {dval:+9.4f}  {bar}")
    # Trailing drought: epochs after last best to end of log
    last_ep   = sorted_eps[-1]
    last_best = best_epochs[-1][0] if best_epochs else 0
    drought   = last_ep - last_best
    print("-" * 55)
    print(f"  {'END':>5}  {last_ep:6d}  {'':8}  {drought:6d}  {'no improvement':>9}  "
          f"{'█' * min(drought, 20)}  ← final drought")
    print(f"\n  Total best epochs : {len(best_epochs)}")
    print(f"  Max wait for best : {max(e-best_epochs[i-1][0] for i,(e,_) in enumerate(best_epochs) if i>0) if len(best_epochs)>1 else 0} epochs")
    print(f"  Final drought     : {drought} epochs (patience consumed: {drought}/20)")

    # ── [EPOCH COMPONENTS] data (full, from patch) ─────────────────────────
    has_epoch_comps = any(epochs[ep].get("epoch_components") for ep in sorted_eps)
    if has_epoch_comps:
        print()
        print("=" * 72)
        print("  PER-EPOCH COMPONENT AVERAGES  (from [EPOCH COMPONENTS] lines)")
        print("  Values are unweighted averages across all training batches.")
        print("=" * 72)

        # Find all component names seen
        all_comp_names = []
        for ep in sorted_eps:
            ec = epochs[ep].get("epoch_components") or {}
            for k in ec:
                if k not in all_comp_names:
                    all_comp_names.append(k)

        # Print exit_bce and key components
        key_comps = ["strategy_ce", "pop_bce", "ev_mse", "risk_mse", "exit_bce",
                     "robust", "consensus_consistency"]
        print_comps = [c for c in key_comps if c in all_comp_names]

        col_w = 10
        header = f"  {'Ep':>3}  " + "  ".join(f"{c:>{col_w}}" for c in print_comps)
        print(header)
        print("-" * len(header))

        exit_bce_vals = []
        for ep in sorted_eps:
            ec = epochs[ep].get("epoch_components")
            if not ec:
                continue
            row = f"  {ep:3d}  " + "  ".join(
                f"{ec.get(c, float('nan')):>{col_w}.4f}" for c in print_comps
            )
            print(row)
            if "exit_bce" in ec:
                exit_bce_vals.append((ep, ec["exit_bce"]))

        if exit_bce_vals:
            print()
            print("  exit_bce trajectory:")
            print(f"    epoch 1  →  {exit_bce_vals[0][1]:.4f}  (ln(2)=0.6931 = zero-init baseline)")
            if len(exit_bce_vals) > 1:
                start = exit_bce_vals[0][1]
                end   = exit_bce_vals[-1][1]
                drop  = start - end
                print(f"    epoch {exit_bce_vals[-1][0]:2d}  →  {end:.4f}  "
                      f"(drop: {drop:+.4f}  {'LEARNING' if drop > 0.005 else 'FLAT — check labels'})")
            print(f"    target floor: ~0.66 (base rate), good fit: ~0.58-0.63")

    # ── Debug batch 0 (epoch 1 only) ──────────────────────────────────────
    debug_found = [(ep, epochs[ep]["debug_comps"]) for ep in sorted_eps
                   if epochs[ep].get("debug_comps")]
    if debug_found:
        print()
        print("=" * 72)
        print("  DEBUG BATCH 0 — Full component snapshot (epoch 1, batch 0)")
        print("=" * 72)
        for ep, dc in debug_found:
            print(f"  Epoch {ep} batch 0:")
            for k, v in dc.items():
                w = WEIGHTS.get(k, "?")
                baseline = " ← ln(2), zero-init" if k == "exit_bce" and abs(v - 0.693147) < 0.001 else ""
                print(f"    {k:<28} {v:.6f}  (weight={w}){baseline}")

    # ── Gap proxy (if no [EPOCH COMPONENTS] lines yet) ─────────────────────
    if not has_epoch_comps:
        print()
        print("=" * 72)
        print("  GAP PROXY  (exit_bce estimate from batch data)")
        print("  gap = total_loss - weighted_sum(5_visible_components)")
        print("  exit_bce_est = (gap - ~0.025) / 0.5   [rough — ±0.05 error]")
        print("  NOTE: Pull the updated condor_train_net_v43.py to get exact values")
        print("=" * 72)
        print(f"  {'Ep':>3}  {'avg_total':>10}  {'w_sum_5':>10}  {'gap':>8}  {'exit_bce_est':>13}")
        print("-" * 55)
        for ep in sorted_eps:
            bd = epochs[ep].get("batch_data", [])
            if not bd:
                continue
            total_vals = [b["loss"] for b in bd if "loss" in b]
            if not total_vals:
                continue
            avg_total = sum(total_vals) / len(total_vals)

            # weighted sum of 5 visible components
            w_sum = 0.0
            for c in VISIBLE_5:
                vals = [b[c] for b in bd if c in b]
                if vals:
                    w_sum += (sum(vals) / len(vals)) * WEIGHTS.get(c, 1.0)

            gap = avg_total - w_sum
            # subtract ~other known small terms: robust*0.2~0.03, consensus*0.05~0.01,
            # pattern_ent*0.1~0.001, logit_penalties~0.002, fuzzy_var*0.2~0.001
            other_est = 0.025
            exit_bce_est = (gap - other_est) / 0.5
            print(f"  {ep:3d}  {avg_total:10.4f}  {w_sum:10.4f}  {gap:8.4f}  {exit_bce_est:13.4f}")

        print()
        print("  Interpretation:")
        print("    exit_bce_est ~ 0.69  →  Still at zero-init (not learning yet)")
        print("    exit_bce_est ~ 0.65  →  Learning base rate (39% exit=1)")
        print("    exit_bce_est < 0.63  →  Conditional exit timing learned")


def main():
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        log_path = find_log()
        if log_path is None:
            print("ERROR: No .log file found. Pass the log filename as argument:")
            print("  python watch_exit_bce.py train_v43_t4_17th_test.log")
            sys.exit(1)

    epochs = parse_log(log_path)
    print_report(epochs)


if __name__ == "__main__":
    main()
