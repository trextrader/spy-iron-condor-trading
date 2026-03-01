#!/usr/bin/env python3
"""
diagnose_trades.py
------------------
Run on Lightning AI after a backtest:

    python diagnose_trades.py [path/to/trades.csv]

Defaults to reports/trades.csv (or reports/run18_Backtest/trades.csv if that exists).
"""
import sys
import os
import csv
import collections

SEP  = "=" * 65
SEP2 = "-" * 65

# ── locate trades.csv ─────────────────────────────────────────────────────────
CANDIDATES = [
    "reports/trades_v2.csv",
    "reports/trades.csv",
    "reports/run18_Backtest/trades.csv",
]

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = None
    for c in CANDIDATES:
        if os.path.exists(c):
            path = c
            break

if path is None or not os.path.exists(path):
    print(f"ERROR: trades.csv not found. Tried: {CANDIDATES}")
    print("Usage: python diagnose_trades.py <path/to/trades.csv>")
    sys.exit(1)

print(f"\n{SEP}")
print(f"  BACKTEST TRADE DIAGNOSTIC  |  {path}")
print(f"{SEP}\n")

rows = list(csv.DictReader(open(path, encoding="utf-8")))
print(f"Total rows in CSV: {len(rows)}")

# ── split OPEN vs CLOSE ───────────────────────────────────────────────────────
opens  = [r for r in rows if r.get("action") == "OPEN"]
closes = [r for r in rows if r.get("action") != "OPEN" and r.get("reason")]

print(f"  OPEN  records : {len(opens)}")
print(f"  CLOSE records : {len(closes)}")

# Trade IDs in opens but not in closes = still open at sim end
open_ids  = {r["trade_id"] for r in opens}
close_ids = {r["trade_id"] for r in closes}
unclosed  = open_ids - close_ids
print(f"  Unclosed (open at sim end): {len(unclosed)}")

if unclosed:
    unclosed_opens = [r for r in opens if r["trade_id"] in unclosed]
    total_margin   = sum(float(r.get("max_loss") or 0) for r in unclosed_opens)
    print(f"    Tied-up margin (unrealized): ${total_margin:,.0f}")
    print(f"    Trade IDs: {sorted(unclosed)}")

# ── PnL by exit reason ────────────────────────────────────────────────────────
by_reason = collections.defaultdict(lambda: {"count": 0, "wins": 0, "losses": 0, "total": 0.0})
for r in closes:
    reason = r.get("reason") or "UNKNOWN"
    try:
        pnl = float(r["pnl"])
    except (KeyError, ValueError, TypeError):
        pnl = 0.0
    by_reason[reason]["count"]  += 1
    by_reason[reason]["total"]  += pnl
    if pnl > 0:
        by_reason[reason]["wins"] += 1
    elif pnl < 0:
        by_reason[reason]["losses"] += 1

print(f"\n{SEP2}")
print(f"  EXIT REASON BREAKDOWN")
print(f"{SEP2}")
hdr = f"  {'Reason':<22} {'N':>4} {'W':>4} {'L':>4} {'Total PnL':>12} {'Avg PnL':>9} {'Win%':>6}"
print(hdr)
print(f"  {'-'*60}")
grand = 0.0
for reason, d in sorted(by_reason.items(), key=lambda x: x[1]["total"]):
    avg  = d["total"] / d["count"] if d["count"] else 0
    winp = 100.0 * d["wins"] / d["count"] if d["count"] else 0
    print(f"  {reason:<22} {d['count']:>4} {d['wins']:>4} {d['losses']:>4} {d['total']:>12,.2f} {avg:>9,.2f} {winp:>5.1f}%")
    grand += d["total"]

print(f"  {'-'*60}")
print(f"  {'TOTAL':<22} {len(closes):>4} {'':>4} {'':>4} {grand:>12,.2f}")

# ── PER_TRADE_STOP deep-dive ──────────────────────────────────────────────────
pts = [r for r in closes if r.get("reason") == "PER_TRADE_STOP"]
if pts:
    print(f"\n{SEP2}")
    print(f"  PER_TRADE_STOP deep-dive  ({len(pts)} trades)")
    print(f"{SEP2}")
    open_map = {r["trade_id"]: r for r in opens}
    print(f"  {'TradeID':<12} {'Credit':>8} {'MaxLoss':>9} {'PnL':>9} {'PnL/ML':>7}  Entry -> Exit")
    print(f"  {'-'*62}")
    for r in sorted(pts, key=lambda x: float(x.get("pnl") or 0)):
        tid    = r["trade_id"]
        pnl    = float(r.get("pnl") or 0)
        o      = open_map.get(tid, {})
        credit = float(o.get("entry_credit") or 0)
        ml     = float(o.get("max_loss") or 1)
        ratio  = pnl / ml if ml else 0
        entry  = o.get("dt", "?")[:16]
        exit_  = (r.get("exit_dt") or "?")[:16]
        print(f"  {tid:<12} {credit:>8.4f} {ml:>9,.0f} {pnl:>9,.2f} {ratio:>6.1%}  {entry} -> {exit_}")

    avg_loss = sum(float(r.get("pnl") or 0) for r in pts) / len(pts)
    print(f"\n  Average PER_TRADE_STOP loss: ${avg_loss:,.2f}")

# ── EXPIRED deep-dive ─────────────────────────────────────────────────────────
expired = [r for r in closes if r.get("reason") == "EXPIRED"]
if expired:
    print(f"\n{SEP2}")
    print(f"  EXPIRED trades deep-dive  ({len(expired)} trades)")
    print(f"{SEP2}")
    open_map = {r["trade_id"]: r for r in opens}
    print(f"  {'TradeID':<12} {'Credit':>8} {'MaxLoss':>9} {'PnL':>9} {'PnL/ML':>7}")
    print(f"  {'-'*52}")
    for r in sorted(expired, key=lambda x: float(x.get("pnl") or 0)):
        tid    = r["trade_id"]
        pnl    = float(r.get("pnl") or 0)
        o      = open_map.get(tid, {})
        credit = float(o.get("entry_credit") or 0)
        ml     = float(o.get("max_loss") or 1)
        ratio  = pnl / ml if ml else 0
        print(f"  {tid:<12} {credit:>8.4f} {ml:>9,.0f} {pnl:>9,.2f} {ratio:>6.1%}")
    avg_loss = sum(float(r.get("pnl") or 0) for r in expired) / len(expired)
    print(f"\n  Average EXPIRED loss: ${avg_loss:,.2f}")

# ── equity curve summary ──────────────────────────────────────────────────────
ec_path = os.path.join(os.path.dirname(os.path.abspath(path)), "equity_curve.csv")
if os.path.exists(ec_path):
    ec_rows  = list(csv.DictReader(open(ec_path, encoding="utf-8")))
    equities = []
    for r in ec_rows:
        try:
            equities.append(float(r.get("equity") or r.get("Equity") or 0))
        except Exception:
            pass
    if equities:
        peak   = max(equities)
        trough = min(equities)
        final  = equities[-1]
        start  = equities[0]
        dd     = (peak - trough) / peak * 100 if peak > 0 else 0
        print(f"\n{SEP2}")
        print(f"  EQUITY CURVE SUMMARY")
        print(f"{SEP2}")
        print(f"  Start :  ${start:>12,.2f}")
        print(f"  Peak  :  ${peak:>12,.2f}")
        print(f"  Trough:  ${trough:>12,.2f}")
        print(f"  Final :  ${final:>12,.2f}")
        print(f"  Max DD:  {dd:.1f}%  (peak->trough)")
        print(f"  Net   :  ${final - start:>+12,.2f}  ({(final/start-1)*100:+.1f}%)")
else:
    print(f"\n  (equity_curve.csv not found at {ec_path})")

print(f"\n{SEP}\n")
