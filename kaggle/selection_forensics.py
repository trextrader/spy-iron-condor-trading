"""
Replay helper for inspecting top optimizer candidates from a results CSV.
"""

from __future__ import annotations

import argparse
import csv


def compute_np_dd(net_pct: float, max_dd: float) -> float:
    if max_dd > 0.01:
        return net_pct / max_dd
    return net_pct * 100.0 if net_pct > 0 else 0.0


def load_top_rows(csv_path: str, top_n: int = 10) -> list[dict]:
    rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    rows.sort(key=lambda row: float(row.get("objective", "-1e18")), reverse=True)
    return rows[:top_n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Optimizer results CSV to inspect.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top rows to print.")
    args = parser.parse_args()

    rows = load_top_rows(args.csv, top_n=args.top_n)

    print("Top optimizer candidates:")
    for row in rows:
        net_pct = float(row.get("net_pct", 0.0))
        max_dd = abs(float(row.get("max_dd", 0.0)))
        np_dd = float(row["np_dd"]) if row.get("np_dd") else compute_np_dd(net_pct, max_dd)
        print(
            row.get("strategy"),
            row.get("objective"),
            f"legacy={row.get('legacy_objective', 'n/a')}",
            row.get("net_pct"),
            row.get("max_dd"),
            f"np_dd={np_dd:.3f}",
            f"eligible={row.get('eligible', 'n/a')}",
        )


if __name__ == "__main__":
    main()
