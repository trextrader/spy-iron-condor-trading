from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


def _read_jsonl(path: str) -> List[dict]:
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _read_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _wrap_lines(text: str, width: int = 110) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def _page_title(fig, title: str) -> None:
    fig.text(0.5, 0.96, title, ha="center", va="top", fontsize=14, weight="bold")


def _page_text(fig, y: float, lines: Iterable[str], size: int = 10) -> None:
    text = "\n".join(lines)
    fig.text(0.05, y, text, ha="left", va="top", fontsize=size, family="monospace")


def _summarize_trace(records: List[dict]) -> Dict[str, int]:
    scope_counts = Counter()
    win_counts = Counter()
    loss_counts = Counter()
    neutral_counts = Counter()
    for rec in records:
        scope = rec.get("decision", {}).get("scope", "UNKNOWN")
        scope_counts[scope] += 1
        outcome = rec.get("outcome", {})
        if outcome.get("win"):
            win_counts[scope] += 1
        elif outcome.get("loss"):
            loss_counts[scope] += 1
        elif outcome.get("neutral"):
            neutral_counts[scope] += 1
    return {
        "scope_counts": scope_counts,
        "win_counts": win_counts,
        "loss_counts": loss_counts,
        "neutral_counts": neutral_counts,
    }


def _top_factors(df: pd.DataFrame, scope: str, n: int = 10) -> pd.DataFrame:
    if df.empty:
        return df
    scoped = df[df["scope"] == scope].copy()
    if scoped.empty:
        return scoped
    if "lift_avg_r" in scoped.columns:
        scoped = scoped.sort_values("lift_avg_r", ascending=False)
    return scoped.head(n)


def generate_pdf(
    inputs_schema_path: str,
    decision_trace_path: str,
    attribution_csv_path: str,
    output_path: str,
    auto_attrib: bool = False,
) -> None:
    if auto_attrib or not os.path.exists(attribution_csv_path):
        cmd = [
            sys.executable,
            os.path.join("audit", "generate_decision_factor_attribution.py"),
            "--trace",
            decision_trace_path,
            "--out",
            attribution_csv_path,
        ]
        try:
            subprocess.check_call(cmd)
        except Exception as exc:
            print(f"[WARN] Attribution generation failed: {exc}")

    inputs_schema = _read_json(inputs_schema_path)
    records = _read_jsonl(decision_trace_path)
    attrib_df = pd.read_csv(attribution_csv_path) if os.path.exists(attribution_csv_path) else pd.DataFrame()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    with PdfPages(output_path) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        _page_title(fig, "Accountability Audit Packet")
        _page_text(
            fig,
            0.90,
            [
                f"Generated (UTC): {now}",
                f"Inputs schema: {inputs_schema_path if inputs_schema_path else 'N/A'}",
                f"Decision trace: {decision_trace_path if decision_trace_path else 'N/A'}",
                f"Attribution CSV: {attribution_csv_path if attribution_csv_path else 'N/A'}",
            ],
        )
        notes = (
            "This packet summarizes decision provenance and factor attribution. "
            "It is intended for internal audit readiness and regulator-facing review. "
            "All figures are derived from the provided artifacts and are reproducible."
        )
        _page_text(fig, 0.80, [_wrap_lines(notes)])
        pdf.savefig(fig)
        plt.close(fig)

        summary = _summarize_trace(records)
        fig = plt.figure(figsize=(8.5, 11))
        _page_title(fig, "Decision Trace Summary")
        lines = ["Scope counts:"]
        for scope, count in summary["scope_counts"].items():
            wins = summary["win_counts"].get(scope, 0)
            losses = summary["loss_counts"].get(scope, 0)
            neutrals = summary["neutral_counts"].get(scope, 0)
            lines.append(f"  {scope}: total={count} wins={wins} losses={losses} neutral={neutrals}")
        if not lines[1:]:
            lines.append("  (no decision_trace records found)")
        _page_text(fig, 0.90, lines)
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.5, 11))
        _page_title(fig, "Inputs Schema Snapshot")
        feature_names = inputs_schema.get("feature_names", [])
        seq_shape = inputs_schema.get("sequence_shape", [])
        lines = [
            f"Sequence shape: {seq_shape}",
            f"Feature count: {len(feature_names)}",
            "Feature names (first 50):",
            ", ".join(feature_names[:50]) if feature_names else "(none)",
        ]
        _page_text(fig, 0.90, lines)
        pdf.savefig(fig)
        plt.close(fig)

        for scope in ["ENTRY", "EXIT", "SIZING"]:
            fig = plt.figure(figsize=(8.5, 11))
            _page_title(fig, f"Top Factors by Lift (Scope: {scope})")
            top_df = _top_factors(attrib_df, scope)
            if top_df.empty:
                _page_text(fig, 0.90, [f"No attribution rows for scope {scope}."])
                pdf.savefig(fig)
                plt.close(fig)
                continue
            lines = ["factor_id | lift_avg_r | win_rate | uses"]
            for _, row in top_df.iterrows():
                lines.append(
                    f"{row.get('factor_id','')} | "
                    f"{row.get('lift_avg_r','')} | "
                    f"{row.get('win_rate','')} | "
                    f"{row.get('uses','')}"
                )
            _page_text(fig, 0.90, lines)
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="inputs_schema.json path")
    ap.add_argument("--trace", required=True, help="decision_trace.jsonl path")
    ap.add_argument("--attrib", required=True, help="decision_factor_attribution.csv path")
    ap.add_argument("--out", required=True, help="output PDF path")
    ap.add_argument("--auto-attrib", action="store_true", help="Generate attribution CSV if missing")
    args = ap.parse_args()

    generate_pdf(
        inputs_schema_path=args.inputs,
        decision_trace_path=args.trace,
        attribution_csv_path=args.attrib,
        output_path=args.out,
        auto_attrib=args.auto_attrib,
    )


if __name__ == "__main__":
    main()
