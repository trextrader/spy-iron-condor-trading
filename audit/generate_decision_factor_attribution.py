from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Tuple


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


def _iter_factors(record: dict) -> Iterable[dict]:
    factors = []
    decision_factors = record.get("decision_factors", {})
    for rule in decision_factors.get("rules", []):
        factors.append({
            "factor_id": rule.get("rule_id", ""),
            "factor_name": rule.get("rule_id", ""),
            "factor_kind": "RULE",
            "factor_origin": "HUMAN_RULE",
            "factor_family": "",
            "threshold_spec": "",
            "polarity": "BIDIR",
            "contribution": 0.0,
            "importance": 0.0,
        })
    for attrib in decision_factors.get("attribution", []):
        factors.append({
            "factor_id": attrib.get("factor_id", ""),
            "factor_name": attrib.get("factor_id", ""),
            "factor_kind": attrib.get("factor_kind", "FEATURE"),
            "factor_origin": "LEARNED",
            "factor_family": "",
            "threshold_spec": "",
            "polarity": "BIDIR",
            "contribution": float(attrib.get("contribution", 0.0) or 0.0),
            "importance": float(attrib.get("importance", 0.0) or 0.0),
        })
    fuzzy = decision_factors.get("fuzzy", {})
    for rule in fuzzy.get("rules_fired", []):
        factors.append({
            "factor_id": rule.get("fuzzy_rule_id", ""),
            "factor_name": rule.get("fuzzy_rule_id", ""),
            "factor_kind": "FUZZY",
            "factor_origin": "HUMAN_RULE",
            "factor_family": "",
            "threshold_spec": "",
            "polarity": "BIDIR",
            "contribution": 0.0,
            "importance": 0.0,
        })
    return factors


def generate_attribution_csv(trace_path: str, out_path: str) -> None:
    records = _read_jsonl(trace_path)
    if not records:
        raise RuntimeError(f"No decision_trace records found at {trace_path}")

    asof_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []

    scope_overall_r = defaultdict(list)
    for rec in records:
        scope = rec.get("decision", {}).get("scope", "UNKNOWN")
        r_mult = rec.get("outcome", {}).get("r_multiple_final", 0.0) or 0.0
        scope_overall_r[scope].append(float(r_mult))
    scope_overall_avg = {
        scope: (sum(vals) / len(vals) if vals else 0.0)
        for scope, vals in scope_overall_r.items()
    }

    agg = defaultdict(lambda: {
        "uses": 0,
        "wins": 0,
        "losses": 0,
        "neutrals": 0,
        "r_vals": [],
        "contrib_vals": [],
        "importance_vals": [],
        "meta": {}
    })

    for rec in records:
        scope = rec.get("decision", {}).get("scope", "UNKNOWN")
        outcome = rec.get("outcome", {})
        win = bool(outcome.get("win"))
        loss = bool(outcome.get("loss"))
        neutral = bool(outcome.get("neutral"))
        r_mult = float(outcome.get("r_multiple_final", 0.0) or 0.0)
        model = rec.get("model", {})
        meta = {
            "model_id": model.get("model_id", ""),
            "model_version": model.get("model_version", ""),
            "model_hash": model.get("model_hash", ""),
            "code_commit": model.get("code_commit", ""),
            "run_id": model.get("run_id", ""),
            "dataset_id": rec.get("governance", {}).get("data_provenance", {}).get("dataset_id", ""),
            "dataset_hash": rec.get("governance", {}).get("data_provenance", {}).get("dataset_hash", ""),
        }

        for f in _iter_factors(rec):
            key = (scope, f["factor_id"])
            entry = agg[key]
            entry["uses"] += 1
            entry["wins"] += 1 if win else 0
            entry["losses"] += 1 if loss else 0
            entry["neutrals"] += 1 if neutral else 0
            entry["r_vals"].append(r_mult)
            entry["contrib_vals"].append(float(f.get("contribution", 0.0)))
            entry["importance_vals"].append(float(f.get("importance", 0.0)))
            entry["meta"] = {**entry["meta"], **f, **meta}

    for (scope, factor_id), entry in agg.items():
        uses = entry["uses"]
        wins = entry["wins"]
        losses = entry["losses"]
        neutrals = entry["neutrals"]
        win_rate = wins / uses if uses else 0.0
        loss_rate = losses / uses if uses else 0.0
        win_loss_ratio = wins / max(1, losses)
        avg_r_active = sum(entry["r_vals"]) / len(entry["r_vals"]) if entry["r_vals"] else 0.0
        avg_r_inactive = scope_overall_avg.get(scope, 0.0)
        lift_avg_r = avg_r_active - avg_r_inactive
        avg_contrib_win = sum(entry["contrib_vals"]) / len(entry["contrib_vals"]) if entry["contrib_vals"] else 0.0
        avg_contrib_loss = avg_contrib_win
        avg_importance = sum(entry["importance_vals"]) / len(entry["importance_vals"]) if entry["importance_vals"] else 0.0

        meta = entry["meta"]
        rows.append({
            "schema_version": "1.0",
            "asof_utc": asof_utc,
            "model_id": meta.get("model_id", ""),
            "model_version": meta.get("model_version", ""),
            "model_hash": meta.get("model_hash", ""),
            "code_commit": meta.get("code_commit", ""),
            "run_id": meta.get("run_id", ""),
            "dataset_id": meta.get("dataset_id", ""),
            "dataset_hash": meta.get("dataset_hash", ""),
            "segment_id": "",
            "scope": scope,
            "factor_id": factor_id,
            "factor_name": meta.get("factor_name", factor_id),
            "factor_kind": meta.get("factor_kind", ""),
            "factor_origin": meta.get("factor_origin", ""),
            "factor_family": meta.get("factor_family", ""),
            "threshold_spec": meta.get("threshold_spec", ""),
            "polarity": meta.get("polarity", "BIDIR"),
            "uses": uses,
            "wins": wins,
            "losses": losses,
            "neutrals": neutrals,
            "win_rate": round(win_rate, 6),
            "loss_rate": round(loss_rate, 6),
            "win_loss_ratio": round(win_loss_ratio, 6),
            "avg_r_when_active": round(avg_r_active, 6),
            "avg_r_when_inactive": round(avg_r_inactive, 6),
            "lift_avg_r": round(lift_avg_r, 6),
            "avg_mfe_r": 0.0,
            "avg_mae_r": 0.0,
            "avg_contribution_win": round(avg_contrib_win, 6),
            "avg_contribution_loss": round(avg_contrib_loss, 6),
            "avg_importance": round(avg_importance, 6),
            "confidence": 0.0,
            "notes": ""
        })

    fieldnames = list(rows[0].keys()) if rows else []
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="decision_trace.jsonl path")
    ap.add_argument("--out", required=True, help="output decision_factor_attribution.csv path")
    args = ap.parse_args()
    generate_attribution_csv(args.trace, args.out)


if __name__ == "__main__":
    main()
