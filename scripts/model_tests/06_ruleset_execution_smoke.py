import argparse
import sys
from pathlib import Path

import pandas as pd

from intelligence.rule_engine.dsl_parser import RuleDSLParser
from intelligence.rule_engine.executor import RuleExecutionEngine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to CSV dataset")
    ap.add_argument("--ruleset", default="docs/Complete_Ruleset_DSL.yaml")
    ap.add_argument("--nrows", type=int, default=5000)
    args = ap.parse_args()

    data_path = Path(args.data)
    ruleset_path = Path(args.ruleset)
    if not data_path.exists():
        print(f"[FAIL] data not found: {data_path}")
        sys.exit(1)
    if not ruleset_path.exists():
        print(f"[FAIL] ruleset not found: {ruleset_path}")
        sys.exit(1)

    df = pd.read_csv(data_path, nrows=args.nrows)
    parser = RuleDSLParser(str(ruleset_path))
    ruleset = parser.load()
    engine = RuleExecutionEngine(ruleset)

    results = engine.execute(df)
    if not results:
        print("[FAIL] rule engine returned empty results")
        sys.exit(1)

    # Smoke: check one rule output keys
    sample_key = next(iter(results.keys()))
    sample_df = results[sample_key]
    required = {"signal_long", "signal_short", "signal_exit", "blocked", "size"}
    if not required.issubset(set(sample_df.columns)):
        print(f"[FAIL] missing expected columns in rule output: {required}")
        sys.exit(1)

    print(f"[OK] rules executed for {len(results)} rules")


if __name__ == "__main__":
    main()
