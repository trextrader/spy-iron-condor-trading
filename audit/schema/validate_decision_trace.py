import json
import os
import sys
from typing import Any, Dict

try:
    import jsonschema
except ImportError:
    jsonschema = None


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield idx, json.loads(line)


def main() -> int:
    if jsonschema is None:
        print("jsonschema not installed; install with `pip install jsonschema`.")
        return 2

    if len(sys.argv) < 2:
        print("Usage: python audit/schema/validate_decision_trace.py <decision_trace.jsonl>")
        return 2

    trace_path = sys.argv[1]
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    schema_path = os.path.join(repo_root, "audit", "schemas", "decision_trace_schema_v1.json")

    if not os.path.exists(schema_path):
        print(f"Schema not found: {schema_path}")
        return 2

    schema = _load_json(schema_path)
    validator = jsonschema.Draft7Validator(schema)

    errors = 0
    for idx, record in _iter_jsonl(trace_path):
        err = sorted(validator.iter_errors(record), key=lambda e: e.path)
        if err:
            errors += 1
            print(f"Line {idx}: {err[0].message}")
            if errors >= 20:
                print("Too many errors; stopping.")
                break

    if errors:
        print(f"Validation failed with {errors} error(s).")
        return 1

    print("Decision trace validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
