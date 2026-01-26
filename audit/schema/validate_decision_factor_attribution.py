import csv
import os
import sys


def _read_header(path: str):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        return next(reader, [])


def validate_decision_factor_attribution(csv_path: str) -> bool:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    schema_path = os.path.join(
        repo_root,
        "audit",
        "schemas",
        "decision_factor_attribution_schema_v1.csv.txt",
    )

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema not found: {schema_path}")

    expected = _read_header(schema_path)
    actual = _read_header(csv_path)

    if expected != actual:
        print("CSV header does not match schema.")
        print(f"Expected: {','.join(expected)}")
        print(f"Actual:   {','.join(actual)}")
        return False

    print("Decision factor attribution CSV header matches schema.")
    return True


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python audit/schema/validate_decision_factor_attribution.py <decision_factor_attribution.csv>")
        return 2

    csv_path = sys.argv[1]
    ok = validate_decision_factor_attribution(csv_path)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
