import json
import os
import sys


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main() -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    md_path = os.path.join(repo_root, "docs", "PIPELINE_CONTRACT.md")
    json_path = os.path.join(repo_root, "docs", "PIPELINE_CONTRACT.json")

    if not os.path.exists(md_path) or not os.path.exists(json_path):
        print("Missing PIPELINE_CONTRACT.md or PIPELINE_CONTRACT.json")
        return 2

    contract = _load_json(json_path)
    md = _load_text(md_path)

    errors = []

    version = str(contract.get("version", ""))
    if version and f"Version: {version}" not in md:
        errors.append("MD missing version line matching JSON.")

    date = str(contract.get("date", ""))
    if date and f"Date: {date}" not in md:
        errors.append("MD missing date line matching JSON.")

    for item in contract.get("contracts", []):
        cid = str(item.get("id", ""))
        name = str(item.get("name", ""))
        if cid and cid not in md:
            errors.append(f"MD missing contract id {cid}.")
        if name and name not in md:
            errors.append(f"MD missing contract name '{name}'.")

    for gate in contract.get("gates", []):
        if gate not in md:
            errors.append(f"MD missing gate '{gate}'.")

    for artifact in contract.get("required_artifacts", []):
        if artifact not in md:
            errors.append(f"MD missing required artifact '{artifact}'.")

    for test in contract.get("tests", []):
        if test not in md:
            errors.append(f"MD missing test entry '{test}'.")

    if errors:
        print("PIPELINE_CONTRACT docs sync check failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    print("PIPELINE_CONTRACT.md matches PIPELINE_CONTRACT.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
