import argparse
import os
import subprocess
import sys


def _run(cmd: list) -> int:
    print(f"[RUN] {' '.join(cmd)}")
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="Path to .pth checkpoint (required for model-based tests)")
    ap.add_argument("--data", default="", help="Path to CSV dataset (required for data-based tests)")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    scripts_dir = os.path.join(repo_root, "scripts", "model_tests")

    tests = []

    if args.model:
        tests.append([sys.executable, os.path.join(scripts_dir, "01_checkpoint_inventory.py"), "--model", args.model])
        tests.append([sys.executable, os.path.join(scripts_dir, "03_norm_stats_sanity.py"), "--model", args.model])
        tests.append([
            sys.executable,
            os.path.join(scripts_dir, "04_model_forward_contract.py"),
            "--model",
            args.model,
            "--seq-len",
            str(args.seq_len),
            "--batch",
            str(args.batch),
        ])
        tests.append([sys.executable, os.path.join(scripts_dir, "05_output_distribution.py"), "--model", args.model])

    if args.model and args.data:
        tests.append([
            sys.executable,
            os.path.join(scripts_dir, "02_feature_schema_alignment.py"),
            "--model",
            args.model,
            "--data",
            args.data,
        ])

    tests.append([sys.executable, os.path.join(scripts_dir, "06_ruleset_execution_smoke.py")])

    if args.data:
        tests.append([
            sys.executable,
            os.path.join(scripts_dir, "07_feature_pipeline_parity.py"),
            "--data",
            args.data,
        ])

    tests.append([sys.executable, os.path.join(scripts_dir, "12_docs_contract_sync_check.py")])

    if not tests:
        print("[WARN] No tests selected. Provide --model and/or --data.")
        return 2

    failures = 0
    for cmd in tests:
        if _run(cmd) != 0:
            failures += 1

    if failures:
        print(f"[FAIL] {failures} test(s) failed.")
        return 1

    print("[OK] Smoke suite passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
