from __future__ import annotations

import argparse
import json
import os
import sys


def _read_jsonl(path: str):
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


def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", default="artifacts/epochs", help="epoch artifacts root")
    ap.add_argument("--require", action="store_true", help="fail if no manifest found")
    ap.add_argument("--expect-epochs", type=int, default=None, help="expected epoch count")
    args = ap.parse_args()

    manifest_path = os.path.join(args.artifact_dir, "manifest.jsonl")
    records = _read_jsonl(manifest_path)
    if not records:
        if args.require:
            return _fail(f"No manifest found at {manifest_path}")
        print("[OK] No manifest found; skipping checks.")
        return 0

    missing = []
    epoch_ids = []
    for rec in records:
        epoch = rec.get("epoch")
        epoch_ids.append(epoch)
        files = rec.get("files", [])
        if len(files) != 4:
            missing.append(f"epoch {epoch} expected 4 files, got {len(files)}")
        for path in files:
            if not os.path.exists(path):
                missing.append(f"missing file: {path}")

    if missing:
        for msg in missing:
            print(f"[FAIL] {msg}")
        return 1

    if args.expect_epochs is not None:
        unique_epochs = len(set(epoch_ids))
        if unique_epochs != args.expect_epochs:
            return _fail(f"expected {args.expect_epochs} epochs, found {unique_epochs}")

    latest_path = os.path.join(args.artifact_dir, "latest.json")
    if os.path.exists(latest_path):
        with open(latest_path, "r", encoding="utf-8") as f:
            latest = json.load(f)
        latest_epoch = latest.get("epoch")
        latest_ckpt = latest.get("checkpoint_path")
        if latest_epoch is None or latest_ckpt is None:
            return _fail("latest.json missing epoch or checkpoint_path")
        if not os.path.exists(latest_ckpt):
            return _fail(f"latest checkpoint missing: {latest_ckpt}")
        max_epoch = max(e for e in epoch_ids if e is not None)
        if latest_epoch != max_epoch:
            return _fail(f"latest epoch {latest_epoch} does not match max epoch {max_epoch}")

    print("[OK] Epoch artifacts and resume pointers look valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
