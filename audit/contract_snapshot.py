import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_head(repo_root: str) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return ""


def _pipeline_contract_path(repo_root: str) -> str:
    return os.path.join(repo_root, "docs", "PIPELINE_CONTRACT.json")


def _feature_schema_id(feature_cols: List[str]) -> str:
    return _sha256_text(",".join(feature_cols))


def generate_contract_snapshot(
    output_path: str,
    repo_root: str,
    feature_cols: Optional[List[str]] = None,
    checkpoint_path: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a contract snapshot for audit artifacts.
    This is intentionally lightweight and safe to call from training/backtest.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    contract_path = _pipeline_contract_path(repo_root)
    contract_data = {}
    contract_hash = ""
    contract_version = ""
    if os.path.exists(contract_path):
        with open(contract_path, "r", encoding="utf-8") as f:
            contract_data = json.load(f)
        contract_version = str(contract_data.get("version", ""))
        contract_hash = _sha256_file(contract_path)

    snapshot: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "pipeline_contract_version": contract_version,
        "pipeline_contract_hash": contract_hash,
        "code_commit": _git_head(repo_root),
    }

    if feature_cols:
        snapshot["feature_schema_id"] = _feature_schema_id(feature_cols)
        snapshot["feature_cols_count"] = len(feature_cols)

    if checkpoint_path and os.path.exists(checkpoint_path):
        snapshot["checkpoint_path"] = checkpoint_path
        snapshot["checkpoint_hash"] = _sha256_file(checkpoint_path)

    if extra:
        snapshot["extra"] = dict(extra)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    return snapshot
