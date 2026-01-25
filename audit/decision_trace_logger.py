from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _sha256_file(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class TraceConfig:
    output_path: str
    model_id: str
    model_version: str
    model_hash: str
    code_commit: str
    run_id: str
    dataset_id: str
    dataset_path: Optional[str] = None


class DecisionTraceLogger:
    def __init__(self, cfg: TraceConfig) -> None:
        self.cfg = cfg
        os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
        self.dataset_hash = _sha256_file(cfg.dataset_path)

    def append(self, record: Dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("schema_version", "1.0")
        record.setdefault("ts_utc", _utc_now())

        model_block = record.get("model", {})
        model_block.setdefault("model_id", self.cfg.model_id)
        model_block.setdefault("model_version", self.cfg.model_version)
        model_block.setdefault("model_hash", self.cfg.model_hash)
        model_block.setdefault("code_commit", self.cfg.code_commit)
        model_block.setdefault("run_id", self.cfg.run_id)
        record["model"] = model_block

        gov = record.get("governance", {})
        prov = gov.get("data_provenance", {})
        prov.setdefault("dataset_id", self.cfg.dataset_id)
        prov.setdefault("dataset_hash", self.dataset_hash)
        gov["data_provenance"] = prov
        record["governance"] = gov

        with open(self.cfg.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
