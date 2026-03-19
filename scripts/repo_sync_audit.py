#!/usr/bin/env python3
"""
Audit the local repo against a reference such as origin/main.

This is intended for environments like Lightning AI where you want a complete
inventory of:
  - tracked files in local HEAD
  - tracked files in a remote ref (default: origin/main)
  - files only present locally
  - files only present in the ref
  - local untracked files that are NOT ignored by .gitignore
  - working tree and staged changes

It writes text inventories plus JSON/Markdown summaries under reports/repo_sync/.
Optionally it can stage the untracked, non-ignored files it discovered.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _git(repo_root: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed with exit code {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc.stdout


def _lines(text: str) -> list[str]:
    return [line.rstrip("\n") for line in text.splitlines() if line.strip()]


def _parse_name_status(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in _lines(text):
        parts = raw.split("\t")
        status = parts[0]
        row: dict[str, Any] = {"status": status, "raw": raw}
        if len(parts) >= 2:
            row["path"] = parts[-1]
        if status[:1] in {"R", "C"} and len(parts) >= 3:
            row["old_path"] = parts[1]
            row["path"] = parts[2]
        rows.append(row)
    return rows


def _write_lines(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(rows)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _stage_paths(repo_root: Path, paths: list[str]) -> None:
    if not paths:
        return
    for chunk in _chunked(paths, 200):
        subprocess.run(
            ["git", "add", "--", *chunk],
            cwd=repo_root,
            check=True,
        )


def _build_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    lines = [
        "# Repo Sync Audit",
        "",
        f"- generated_at_utc: `{report['generated_at_utc']}`",
        f"- repo_root: `{report['repo_root']}`",
        f"- current_branch: `{report['current_branch']}`",
        f"- compare_ref: `{report['compare_ref']}`",
        f"- head_commit: `{report['head_commit']}`",
        f"- ref_commit: `{report['ref_commit']}`",
        "",
        "## Counts",
        "",
        f"- tracked_head_files: `{counts['tracked_head_files']}`",
        f"- tracked_ref_files: `{counts['tracked_ref_files']}`",
        f"- local_only_tracked_files: `{counts['local_only_tracked_files']}`",
        f"- ref_only_tracked_files: `{counts['ref_only_tracked_files']}`",
        f"- untracked_not_ignored: `{counts['untracked_not_ignored']}`",
        f"- unstaged_changes: `{counts['unstaged_changes']}`",
        f"- staged_changes: `{counts['staged_changes']}`",
        f"- local_commits_ahead: `{counts['local_commits_ahead']}`",
        f"- ref_commits_ahead: `{counts['ref_commits_ahead']}`",
        "",
        "## Output Files",
        "",
    ]
    for key, value in sorted(report["artifacts"].items()):
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Suggested Workflow",
            "",
            "1. Review `untracked_not_ignored.txt`, `local_only_tracked_files.txt`, and the two `*_name_status.txt` files.",
            "2. If the untracked files should be preserved, stage them with `--stage-untracked` or `git add -- <paths>`.",
            "3. Commit the Lightning-only work.",
            "4. Rebase or merge onto the compare ref.",
            "5. Push the result back to GitHub.",
            "",
            "### Example Commands",
            "",
            "```bash",
            "python scripts/repo_sync_audit.py --fetch",
            "git add <wanted-files>",
            "git commit -m \"save Lightning AI work\"",
            "git rebase origin/main",
            "git push origin main",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit local repo vs a compare ref.")
    parser.add_argument(
        "--ref",
        default="origin/main",
        help="Git ref to compare against. Default: origin/main",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Run `git fetch origin` before auditing.",
    )
    parser.add_argument(
        "--stage-untracked",
        action="store_true",
        help="Stage untracked, non-ignored files discovered by the audit.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Default: reports/repo_sync/<timestamp>",
    )
    args = parser.parse_args()

    repo_root = Path.cwd()
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "reports" / "repo_sync" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.fetch:
        _git(repo_root, "fetch", "origin")

    tracked_head_files = sorted(_lines(_git(repo_root, "ls-tree", "-r", "--name-only", "HEAD")))
    tracked_ref_files = sorted(_lines(_git(repo_root, "ls-tree", "-r", "--name-only", args.ref)))
    untracked_not_ignored = sorted(_lines(_git(repo_root, "ls-files", "--others", "--exclude-standard")))
    unstaged_changes = _parse_name_status(_git(repo_root, "diff", "--name-status"))
    staged_changes = _parse_name_status(_git(repo_root, "diff", "--cached", "--name-status"))
    local_committed_changes = _parse_name_status(
        _git(repo_root, "diff", "--name-status", "--find-renames", f"{args.ref}..HEAD")
    )
    ref_committed_changes = _parse_name_status(
        _git(repo_root, "diff", "--name-status", "--find-renames", f"HEAD..{args.ref}")
    )
    ahead_behind = _git(repo_root, "rev-list", "--left-right", "--count", f"HEAD...{args.ref}").strip()
    local_commits_ahead, ref_commits_ahead = [int(part) for part in ahead_behind.split()]

    local_only_tracked_files = sorted(set(tracked_head_files) - set(tracked_ref_files))
    ref_only_tracked_files = sorted(set(tracked_ref_files) - set(tracked_head_files))

    artifacts = {
        "tracked_head_files": str(output_dir / "tracked_head_files.txt"),
        "tracked_ref_files": str(output_dir / "tracked_ref_files.txt"),
        "local_only_tracked_files": str(output_dir / "local_only_tracked_files.txt"),
        "ref_only_tracked_files": str(output_dir / "ref_only_tracked_files.txt"),
        "untracked_not_ignored": str(output_dir / "untracked_not_ignored.txt"),
        "unstaged_name_status": str(output_dir / "unstaged_name_status.txt"),
        "staged_name_status": str(output_dir / "staged_name_status.txt"),
        "local_committed_name_status": str(output_dir / "local_committed_name_status.txt"),
        "ref_committed_name_status": str(output_dir / "ref_committed_name_status.txt"),
        "summary_json": str(output_dir / "summary.json"),
        "summary_md": str(output_dir / "summary.md"),
    }

    _write_lines(Path(artifacts["tracked_head_files"]), tracked_head_files)
    _write_lines(Path(artifacts["tracked_ref_files"]), tracked_ref_files)
    _write_lines(Path(artifacts["local_only_tracked_files"]), local_only_tracked_files)
    _write_lines(Path(artifacts["ref_only_tracked_files"]), ref_only_tracked_files)
    _write_lines(Path(artifacts["untracked_not_ignored"]), untracked_not_ignored)
    _write_lines(Path(artifacts["unstaged_name_status"]), [row["raw"] for row in unstaged_changes])
    _write_lines(Path(artifacts["staged_name_status"]), [row["raw"] for row in staged_changes])
    _write_lines(Path(artifacts["local_committed_name_status"]), [row["raw"] for row in local_committed_changes])
    _write_lines(Path(artifacts["ref_committed_name_status"]), [row["raw"] for row in ref_committed_changes])

    if args.stage_untracked and untracked_not_ignored:
        _stage_paths(repo_root, untracked_not_ignored)

    report = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "repo_root": str(repo_root),
        "compare_ref": args.ref,
        "current_branch": _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD").strip(),
        "head_commit": _git(repo_root, "rev-parse", "HEAD").strip(),
        "ref_commit": _git(repo_root, "rev-parse", args.ref).strip(),
        "counts": {
            "tracked_head_files": len(tracked_head_files),
            "tracked_ref_files": len(tracked_ref_files),
            "local_only_tracked_files": len(local_only_tracked_files),
            "ref_only_tracked_files": len(ref_only_tracked_files),
            "untracked_not_ignored": len(untracked_not_ignored),
            "unstaged_changes": len(unstaged_changes),
            "staged_changes": len(staged_changes),
            "local_commits_ahead": local_commits_ahead,
            "ref_commits_ahead": ref_commits_ahead,
        },
        "stage_untracked_requested": bool(args.stage_untracked),
        "artifacts": artifacts,
        "local_only_tracked_files": local_only_tracked_files,
        "ref_only_tracked_files": ref_only_tracked_files,
        "untracked_not_ignored": untracked_not_ignored,
        "unstaged_changes": unstaged_changes,
        "staged_changes": staged_changes,
        "local_committed_changes": local_committed_changes,
        "ref_committed_changes": ref_committed_changes,
    }

    Path(artifacts["summary_json"]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    Path(artifacts["summary_md"]).write_text(_build_markdown(report) + "\n", encoding="utf-8")

    print(f"[repo_sync_audit] compare_ref={args.ref}")
    print(f"[repo_sync_audit] output_dir={output_dir}")
    print(f"[repo_sync_audit] tracked_head_files={len(tracked_head_files)}")
    print(f"[repo_sync_audit] tracked_ref_files={len(tracked_ref_files)}")
    print(f"[repo_sync_audit] local_only_tracked_files={len(local_only_tracked_files)}")
    print(f"[repo_sync_audit] ref_only_tracked_files={len(ref_only_tracked_files)}")
    print(f"[repo_sync_audit] untracked_not_ignored={len(untracked_not_ignored)}")
    print(f"[repo_sync_audit] unstaged_changes={len(unstaged_changes)}")
    print(f"[repo_sync_audit] staged_changes={len(staged_changes)}")
    print(f"[repo_sync_audit] local_commits_ahead={local_commits_ahead}")
    print(f"[repo_sync_audit] ref_commits_ahead={ref_commits_ahead}")
    if args.stage_untracked:
        print(f"[repo_sync_audit] staged_untracked={len(untracked_not_ignored)}")
    print(f"[repo_sync_audit] summary_json={artifacts['summary_json']}")
    print(f"[repo_sync_audit] summary_md={artifacts['summary_md']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
