# Repo Sync Audit

- generated_at_utc: `2026-03-19T17:04:29.786027Z`
- repo_root: `/teamspace/studios/this_studio/spy-iron-condor-trading/scripts`
- current_branch: `main`
- compare_ref: `origin/main`
- head_commit: `7a2e08d76cce983e32aae5af50ef23d7a2a10004`
- ref_commit: `5e067380794bd3dc9ea2c9629dfbdb5e810d16ee`

## Counts

- tracked_head_files: `97`
- tracked_ref_files: `98`
- local_only_tracked_files: `0`
- ref_only_tracked_files: `1`
- untracked_not_ignored: `7`
- unstaged_changes: `248`
- staged_changes: `1`
- local_commits_ahead: `2`
- ref_commits_ahead: `2`

## Output Files

- local_committed_name_status: `/teamspace/studios/this_studio/spy-iron-condor-trading/scripts/reports/repo_sync/20260319_170429Z/local_committed_name_status.txt`
- local_only_tracked_files: `/teamspace/studios/this_studio/spy-iron-condor-trading/scripts/reports/repo_sync/20260319_170429Z/local_only_tracked_files.txt`
- ref_committed_name_status: `/teamspace/studios/this_studio/spy-iron-condor-trading/scripts/reports/repo_sync/20260319_170429Z/ref_committed_name_status.txt`
- ref_only_tracked_files: `/teamspace/studios/this_studio/spy-iron-condor-trading/scripts/reports/repo_sync/20260319_170429Z/ref_only_tracked_files.txt`
- staged_name_status: `/teamspace/studios/this_studio/spy-iron-condor-trading/scripts/reports/repo_sync/20260319_170429Z/staged_name_status.txt`
- summary_json: `/teamspace/studios/this_studio/spy-iron-condor-trading/scripts/reports/repo_sync/20260319_170429Z/summary.json`
- summary_md: `/teamspace/studios/this_studio/spy-iron-condor-trading/scripts/reports/repo_sync/20260319_170429Z/summary.md`
- tracked_head_files: `/teamspace/studios/this_studio/spy-iron-condor-trading/scripts/reports/repo_sync/20260319_170429Z/tracked_head_files.txt`
- tracked_ref_files: `/teamspace/studios/this_studio/spy-iron-condor-trading/scripts/reports/repo_sync/20260319_170429Z/tracked_ref_files.txt`
- unstaged_name_status: `/teamspace/studios/this_studio/spy-iron-condor-trading/scripts/reports/repo_sync/20260319_170429Z/unstaged_name_status.txt`
- untracked_not_ignored: `/teamspace/studios/this_studio/spy-iron-condor-trading/scripts/reports/repo_sync/20260319_170429Z/untracked_not_ignored.txt`

## Suggested Workflow

1. Review `untracked_not_ignored.txt`, `local_only_tracked_files.txt`, and the two `*_name_status.txt` files.
2. If the untracked files should be preserved, stage them with `--stage-untracked` or `git add -- <paths>`.
3. Commit the Lightning-only work.
4. Rebase or merge onto the compare ref.
5. Push the result back to GitHub.

### Example Commands

```bash
python scripts/repo_sync_audit.py --fetch
git add <wanted-files>
git commit -m "save Lightning AI work"
git rebase origin/main
git push origin main
```

