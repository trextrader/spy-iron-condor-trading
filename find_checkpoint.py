"""
find_checkpoint.py — locate v43 checkpoints on this machine and print the run command.
Run: python find_checkpoint.py
"""
import os, glob, sys

print("Searching for .pth files...\n")

found = []
for root, dirs, files in os.walk("."):
    # Skip hidden dirs and very deep vendor dirs
    dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', 'node_modules')]
    for f in files:
        if f.endswith(".pth"):
            p = os.path.join(root, f)
            size_mb = os.path.getsize(p) / 1e6
            found.append((p, size_mb))

if not found:
    print("No .pth files found anywhere under the current directory.")
    print("The checkpoint may be in a different location (e.g. /root, /workspace, ~/).")
    print("Try:  find / -name '*.pth' 2>/dev/null | grep -v miniconda | grep -v site-packages")
    sys.exit(1)

found.sort(key=lambda x: -x[1])   # largest first (models are big)
print(f"Found {len(found)} .pth file(s):\n")
for p, mb in found:
    tag = "  <-- likely model" if mb > 10 else ""
    print(f"  {mb:>8.1f} MB   {p}{tag}")

# Best guess: largest file with 'v43' or 'condor' in name
best = None
for p, mb in found:
    name = os.path.basename(p).lower()
    if any(k in name for k in ('v43', 'condor', 'run')) and mb > 10:
        best = p
        break
if best is None and found:
    best = found[0][0]  # fallback: just largest

if best:
    print(f"\nBest guess checkpoint: {best}")
    print(f"\nRun command:")
    print(f"  python kaggle/condor_brain_backtest_v45.py --use-v43 --v43-model {best} --limit 500 --verbose")
