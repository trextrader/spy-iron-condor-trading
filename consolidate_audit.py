
import os

files = [
    "intelligence/condor_brain_net.py",
    "intelligence/condor_brain.py",
    "intelligence/generative/diffusion.py",
    "intelligence/rule_engine/dsl_parser.py",
    "intelligence/rule_engine/executor.py",
    "intelligence/fuzzy_engine.py",
    "kaggle/condor_brain_backtest_v2.py"
]

output_patch = "intelligence/architecture_audit_consolidated.txt"

with open(output_patch, "w", encoding="utf-8") as outfile:
    for f in files:
        if os.path.exists(f):
            outfile.write(f"\n\n{'='*80}\n")
            outfile.write(f"FILE: {f}\n")
            outfile.write(f"{'='*80}\n\n")
            with open(f, "r", encoding="utf-8") as infile:
                outfile.write(infile.read())
            outfile.write("\n")
        else:
            outfile.write(f"\n\n!!! FILE NOT FOUND: {f} !!!\n\n")

print(f"Consolidated {len(files)} files into {output_patch}")
