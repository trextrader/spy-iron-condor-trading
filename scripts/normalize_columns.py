import sys
import pandas as pd
import re

if len(sys.argv) < 3:
    print("Usage: python normalize_columns.py <input> <output>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

df = pd.read_csv(input_file)

new_cols = {}

for col in df.columns:
    original = col
    c = col.strip().lower()

    # Strip prefixes like m1_, m5_, m15_, h1_, etc.
    c = re.sub(r'^(m\d+_|h\d+_)', '', c)

    # Normalize OHLCV
    if c in ["open", "o"]:
        new_cols[original] = "open"
    elif c in ["high", "h"]:
        new_cols[original] = "high"
    elif c in ["low", "l"]:
        new_cols[original] = "low"
    elif c in ["close", "c"]:
        new_cols[original] = "close"
    elif c in ["volume", "vol", "v"]:
        new_cols[original] = "volume"
    else:
        new_cols[original] = original  # leave indicators untouched

df = df.rename(columns=new_cols)
df.to_csv(output_file, index=False)

print(f"[OK] Normalized columns → {output_file}")
