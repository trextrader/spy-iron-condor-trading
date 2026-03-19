import pandas as pd
import sys
import re

if len(sys.argv) < 3:
    print("Usage: python strip_timeframe_prefixes.py <input.csv> <output.csv>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

df = pd.read_csv(input_file)

def clean_col(col):
    original = col

    # Remove prefixes like m5_, m15_, h1_
    col = re.sub(r'^(m5_|m15_|h1_)', '', col, flags=re.IGNORECASE)

    # Remove suffixes like _m5, _m15, _h1
    col = re.sub(r'(_m5|_m15|_h1)$', '', col, flags=re.IGNORECASE)

    # Remove standalone m5, m15, h1 inside names (but NOT timestamp)
    col = re.sub(r'\bm5\b', '', col, flags=re.IGNORECASE)
    col = re.sub(r'\bm15\b', '', col, flags=re.IGNORECASE)
    col = re.sub(r'\bh1\b', '', col, flags=re.IGNORECASE)

    # Clean double underscores created by removals
    col = col.replace("__", "_").strip("_")

    # Avoid empty column names
    if col == "":
        col = original

    return col

df.columns = [clean_col(c) for c in df.columns]

df.to_csv(output_file, index=False)
print(f"[OK] Cleaned timeframe prefixes → {output_file}")
