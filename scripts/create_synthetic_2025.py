import pandas as pd
import numpy as np
import os
from tqdm import tqdm

INPUT_FILE = r"data/processed/Spy_Options_dataset_1.csv"
OUTPUT_FILE = r"data/processed/Spy_Options_dataset_2.csv"
CHUNK_SIZE = 500_000

def create_synthetic_2025():
    print(f"🧬 Creating Synthetic 2025 Dataset...")
    print(f"   Input: {INPUT_FILE}")
    print(f"   Output: {OUTPUT_FILE}")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        return

    # 52 Weeks = 364 Days. 
    # 2024-01-02 (Tue) + 364 days = 2024-12-31 (Tue).
    # This aligns day-of-week perfectly.
    SHIFT_DELTA = pd.Timedelta(weeks=52)
    print(f"   Shift Delta: +{SHIFT_DELTA} (52 weeks)")

    # Read/Write in chunks
    mode = 'w'
    header = True
    processed_rows = 0
    
    # Validation preview
    preview_df = pd.read_csv(INPUT_FILE, nrows=5)
    print(f"   Schema check: {len(preview_df.columns)} columns detected.")
    
    with pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE) as reader:
        for chunk in tqdm(reader, desc="Processing Chunks", unit="chunk"):
            # Detect dt column
            dt_col = 'dt' if 'dt' in chunk.columns else 'datetime'
            
            # Convert to datetime
            chunk[dt_col] = pd.to_datetime(chunk[dt_col])
            
            # 🌟 APPLY TIME SHIFT 🌟
            chunk[dt_col] = chunk[dt_col] + SHIFT_DELTA
            
            # Write to output
            chunk.to_csv(OUTPUT_FILE, mode=mode, header=header, index=False)
            
            processed_rows += len(chunk)
            header = False
            mode = 'a'
            
    print(f"✅ Synthetic Dataset Created!")
    print(f"   Total Rows: {processed_rows:,}")
    print(f"   Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    create_synthetic_2025()
