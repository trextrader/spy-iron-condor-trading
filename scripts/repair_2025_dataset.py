import pandas as pd
import numpy as np
import os
import csv
import sys
from tqdm import tqdm

# Increase CSV field size limit safely
try:
    csv.field_size_limit(10**7) # 10MB limit per field should be enough
except Exception:
    pass

INPUT_FILE = r"data/processed/mamba_institutional_1m_v21.csv"
OUTPUT_FILE = r"data/processed/Spy_Options_dataset_2.csv"
SPLIT_DATE = "2025-01-02"

def repair_and_extract_2025():
    print(f"🔧 Starting repair and extraction for 2025 data...")
    print(f"   Input: {INPUT_FILE}")
    print(f"   Output: {OUTPUT_FILE}")
    print(f"   Split Date: >= {SPLIT_DATE}")

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        return

    # Open output file
    mode = 'w'
    write_header = True
    
    # We will use manual line processing to handle NUL bytes
    processed_rows = 0
    saved_rows = 0
    corrupted_lines = 0

    with open(INPUT_FILE, 'rb') as f_in, open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f_out:
        writer = None
        
        # Read header first
        header_line = f_in.readline().decode('utf-8').strip()
        header_cols = header_line.split(',')
        print(f"   Header columns: {len(header_cols)}")
        
        # Find dt column index
        try:
            dt_idx = header_cols.index('dt')
        except ValueError:
            try:
                dt_idx = header_cols.index('datetime')
            except ValueError:
                print("❌ Could not find 'dt' or 'datetime' column in header")
                return
                
        # Initialize CSV writer
        writer = csv.writer(f_out)
        writer.writerow(header_cols)
        
        chunk_size = 100000
        buffer = []
        
        # Generator for reading clean lines
        def clean_line_generator(f):
            for binary_line in f:
                try:
                    # Replace NUL bytes
                    if b'\x00' in binary_line:
                        binary_line = binary_line.replace(b'\x00', b'')
                        yield binary_line.decode('utf-8', errors='replace'), True
                    else:
                        yield binary_line.decode('utf-8', errors='replace'), False
                except Exception as e:
                    yield None, True # Treat as hard corruption

        pbar = tqdm(desc="Scanning lines", unit="lines")
        
        for line, was_corrupted in clean_line_generator(f_in):
            processed_rows += 1
            if line is None:
                corrupted_lines += 1
                continue
                
            if was_corrupted:
                corrupted_lines += 1
                
            # Parse line
            cols = line.strip().split(',')
            
            # Basic validation
            if len(cols) != len(header_cols):
                # Try to salvage if it's just a trailing comma issue or quotes
                # For now, skip bad length
                continue
                
            # Check date
            try:
                row_dt = cols[dt_idx]
                # String comparison is fast and sufficient for ISO format YYYY-MM-DD
                if row_dt >= SPLIT_DATE:
                    writer.writerow(cols)
                    saved_rows += 1
            except IndexError:
                continue
                
            if processed_rows % 10000 == 0:
                pbar.update(10000)
                
        pbar.close()

    print(f"\n✅ Done!")
    print(f"   Processed: {processed_rows:,}")
    print(f"   Saved (>= {SPLIT_DATE}): {saved_rows:,}")
    print(f"   Corrupted/Fixed Lines: {corrupted_lines:,}")
    print(f"   Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    repair_and_extract_2025()
