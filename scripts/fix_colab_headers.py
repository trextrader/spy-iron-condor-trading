import pandas as pd
import os

# Configuration
FILES_TO_CHECK = [
    "data/processed/mamba_institutional_2024_1m_last 100k.csv",
    "data/processed/mamba_institutional_2024_1m_last 500k.csv"
]

def fix_csv_headers():
def fix_csv_headers():
    for input_file in FILES_TO_CHECK:
        print(f"--- Processing {input_file} ---")
        if not os.path.exists(input_file):
            print(f"❌ Input file not found: {input_file}")
            continue

        print(f"🔍 Reading {input_file}...")
        try:
            # Read just the header first to be fast/safe
            df_iter = pd.read_csv(input_file, chunksize=1)
            df_sample = next(df_iter)
            
            current_cols = list(df_sample.columns)
            print(f"   Current columns: {current_cols[:5]} ...")
            
            # Check if 'dt' is missing but 'timestamp' exists
            if 'dt' not in current_cols and 'timestamp' in current_cols:
                print("   ⚠️ 'dt' column missing, found 'timestamp'. Renaming...")
                
                # Load full file to rename
                df = pd.read_csv(input_file)
                df.rename(columns={'timestamp': 'dt'}, inplace=True)
                
                # Overwrite in place or create fixed version? 
                # User asked to "setup... to run of this file", implies standardizing the file itself is best.
                # But to be safe, let's just overwrite since we are fixing headers.
                
                print(f"   💾 Saving fixed CSV headers to {input_file}...")
                df.to_csv(input_file, index=False)
                print("   ✅ Done.")
                
            elif 'dt' in current_cols:
                 print("   ✅ 'dt' column already exists. No changes needed.")
            else:
                print("   ❌ neither 'dt' nor 'timestamp' column found. Inspection required.")
                print(f"      Columns found: {current_cols}")
        except Exception as e:
            print(f"   ❌ Error processing file: {e}")

if __name__ == "__main__":
    fix_csv_headers()
