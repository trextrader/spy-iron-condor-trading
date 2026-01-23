import pandas as pd
import os

# Configuration
INPUT_FILE = "data/processed/mamba_institutional_2024_1m_last 100k.csv"
OUTPUT_FILE = "data/processed/mamba_institutional_2024_1m_last_100k_fixed.csv"

def fix_csv_headers():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        return

    print(f"🔍 Reading {INPUT_FILE}...")
    # Read just the header first to be fast/safe
    df_iter = pd.read_csv(INPUT_FILE, chunksize=1)
    df_sample = next(df_iter)
    
    current_cols = list(df_sample.columns)
    print(f"   Current columns: {current_cols[:5]} ...")
    
    # Check if 'dt' is missing but 'timestamp' exists
    if 'dt' not in current_cols and 'timestamp' in current_cols:
        print("   ⚠️ 'dt' column missing, found 'timestamp'. Renaming...")
        
        # Load full file to rename (for 100k rows this is fine)
        df = pd.read_csv(INPUT_FILE)
        df.rename(columns={'timestamp': 'dt'}, inplace=True)
        
        # Verify timestamp format
        # Force conversion to ensure standardized string format if needed, but usually just header rename is enough
        # df['dt'] = pd.to_datetime(df['dt']).dt.strftime('%Y-%m-%d %H:%M:%S%z')
        
        print(f"   💾 Saving fixed CSV to {OUTPUT_FILE}...")
        df.to_csv(OUTPUT_FILE, index=False)
        print("   ✅ Done. Use this file for training.")
        
    elif 'dt' in current_cols:
         print("   ✅ 'dt' column already exists. No changes needed.")
    else:
        print("   ❌ neither 'dt' nor 'timestamp' column found. Inspection required.")
        print(f"      Columns found: {current_cols}")

if __name__ == "__main__":
    fix_csv_headers()
