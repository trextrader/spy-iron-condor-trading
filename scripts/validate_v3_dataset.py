import pandas as pd
import os

def validate(file_path):
    print(f"🔍 Validating: {file_path}")
    if not os.path.exists(file_path):
        print("❌ File NOT FOUND")
        return

    try:
        df = pd.read_csv(file_path, nrows=10)
        print(f"✅ Loaded Header + 10 rows")
        print(f"   Shape: {df.shape} (partial)")
        print(f"   Columns ({len(df.columns)}):")
        print(df.columns.tolist())
        
        # Check Critical Cols
        critical = ['symbol', 'timestamp', 'strike', 'bid', 'ask', 'underlying_price', 'open', 'high', 'low', 'close']
        missing = [c for c in critical if c not in df.columns]
        if missing:
            print(f"❌ MISSING CRITICAL COLS: {missing}")
        else:
            print(f"✅ Critical Base Columns Present")

        # Check Features (if enriched)
        features = ['rsi_14', 'atr_14', 'mom_10', 'target_spot', 'max_dd_60m']
        found_features = [c for c in features if c in df.columns]
        if found_features:
            print(f"✅ Found Features: {found_features}")
        else:
            print(f"⚠️ No Dynamic Features Found (Expected if raw dataset)")

    except Exception as e:
        print(f"❌ Error reading file: {e}")

if __name__ == "__main__":
    base = r"data/processed/v3"
    files = [
        "Spy_Options_2024_1m.csv",
        "Spy_Options_2024_1m_enriched.csv",
        "Spy_Options_2025_1m.csv",
        "Spy_Options_2025_1m_enriched.csv"
    ]
    
    for f in files:
        validate(os.path.join(base, f))
        print("-" * 50)
