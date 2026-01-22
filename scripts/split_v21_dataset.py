
import pandas as pd
import os
from tqdm import tqdm

def split_dataset():
    input_path = r"c:\SPYOptionTrader_test\data\processed\mamba_institutional_1m_v21.csv"
    output_dir = r"c:\SPYOptionTrader_test\data\processed"
    
    out1_path = os.path.join(output_dir, "Spy_Options_dataset_1.csv")
    out2_path = os.path.join(output_dir, "Spy_Options_dataset_2.csv")
    
    print(f"Splitting {input_path}...")
    print(f"  -> {out1_path} ( < 2025-01-02)")
    print(f"  -> {out2_path} ( >= 2025-01-02)")
    
    # Split date
    split_date = "2025-01-02"
    
    # Read/Write in chunks to handle 7GB file
    chunksize = 500000
    
    # Initialize files (write header first)
    first_chunk = True
    
    # Counters
    count1 = 0
    count2 = 0
    
    with pd.read_csv(input_path, chunksize=chunksize, dtype={'dt': str}, on_bad_lines='skip', engine='python') as reader:
        for chunk in tqdm(reader, desc="Processing chunks"):
            # Ensure proper datetime comparison
            # We can do string comparison if format is ISO, which it seems to be (YYYY-MM-DD...)
            # But pd.to_datetime is safer
            
            # Create masks
            # Note: We rely on string comparison for speed if format is ISO-8601 compatible
            # "2024-..." < "2025-01-02" is true.
            # "2025-01-02..." >= "2025-01-02" is true.
            mask1 = chunk['dt'] < split_date
            mask2 = ~mask1 # >= split_date
            
            df1 = chunk[mask1]
            df2 = chunk[mask2]
            
            mode = 'w' if first_chunk else 'a'
            header = first_chunk
            
            if not df1.empty:
                df1.to_csv(out1_path, mode=mode, header=header, index=False)
                count1 += len(df1)
                
            if not df2.empty:
                df2.to_csv(out2_path, mode=mode, header=header, index=False)
                count2 += len(df2)
                
            first_chunk = False

    print("\nSplit Complete.")
    print(f"Dataset 1 Rows: {count1:,}")
    print(f"Dataset 2 Rows: {count2:,}")

if __name__ == "__main__":
    split_dataset()
