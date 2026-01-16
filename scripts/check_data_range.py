import pandas as pd
import os

files = [
    r"C:\SPYOptionTrader_test\data\alpaca_options\spy_options_intraday_tradeable_m15.csv",
    r"C:\SPYOptionTrader_test\data\ivolatility\spy_options_ivol_large.csv",
    r"C:\SPYOptionTrader_test\data\ivolatility\spy_options_ivol_large_clean.csv",
    r"C:\SPYOptionTrader_test\data\synthetic_options\spy_options_marks.csv",
    r"C:\SPYOptionTrader_test\data\spot\SPY_15.csv"
]

for f in files:
    if os.path.exists(f):
        try:
            print(f"--- Checking {os.path.basename(f)} ---")
            # Read first row
            header = pd.read_csv(f, nrows=1)
            time_col = None
            for c in header.columns:
                if 'time' in c.lower() or 'date' in c.lower() or 'dt' == c.lower():
                    time_col = c
                    break
            
            if time_col:
                first = pd.read_csv(f, nrows=1, usecols=[time_col])[time_col].iloc[0]
                # To get the last row efficiently, we can't use read_csv on 1GB file.
                # But we can try to read just the tail?
                # For now let's just use read_csv with skiprows if we have to, 
                # or just use the end of the file.
                print(f"Start Date: {first}")
            else:
                print(f"No time column found in {header.columns}")
        except Exception as e:
            print(f"Error checking {f}: {e}")
    else:
        print(f"File not found: {f}")
