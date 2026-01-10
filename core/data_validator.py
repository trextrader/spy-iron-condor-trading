import os
import datetime
import pandas as pd
from typing import Tuple, Optional

def get_csv_date_range(csv_path: str, date_col_index: int = None) -> Tuple[Optional[datetime.date], Optional[datetime.date]]:
    """
    Efficiently get the first and last date from a sorted CSV file.
    Auto-detects date column by name if date_col_index is None.
    """
    if not os.path.exists(csv_path):
        return None, None
        
    start_date = None
    end_date = None
    
    # Common date column names to search for
    DATE_COLUMN_NAMES = ['date', 'trade_date', 'timestamp', 'datetime', 'time', 'quote_date']
    
    try:
        # Read header and first row
        with open(csv_path, 'r', encoding='utf-8') as f:
            header_line = f.readline().strip()
            header_cols = [col.strip().lower().replace('"', '').replace("'", "") for col in header_line.split(',')]
            
            # Auto-detect date column if not specified
            if date_col_index is None:
                for name in DATE_COLUMN_NAMES:
                    if name in header_cols:
                        date_col_index = header_cols.index(name)
                        break
                if date_col_index is None:
                    date_col_index = 0  # Fallback to first column
            
            first_line = f.readline()
            if first_line:
                parts = first_line.split(',')
                if len(parts) > date_col_index:
                    try:
                        # Attempt to parse ISO format or similar
                        dt_str = parts[date_col_index].replace('"', '').replace("'", "").strip()
                        start_date = pd.to_datetime(dt_str).date()
                    except Exception:
                        pass
            
            # Read last line efficiently
            f.seek(0, os.SEEK_END)
            filesize = f.tell()
            buffer_size = 1024
            last_line = ""
            
            # Handle empty file
            if filesize == 0:
                return None, None
                
            # Seek backwards
            offset = min(buffer_size, filesize)
            f.seek(filesize - offset)
            lines = f.readlines()
            
            # Get the very last non-empty line
            if lines:
                last_line = lines[-1].strip()
                # If the last chunk didn't capture a full line at the start, lines[0] might be partial
                # But lines[-1] should be the actual last line if file ends with newline
                if not last_line and len(lines) > 1:
                     last_line = lines[-2].strip()
            
            if last_line:
                 parts = last_line.split(',')
                 if len(parts) > date_col_index:
                    try:
                        dt_str = parts[date_col_index].replace('"', '').replace("'", "").strip()
                        end_date = pd.to_datetime(dt_str).date()
                    except Exception:
                        pass

    except Exception as e:
        print(f"[Validator Error] Could not read date range: {e}")
        return None, None
        
    return start_date, end_date

def validate_data_coverage(csv_path: str, 
                         req_start: datetime.date, 
                         req_end: datetime.date, 
                         symbol: str = "SPY") -> bool:
    """
    Check if the CSV data covers the requested backtest range.
    Returns True if coverage is sufficient (or if validly ignored), False if insufficient.
    """
    
    # 1. Check existence
    if not os.path.exists(csv_path):
        print(f"[Data Check] ERROR: Missing data file: {csv_path}")
        return False
        
    # 2. Check Range
    data_start, data_end = get_csv_date_range(csv_path)
    
    if not data_start or not data_end:
        print(f"[Data Check] WARNING: Could not determine date range for {csv_path}.")
        # We allow it to proceed potentially, or return False. Let's return False to be safe.
        return False
        
    print(f"[Data Check] File: {os.path.basename(csv_path)} | Range: {data_start} to {data_end}")
    
    # 3. Validate
    # Allow small buffer? No, rigorous check.
    
    failed = False
    if req_start < data_start:
        print(f"  [!] Missing Data (Start): Requested {req_start}, Data starts {data_start}")
        failed = True
        
    if req_end > data_end:
        print(f"  [!] Missing Data (End): Requested {req_end}, Data ends {data_end}")
        failed = True
        
    if failed:
        return False
        
    print(f"  [OK] Data coverage verified.")
    return True
