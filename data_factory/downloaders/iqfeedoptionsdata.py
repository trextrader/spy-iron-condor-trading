import socket
import pandas as pd
from datetime import datetime, timedelta

# ============================================================
# IQFeed TCP helper with timeout + ENDMSG detection
# ============================================================
def iqfeed_request(message, host="127.0.0.1", port=9100, timeout=5):
    """Send a request to IQFeed and return the raw response with timeout."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect((host, port))
    s.sendall((message + "\r\n").encode("utf-8"))

    data = b""
    try:
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"!ENDMSG!" in data:
                break
    except socket.timeout:
        print("⚠️  Timeout waiting for IQFeed response")
    finally:
        s.close()

    return data.decode("utf-8")


# ============================================================
# Get all SPY option symbols for a given date
# ============================================================
def get_spy_chain(date):
    """Returns all SPY option symbols (OCC codes) for a given date."""
    datestr = date.strftime("%Y%m%d")
    raw = iqfeed_request(f"SCO,SPY,{datestr}")

    rows = [
        line.split(",")
        for line in raw.splitlines()
        if line and "ENDMSG" not in line
    ]

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


# ============================================================
# Get historical option data (Greeks, IV, bid/ask, etc.)
# ============================================================
def get_option_history(symbol, start_date, end_date):
    """Pulls historical option data for a single OCC symbol."""
    sdate = start_date.strftime("%Y%m%d")
    edate = end_date.strftime("%Y%m%d")

    raw = iqfeed_request(f"HGO,{symbol},{sdate},{edate}")

    rows = [
        line.split(",")
        for line in raw.splitlines()
        if line and "ENDMSG" not in line
    ]

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


# ============================================================
# Main: Pull 6 months of SPY options data
# ============================================================
end = datetime.now()
start = end - timedelta(days=180)

today = datetime.now().date()

# *** CRITICAL FIX ***
dates = [
    d for d in pd.date_range(start, end, freq="D")
    if d.date() <= today
]

# Debug print to confirm correct dates
print("Dates being requested:")
print(dates[0].date(), "...", dates[-1].date())

print("\nFetching SPY option chains for the last 6 months...")

all_data = []

for day in dates:
    print(f"\nGetting chain for {day.date()}...")
    chain = get_spy_chain(day)

    if chain.empty:
        print("  ⚠️  No chain returned — skipping")
        continue

    # Column 0 = OCC symbol
    symbols = chain[0].unique()

    # Filter out invalid symbols (e.g., "E", "", "ENDMSG")
    symbols = [s for s in symbols if isinstance(s, str) and len(s) > 5]

    for sym in symbols:
        print(f"  Pulling history for {sym}...")
        hist = get_option_history(sym, day, day)

        if hist.empty:
            continue

        hist["symbol"] = sym
        hist["date"] = day
        all_data.append(hist)


# ============================================================
# Combine and save
# ============================================================
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    print("\nFinal dataset shape:", final_df.shape)
else:
    final_df = pd.DataFrame()
    print("\nNo data returned.")

final_df.to_csv("spy_options_6mo.csv", index=False)
print("Saved spy_options_6mo.csv")
