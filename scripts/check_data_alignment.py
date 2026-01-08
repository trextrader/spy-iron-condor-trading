import pandas as pd

# Check spot data
spot_df = pd.read_csv('reports/SPY/SPY_5.csv')
spot_df['timestamp'] = pd.to_datetime(spot_df['timestamp'], utc=True)
print("=== Spot Data (SPY_5.csv) ===")
print(f"Total rows: {len(spot_df)}")
print(f"Date range: {spot_df.timestamp.min()} to {spot_df.timestamp.max()}")
print(f"2026-01-02 bars: {len(spot_df[spot_df.timestamp.dt.date == pd.Timestamp('2026-01-02').date()])}")
if len(spot_df[spot_df.timestamp.dt.date == pd.Timestamp('2026-01-02').date()]) > 0:
    day_data = spot_df[spot_df.timestamp.dt.date == pd.Timestamp('2026-01-02').date()]
    print(f"  First bar: {day_data.timestamp.min()}")
    print(f"  Last bar: {day_data.timestamp.max()}")

print("\n=== Options Data (spy_options_marks.csv) ===")
opt_df = pd.read_csv('data/synthetic_options/spy_options_marks.csv')
opt_df['timestamp'] = pd.to_datetime(opt_df['timestamp'], utc=True)
print(f"Total rows: {len(opt_df)}")
print(f"Date range: {opt_df.timestamp.min()} to {opt_df.timestamp.max()}")
print(f"Unique timestamps: {opt_df.timestamp.nunique()}")
print(f"2026-01-02 rows: {len(opt_df[opt_df.timestamp.dt.date == pd.Timestamp('2026-01-02').date()])}")
if len(opt_df[opt_df.timestamp.dt.date == pd.Timestamp('2026-01-02').date()]) > 0:
    day_data = opt_df[opt_df.timestamp.dt.date == pd.Timestamp('2026-01-02').date()]
    print(f"  Unique timestamps on 2026-01-02: {day_data.timestamp.nunique()}")
    print(f"  First timestamp: {day_data.timestamp.min()}")
    print(f"  Last timestamp: {day_data.timestamp.max()}")
