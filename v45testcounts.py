import pandas as pd

df = pd.read_csv('data/Datasetv4/v43/m5_dataset_v43_final.csv', usecols=['strategy_label'])
print(df['strategy_label'].value_counts())
print(f"\nTotal bars: {len(df)}")
print(f"IC (label=7) pct: {(df['strategy_label'] == 7).mean()*100:.1f}%")
