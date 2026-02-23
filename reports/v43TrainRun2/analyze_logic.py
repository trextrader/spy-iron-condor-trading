import json
from collections import Counter

file_path = r'C:\SPYOptionTrader_test\reports\v43TrainRun2\Epoch6_CNv43_DSv43_02232026_154527.json'

with open(file_path, 'r') as f:
    data = json.load(f)

# Count the frequency of predicate pairs used across all SuperSets and Sets
pair_freq = Counter()
total_sets = 0

for ss in data['super_sets']:
    for s in ss['sets']:
        total_sets += 1
        for pair_idx in s['top_pair_indices']:
            pair_freq[pair_idx] += 1

print(f"--- CondorNet V43 Epoch 6 Analysis ---")
print(f"Total SuperSets: {data['n_super_sets']}")
print(f"Total Sets Evaluated: {total_sets}")
print(f"\nTop 15 Most Frequently Used Predicate Pair Combinations (Indices):")
for idx, count in pair_freq.most_common(15):
    print(f"  Pair Index {idx}: User {count} times")

print(f"\nPredicate Thresholds Extracted:")
for k, v in data['predicates'].items():
    print(f"  {k}: {round(v, 4)}")
