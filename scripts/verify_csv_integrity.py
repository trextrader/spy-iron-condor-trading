import os

file_path = r"data/ivolatility/spy_options_ivol_1year.csv"

if not os.path.exists(file_path):
    print("File not found.")
    exit()

line_count = 0
empty_lines = 0
null_bytes = False

with open(file_path, 'rb') as f:
    if b'\0' in f.read():
        null_bytes = True

with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f):
        line_count += 1
        if not line.strip():
            empty_lines += 1
        if i < 5:
            print(f"Line {i+1}: {line.strip()[:50]}...")

print(f"Total Lines: {line_count}")
print(f"Empty Lines: {empty_lines}")
print(f"Null Bytes Detected: {null_bytes}")
