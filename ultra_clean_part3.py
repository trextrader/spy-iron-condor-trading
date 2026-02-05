
import os

filepath = "intelligence/audit_part_3.txt"
if not os.path.exists(filepath):
    print(f"Error: {filepath} not found.")
    exit(1)

with open(filepath, "rb") as f:
    data = f.read()

# Aggressive ASCII filter: Only allow printable ASCII, newline, carriage return, and tab
# range(32, 127) is printable ASCII
# 10 is \n, 13 is \r, 9 is \t
allowed_bytes = set([10, 13, 9] + list(range(32, 127)))

cleaned_data = bytes([b for b in data if b in allowed_bytes])

with open(filepath, "wb") as f:
    f.write(cleaned_data)

print(f"Ultra-Cleaned {filepath}. Reduced from {len(data)} to {len(cleaned_data)} bytes.")
