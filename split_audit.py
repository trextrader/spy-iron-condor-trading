
import os

filepath = "intelligence/audit_consolidated.txt"
output_prefix = "intelligence/audit_part_"
target_size = 15000
tolerance = 2500

if not os.path.exists(filepath):
    print(f"Error: {filepath} not found.")
    exit(1)

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

parts = []
current_pos = 0
total_len = len(content)

while current_pos < total_len:
    end_pos = min(current_pos + target_size, total_len)
    
    # If not at the end, try to find a clean break point
    if end_pos < total_len:
        # Look for the file header pattern # ===
        # Searching within the window [end_pos - tolerance, end_pos + tolerance]
        search_start = max(current_pos + target_size - tolerance, current_pos)
        search_end = min(current_pos + target_size + tolerance, total_len)
        
        window = content[search_start:search_end]
        header_idx = window.rfind("# =========")
        
        if header_idx != -1:
            end_pos = search_start + header_idx
        else:
            # Fallback to double newline
            newline_idx = window.rfind("\n\n")
            if newline_idx != -1:
                end_pos = search_start + newline_idx + 2
            else:
                # Last resort: just break at target_size
                pass
                
    parts.append(content[current_pos:end_pos])
    current_pos = end_pos

for i, part in enumerate(parts):
    out_file = f"{output_prefix}{i+1}.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(part)
    print(f"Created {out_file} ({len(part)} chars)")

print(f"Successfully split into {len(parts)} parts.")
