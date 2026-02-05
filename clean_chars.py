
import os

filepath = "intelligence/audit_part_3.txt"
if not os.path.exists(filepath):
    print(f"Error: {filepath} not found.")
    exit(1)

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Mapping for common non-ASCII symbols in the audit code
mapping = {
    "™": "(TM)",
    "θ": "theta",
    "Δ": "delta",
    "φ": "phi",
    "₁": "_1",
    "α": "alpha",
    "β": "beta",
    "⊙": "*",
    "Π": "Pi",
    "π": "pi"
}

cleaned_content = content
for symbol, replacement in mapping.items():
    cleaned_content = cleaned_content.replace(symbol, replacement)

# Final sweep to ensure only ASCII characters remain (optional but safer)
# For now, just replacing the known culprits.

with open(filepath, "w", encoding="utf-8") as f:
    f.write(cleaned_content)

print(f"Successfully cleaned {filepath}")
