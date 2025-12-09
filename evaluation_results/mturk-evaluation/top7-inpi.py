import re
from pathlib import Path
from collections import defaultdict

# === Configuration ===
IMAGE_DIR = Path("merged_data/INTB")   # change this path if needed
TOP_N = 7

# === Helper: extract Y value ===
def extract_y_value(filename):
    match = re.search(r"Y(\d+)", filename)
    return int(match.group(1)) if match else None

# === Count occurrences of each Y ===
y_counts = defaultdict(int)
for f in IMAGE_DIR.glob("*.png"):
    y_val = extract_y_value(f.name)
    if y_val is not None:
        y_counts[y_val] += 1

# === Sort by count descending ===
sorted_counts = sorted(y_counts.items(), key=lambda x: x[1], reverse=True)

# === Display top N ===
print(f"\nTop {TOP_N} Y labels by count:\n")
for yval, count in sorted_counts[:TOP_N]:
    print(f"Y={yval}\tCount={count}")
