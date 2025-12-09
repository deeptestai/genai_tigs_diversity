import re
import pandas as pd

# === Step 1: Load MTurk batch file ===
path = "./answer/celeb_result_mturk1.csv"
df = pd.read_csv(path)

# === Step 2: Identify image columns ===
image_cols = sorted(
    [c for c in df.columns if re.search(r'Input\.image_url\d+$', c)],
    key=lambda x: int(re.search(r'(\d+)$', x).group(1))
)
image_map = {int(re.search(r'(\d+)$', c).group(1)): c for c in image_cols}

# === Step 3: Identify all answer columns (category + ACQ) ===
answer_cols = [c for c in df.columns if c.startswith("Answer.")]

# === Step 4: Extract long-format rows ===
rows = []
for _, r in df.iterrows():
    for ac in answer_cols:
        # Match both category and ACQ question formats
        m = re.match(r'Answer\.(?:category|ACQ)_(\d+)_(.+?)\.(.+)', ac)
        if not m:
            continue

        cat_idx = int(m.group(1))
        question_type = "ACQ" if "ACQ" in ac else "category"
        label = m.group(3)
        val = r[ac]
        img_col = image_map.get(cat_idx, image_cols[0])
        img_url = r[img_col]

        rows.append({
            "WorkerId": r.get("WorkerId"),
            "image_url": img_url,
            "question_type": question_type,
            "option_label": label,
            "option_value": bool(val) if not pd.isna(val) else None
        })

# === Step 5: Convert to DataFrame ===
long_df = pd.DataFrame(rows)

# === Step 6: Keep only True options ===
true_df = long_df[long_df["option_value"] == True]

# === Step 7: Aggregate each worker’s True options per image ===
worker_grouped = (
    true_df.groupby(["WorkerId", "image_url"])["option_label"]
    .apply(lambda x: ", ".join(sorted(set(x))))
    .reset_index()
)

# === Step 8: Merge exactly two workers’ responses per image ===
image_groups = (
    worker_grouped.groupby("image_url")["option_label"]
    .apply(list)
    .reset_index()
)

# Handle cases with fewer or more than two workers
image_groups["Worker_1_true_options"] = image_groups["option_label"].apply(lambda x: x[0] if len(x) > 0 else "")
image_groups["Worker_2_true_options"] = image_groups["option_label"].apply(lambda x: x[1] if len(x) > 1 else "")

# Keep only the required columns
final_df = image_groups[["image_url", "Worker_1_true_options", "Worker_2_true_options"]]

# === Step 9: Save final file ===
final_df.to_csv("mturk_answer_extract1.csv", index=False)

print(f"Done! Saved: image_true_options_two_workers_with_acq.csv ({len(final_df)} images)")
print(final_df.head(10))
