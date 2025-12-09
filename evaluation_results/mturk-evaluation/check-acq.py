#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MTurk Results ACQ Checker (Correct Label = 'cat' for vr_ images)
Auto-detects HIT ID column.
"""

import pandas as pd

# === Load data ===
df = pd.read_csv("./answer/Batch_intb1_batch_results.csv", sep=',', dtype=str)  # use ',' if comma-separated

# === Normalize column names ===
df.columns = df.columns.str.strip()

# === Detect HIT ID column automatically ===
hit_col = None
for col in df.columns:
    if col.lower().replace(" ", "") in ["hitid", "hit_id", "hit-id"]:
        hit_col = col
        break

if not hit_col:
    raise KeyError("❌ Could not find any HIT ID column in your file. Please verify headers.")

print(f"✅ Using '{hit_col}' as HIT ID column.")

# === Detect relevant columns ===
image_cols = [col for col in df.columns if col.startswith("Input.image_url")]
answer_cols = [col for col in df.columns if col.startswith("Answer.category_") and not col.endswith(".label")]

expected_correct_label = "cat"
records = []

# === Group by HIT ID ===
for hit_id, group in df.groupby(hit_col):
    if len(group) != 2:
        continue

    worker1, worker2 = group.iloc[0], group.iloc[1]

    for i, img_col in enumerate(image_cols):
        if i >= len(answer_cols):
            continue

        img_url = str(worker1[img_col])
        label1 = str(worker1[answer_cols[i]]).strip()
        label2 = str(worker2[answer_cols[i]]).strip()

        if pd.isna(img_url) or img_url == "nan":
            continue

        is_acq = "vr_" in img_url.lower()
        acq_check_1 = acq_check_2 = ""
        highlight = ""

        if is_acq:
            acq_check_1 = "✔" if label1.lower() == expected_correct_label else "✘"
            acq_check_2 = "✔" if label2.lower() == expected_correct_label else "✘"

            if acq_check_1 == "✘" and acq_check_2 == "✘":
                highlight = "❌ Both Wrong"
            elif acq_check_1 == "✘":
                highlight = "⚠ Worker1 Wrong"
            elif acq_check_2 == "✘":
                highlight = "⚠ Worker2 Wrong"
            else:
                highlight = "✔ Both Correct"

        records.append({
            "HITId": hit_id,
            "Image_URL": img_url,
            "Worker1_ID": worker1.get("WorkerId", ""),
            "Worker1_Label": label1,
            "Worker1_ACQ": acq_check_1,
            "Worker2_ID": worker2.get("WorkerId", ""),
            "Worker2_Label": label2,
            "Worker2_ACQ": acq_check_2,
            "ACQ_Result": highlight
        })

# === Save output ===
out_df = pd.DataFrame(records)
out_df.to_csv("mturk_acq_vr_cat_check.csv", index=False)

print("✅ Output saved as 'mturk_acq_vr_cat_check.csv'")
print(out_df[out_df["ACQ_Result"] != ""].head())
