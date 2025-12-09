import pandas as pd
import re

# === Step 1: Load the worker output ===
df = pd.read_csv("mturk_answer_combined.csv")

# === Step 2: Extract filename from image_url ===
df["ID_Image_URL"] = df["image_url"].apply(lambda x: x.split("/")[-1])

# === Step 3: Parse filename into structured columns ===
def parse_filename(fname):
    # Example: 100_celebahq_high_iteration199_X31_Y0.png
    m = re.match(r'(\d+)_([a-zA-Z]+)_(.*?)_iteration(\d+)_X(\d+)_Y(\d+)\.png', fname)
    if not m:
        return pd.Series([None]*7, index=["ID","Dataset","TOOL","Mut_extent","Iteration","Expected","Predicted"])
    
    ID = m.group(1)
    dataset = m.group(2)
    maybe_tool = m.group(3)
    iteration = m.group(4)
    expected = m.group(5)
    predicted = m.group(6)
    
    # TOOL & mutation extent logic
    if maybe_tool.startswith("icst"):
        TOOL = "icst"
        mut_extent = maybe_tool.replace("icst_", "")
    else:
        TOOL = "sdllm"
        mut_extent = maybe_tool
    
    return pd.Series({
        "ID": ID,
        "Dataset": dataset,
        "TOOL": TOOL,
        "Mut_extent": mut_extent,
        "Iteration": iteration,
        "Expected": expected,
        "Predicted": predicted
    })

parsed = df["ID_Image_URL"].apply(parse_filename)
df = pd.concat([df, parsed], axis=1)

# === Step 4: Map Expected (X value) → readable label ===
x_to_label = {
    "31": "Smiling",
    "15": "Eyeglasses",
    "5": "Bangs"
}

df["Expected_Label"] = df["Expected"].map(x_to_label).fillna(df["Expected"])

# === Step 5: Assign ASS1 and ASS2 (1 if worker marked that label true) ===
def assign_worker_label(row):
    label = x_to_label.get(row["Expected"], None)
    w1 = 1 if (label and label in str(row["Worker_1_true_options"])) else 0
    w2 = 1 if (label and label in str(row["Worker_2_true_options"])) else 0
    return pd.Series({"ASS1": w1, "ASS2": w2})

assignments = df.apply(assign_worker_label, axis=1)
df = pd.concat([df, assignments], axis=1)

# === Step 6: Final column arrangement ===
final_cols = [
    "ID_Image_URL", "ID", "Dataset", "TOOL", "Mut_extent",
    "Iteration", "Expected", "Expected_Label", "Predicted",
    "Worker_1_true_options", "Worker_2_true_options",
    "ASS1", "ASS2"
]
final_df = df[final_cols]

# === Step 7: Save output ===
final_df.to_csv("celeb_final_mturk.csv", index=False)

print(f"Done! Saved file: image_label_match_summary_with_labels.csv ({len(final_df)} rows)")
print(final_df.head(10))
