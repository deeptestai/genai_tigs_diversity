import pandas as pd
import glob
import os
import numpy as np
from scipy.stats import ttest_rel, pearsonr, spearmanr
from scipy.stats import ttest_ind
#t_stat, p_val = ttest_ind(a, b, equal_var=False)

# === Configuration ===
data_path = "/home/vincenzo.riccio/ddpm_diffuser/style-generator/diversity_stat_test1/full-diversity-stat"
csv_files = glob.glob(os.path.join(data_path, "*.csv"))
results = []

def parse_filename(fname):
    name = os.path.basename(fname).replace(".csv", "")
    parts = name.split("_")
    dataset = parts[0].upper() if len(parts) > 0 else "UNKNOWN"
    model = "_".join(parts[1:]).upper() if len(parts) > 1 else "UNKNOWN"
    return dataset, model

for file in csv_files:
    dataset, model = parse_filename(file)
    print(f" Reading: {os.path.basename(file)} → {dataset}, {model}")

    try:
        df = pd.read_csv(file, sep=None, engine="python")
    except Exception as e:
        print(f"Could not read {file}: {e}")
        continue

    df.columns = [c.strip().lower() for c in df.columns]
    if "diversity" not in df.columns or "group" not in df.columns:
        print(f"Skipping {file}: Missing 'group' or 'diversity' column.")
        continue

    df["diversity"] = pd.to_numeric(df["diversity"], errors="coerce")
    df = df.dropna(subset=["diversity"])

    # Collect per-condition values
    icst_low = df.loc[df["group"] == "ICST_LOW", "diversity"].values
    sdllm_low = df.loc[df["group"] == "SDLLM_LOW", "diversity"].values
    icst_high = df.loc[df["group"] == "ICST_HIGH", "diversity"].values
    sdllm_high = df.loc[df["group"] == "SDLLM_HIGH", "diversity"].values
    icst_overall = df.loc[df["group"] == "ICST_OVERALL", "diversity"].values
    sdllm_overall = df.loc[df["group"] == "SDLLM_OVERALL", "diversity"].values
    def analyze_condition(cond, a, b):
        if len(a) and len(b):
            min_len = min(len(a), len(b))
            a, b = a[:min_len], b[:min_len]
            print(f"🔹 {dataset}-{model}-{cond}: Paired {len(a)} ICST vs {len(b)} SDLLM values")
            t_stat, p_val = ttest_rel(a, b)
            pearson, _ = pearsonr(a, b)
            spearman, _ = spearmanr(a, b)
            diff = a - b
            cohen_d = np.mean(diff) / np.std(diff, ddof=1)
            return {
                "Dataset": dataset, "Model": model, "Condition": cond,
                "Mean_ICST": np.mean(a), "Mean_SDLLM": np.mean(b),
                "Mean_Diff": np.mean(a) - np.mean(b),
                "t_stat": t_stat, "p_value": p_val,
                "Pearson_r": pearson, "Spearman_rho": spearman,"Cohen_d": cohen_d,
                "Direction": "ICST > SDLLM" if np.mean(a) > np.mean(b) else "SDLLM > ICST",
                "Significant_(p<0.05)": "Yes" if p_val < 0.05 else "No"
            }

    # Low, High, Overall comparisons
    for cond, a, b in [
        ("LOW", icst_low, sdllm_low),
        ("HIGH", icst_high, sdllm_high),
        ("OVERALL", icst_overall, sdllm_overall)
    ]:
        res = analyze_condition(cond, a, b)
        if res:
            results.append(res)

# Save results
out = pd.DataFrame(results)
out = out.sort_values(["Dataset", "Model", "Condition"])
out_file = os.path.join(data_path, "paired_ttest_results_full.csv")
out.to_csv(out_file, index=False)

print("\n Results saved to:", out_file)
print(out)
