#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shutil
from pathlib import Path

# === Set your base and destination directories ===
BASE_DIR = Path("./dataM")             # folder containing original dataset folders
DEST_BASE = BASE_DIR / "./merged_data"         # new folder for merged datasets
DEST_BASE.mkdir(parents=True, exist_ok=True)

# === Define datasets and their related folders ===
datasets = {
    "celebahq": [
        "celebahq_low",
        "celebahq_high",
        "celebahq_icst_low",
        "celebahq_icst_high"
    ],
    "INPI": [
        "INPI_sdllm_low",
        "INPI_sdllm_high",
        "INPI_icst_low",
        "INPI_icst_high"
    ],
    "INTB": [
        "INTB_sdllm_low",
        "INTB_sdllm_high",
        "INTB_icst_low",
        "INTB_icst_high"
    ],
    "cifar10": [
        "cifar10_sdllm_low",
        "cifar10_sdllm_high",
        "cifar10_icst_low",
        "cifar10_icst_high"
    ]
}

# === Merge all datasets ===
for dataset, folders in datasets.items():
    dest_folder = DEST_BASE / dataset
    dest_folder.mkdir(exist_ok=True)
    img_counter = 0

    print(f"\n🚀 Merging dataset: {dataset}")
    for folder_name in folders:
        folder = BASE_DIR / folder_name
        if not folder.exists():
            print(f"⚠️ Skipping missing folder: {folder_name}")
            continue

        print(f"  Processing: {folder_name}")
        for img_path in sorted(folder.glob("*.png")):
            stem = img_path.stem

            # --- Remove old numeric prefix (e.g., "76_")
            parts = stem.split("_", 1)
            if len(parts) > 1:
                stem = parts[1]

            # --- Split into tokens (e.g., ['cifar10', 'icst', 'low', 'iteration1', 'X6', 'Y2'])
            tokens = stem.split("_")

            # --- Extract key parts safely
            iteration = next((p for p in tokens if p.startswith("iteration")), "iteration0")
            x_part = next((p for p in tokens if p.startswith("X")), "XNA")
            y_part = next((p for p in tokens if p.startswith("Y")), "YNA")

            # --- Clean duplicates like sdllm_high_sdllm_high
            seen = set()
            clean_tokens = []
            for t in tokens:
                if t not in seen:
                    clean_tokens.append(t)
                    seen.add(t)

            # --- Remove iteration/X/Y from clean_tokens to avoid duplication
            clean_tokens = [t for t in clean_tokens if not (t.startswith("iteration") or t.startswith("X") or t.startswith("Y"))]

            # --- Rebuild the name consistently
            base_name = "_".join(clean_tokens)
            new_filename = f"{img_counter}_{base_name}_{iteration}_{x_part}_{y_part}.png"

            new_path = dest_folder / new_filename
            shutil.copy2(img_path, new_path)
            img_counter += 1

    print(f"✅ {dataset}: merged {img_counter} images into {dest_folder}")

print("\n🎉 All datasets merged successfully into 'merged_data' folder.")



