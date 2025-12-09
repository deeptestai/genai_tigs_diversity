#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

# === Set your base directory path ===
BASE_DIR = Path("./dataM")  # Change this to your actual path

# === Process each subfolder ===
for folder in BASE_DIR.iterdir():
    if folder.is_dir():
        folder_name = folder.name
        print(f"Processing folder: {folder_name}")
        
        # Collect all .png images
        images = sorted(folder.glob("*.png"))
        
        # Rename images with new format
        for idx, img_path in enumerate(images):
            # Extract iteration, X, Y from original filename
            parts = img_path.stem.split("_")  # e.g. ['image', '2', 'iteration1', 'X5', 'Y0']
            if len(parts) >= 5:
                iteration = parts[2]  # 'iteration1'
                x_part = parts[3]     # 'X5'
                y_part = parts[4]     # 'Y0'
            else:
                print(f"Skipping unexpected file name: {img_path.name}")
                continue

            # Construct new filename
            new_filename = f"{idx}_{folder_name}_{iteration}_{x_part}_{y_part}.png"
            new_path = img_path.parent / new_filename

            # Rename file
            img_path.rename(new_path)

        print(f"Renamed {len(images)} images in {folder_name}\n")
