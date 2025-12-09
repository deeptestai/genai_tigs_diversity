#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MTurk Questionnaire Generator for INPI Dataset
Author: Maryam
Description:
This script prepares randomized image URLs for MTurk surveys, displaying
10 images (questions) per page + 1 ACQ (attention-check) image.
"""

import csv
import glob
import ntpath
import random
from math import floor
from pathlib import Path

# === CONFIGURATION ===
survey_folder = Path("./merged_data/INPI")  
base_url = "https://sdllm-inpi-bucket1.s3.eu-central-1.amazonaws.com/"
csv_file = "inpi_mturk_links.csv"
extra_csv_file = "inpi_mturk_links_extra.csv"

# ACQ (attention check) image
acq_url = base_url + "vr_0_e281_pNone.png"

# Number of questions (images per survey page)
number_questions = 10  

# === LOAD IMAGES ===
images = glob.glob(str(survey_folder / "*"))
total_images = len(images)
print(f"Found {total_images} images in INPI folder.")

# Ensure reproducible shuffle
random.seed(42)
random.shuffle(images)

# === CREATE PARTITIONS ===
number_surveys = floor(total_images / number_questions)
remainder = total_images % number_questions

# Add one more survey if leftover images exist
extra_survey = remainder != 0
if extra_survey:
    number_surveys += 1

partitions = [images[i:i + number_questions] for i in range(0, total_images, number_questions)]

# === MAIN CSV FILE ===
with open(csv_file, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    header = [f"image_url{i}" for i in range(number_questions + 1)]
    writer.writerow(header)

    for i in range(number_surveys - (1 if extra_survey else 0)):
        row = []
        for element in partitions[i]:
            element_name = ntpath.basename(element)
            link = base_url + element_name
            row.append(link)
        row.append(acq_url)
        random.shuffle(row)
        writer.writerow(row)

# === EXTRA CSV FILE (if leftover images) ===
if extra_survey:
    with open(extra_csv_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        header = [f"image_url{i}" for i in range(len(partitions[-1]) + 1)]
        writer.writerow(header)
        row = []
        for element in partitions[-1]:
            element_name = ntpath.basename(element)
            link = base_url + element_name
            row.append(link)
        row.append(acq_url)
        random.shuffle(row)
        writer.writerow(row)

print(f"\n CSV files created successfully!")
print(f" - Main file: {csv_file}")
print(f" - Extra file: {extra_csv_file if extra_survey else 'No extra file created'}")
