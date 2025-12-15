import os
import json
import random
import re
import unicodedata
import argparse
from datasets import load_dataset
from typing import List, Dict, Literal
from tqdm import tqdm
from utils import set_seed
import pandas as pd

set_seed()

SAVE_DIR = "./dataset/processed"
RAW_DIR = "./dataset/downloaded_data"

def clean_text(text: str) -> str:

    # Chuẩn hóa Unicode
    text = unicodedata.normalize("NFKC", text)
    
    # Loại bỏ ký tự không mong muốn và chuẩn hóa khoảng trắng
    text = re.sub(r'[\u200b\u200e\u200f\ufeff]', '', text)
    
    # Thay thế mọi loại khoảng trắng (tab, xuống dòng, space lạ) bằng một space thường
    text = re.sub(r'[\s\t\n\r\xa0]+', ' ', text)
    
    # Loại bỏ các ký tự đặc biệt thừa ở đầu và cuối câu
    text = re.sub(r'^[\-\*\•\>]\s+', '', text)
    text = re.sub(r'\s+[\-\*\•\>]$', '', text)
    
    # Chuẩn hóa dấu câu
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    
    # Thêm khoảng trắng sau dấu câu nếu thiếu
    text = re.sub(r'([,.!?;:])(?=[a-zA-Z0-9])', r'\1 ', text)
    
    return text.strip()

def prepare_training_dataset(output_format: Literal["jsonl", "parquet"] = "parquet"):
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    output_file = os.path.join(
        SAVE_DIR, 
        f"train.{output_format}"
    )

    print(f"Preparing training dataset (format: {output_format})...")
    ds = load_dataset(
        "text",
        data_files= {
            "en": os.path.join(RAW_DIR, "train.en"),
            "vi": os.path.join(RAW_DIR, "train.vi")
        },
        streaming=False
    )

    total_rows = len(ds["en"])
    print(f"Total rows: {total_rows}")
    en_list = ds["en"]["text"]
    vi_list = ds["vi"]["text"]

    processed_count = 0
    skipped_count = 0

    data = []
    
    for i in tqdm(range(total_rows)):
        try:
            en_text = clean_text(en_list[i])
            vi_text = clean_text(vi_list[i])

            if not en_text or \
                    not vi_text or \
                    len(en_text) < 2\
                    or len(vi_text) < 2:
                skipped_count += 1
                continue

            data.append({
                "translation": {
                    "en": en_text,
                    "vi": vi_text
                }
            })
            processed_count += 1    
        except Exception as e:
            print(f"Error processing row {i}: {e}")
            skipped_count += 1    
            continue

    random.shuffle(data)
    
    print(f"Processed {processed_count} rows, skipped {skipped_count} rows")
    
    if output_format == "jsonl":
        print(f"Saving as JSONL to {output_file}...")
        with open(output_file, "w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")
    elif output_format == "parquet":
        print(f"Saving as Parquet to {output_file}...")
        pd.DataFrame(data).to_parquet(output_file, engine="pyarrow")
    else:
        raise ValueError(f"Invalid output format: {output_format}")

    print(f"\n✅ Training data saved to {output_file}")
    print(f"   Format: {output_format.upper()}")
    print(f"   Processed: {processed_count:,} samples")
    print(f"   Skipped: {skipped_count:,} samples")
    
    file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    print(f"   File size: {file_size:.2f} MB")

def prepare_validation_data(output_format: Literal["jsonl", "parquet"] = "parquet"):
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    output_file = os.path.join(
        SAVE_DIR, 
        f"validation.{output_format}"
    )

    print(f"Preparing validation dataset (format: {output_format})...")
    
    # Validation data consists of tst2012 and tst2013
    val_files = {
        "tst2012": {
            "en": os.path.join(RAW_DIR, "tst2012.en"),
            "vi": os.path.join(RAW_DIR, "tst2012.vi")
        },
        "tst2013": {
            "en": os.path.join(RAW_DIR, "tst2013.en"),
            "vi": os.path.join(RAW_DIR, "tst2013.vi")
        }
    }

    data = []
    processed_count = 0
    skipped_count = 0

    for dataset_name, files in val_files.items():
        print(f"Processing {dataset_name}...")
        ds = load_dataset(
            "text",
            data_files={
                "en": files["en"],
                "vi": files["vi"]
            },
            streaming=False
        )

        total_rows = len(ds["en"])
        en_list = ds["en"]["text"]
        vi_list = ds["vi"]["text"]

        for i in tqdm(range(total_rows), desc=f"Processing {dataset_name}"):
            try:
                en_text = clean_text(en_list[i])
                vi_text = clean_text(vi_list[i])

                if not en_text or \
                        not vi_text or \
                        len(en_text) < 2\
                        or len(vi_text) < 2:
                    skipped_count += 1
                    continue

                data.append({
                    "translation": {
                        "en": en_text,
                        "vi": vi_text
                    }
                })
                processed_count += 1    
            except Exception as e:
                print(f"Error processing row {i} in {dataset_name}: {e}")
                skipped_count += 1    
                continue
    
    print(f"Processed {processed_count} rows, skipped {skipped_count} rows")
    
    if output_format == "jsonl":
        print(f"Saving as JSONL to {output_file}...")
        with open(output_file, "w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")
    elif output_format == "parquet":
        print(f"Saving as Parquet to {output_file}...")
        pd.DataFrame(data).to_parquet(output_file, engine="pyarrow")
    else:
        raise ValueError(f"Invalid output format: {output_format}")

    print(f"\n✅ Validation data saved to {output_file}")
    print(f"   Format: {output_format.upper()}")
    print(f"   Processed: {processed_count:,} samples")
    print(f"   Skipped: {skipped_count:,} samples")
    
    file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    print(f"   File size: {file_size:.2f} MB")
    
if __name__ == "__main__":
    prepare_training_dataset()
    prepare_validation_data()