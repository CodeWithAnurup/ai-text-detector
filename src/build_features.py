"""
Extract and cache features across train, val, and test splits (Phase 3).
"""

import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from tqdm import tqdm
from src.features import extract_features, get_gpt2

PROCESSED_DIR = os.path.join("data", "processed")
FEATURE_COLS = ["ppl", "burstiness", "num_tokens", "avg_sent_len", "ttr"]

def process_split(split_name: str) -> pd.DataFrame:
    raw_path = os.path.join("data", f"{split_name}.csv")
    out_path = os.path.join(PROCESSED_DIR, f"features_{split_name}.csv")
    
    if os.path.exists(out_path):
        print(f"Features for {split_name} already exist at {out_path}, loading cached file...")
        return pd.read_csv(out_path)
        
    print(f"Extracting features for {split_name} from {raw_path}...")
    df = pd.read_csv(raw_path)
    
    records = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
        text = str(row["text"])
        label = int(row["label"])
        feats = extract_features(text)
        feats["label"] = label
        records.append(feats)
        
    feat_df = pd.DataFrame(records)
    
    # Reorder columns explicitly: ppl, burstiness, num_tokens, avg_sent_len, ttr, label
    feat_df = feat_df[["ppl", "burstiness", "num_tokens", "avg_sent_len", "ttr", "label"]]
    
    # Save raw features before imputation
    feat_df.to_csv(out_path, index=False)
    print(f"Saved {len(feat_df)} rows to {out_path}")
    return feat_df

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Pre-load model once
    get_gpt2()
    
    # 1. Process or load each split
    train_df = process_split("train")
    val_df = process_split("val")
    test_df = process_split("test")
    
    # 2. Check and handle NaNs: Impute using train median (no data leakage)
    train_medians = {}
    for col in FEATURE_COLS:
        med = float(train_df[col].median(skipna=True))
        train_medians[col] = med
        
    print(f"Computed train set medians for imputation: {train_medians}")
    
    # Save imputation medians for inference pipeline
    with open(os.path.join(PROCESSED_DIR, "train_medians.json"), "w") as f:
        json.dump(train_medians, f, indent=2)
        
    # Apply imputation to all splits
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        nan_count_before = df.isna().sum().to_dict()
        for col in FEATURE_COLS:
            df[col] = df[col].fillna(train_medians[col])
        out_path = os.path.join(PROCESSED_DIR, f"features_{name}.csv")
        df.to_csv(out_path, index=False)
        nan_count_after = df.isna().sum().sum()
        print(f"Split {name}: NaNs before = {nan_count_before}, NaNs after = {nan_count_after}")
        assert nan_count_after == 0, f"NaNs still present in features_{name}.csv!"
        
    print("Phase 3 complete! All feature CSVs verified and saved without NaNs.")

if __name__ == "__main__":
    main()
