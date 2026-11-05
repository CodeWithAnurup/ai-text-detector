"""
Data preparation script for AI Text Detector (Phase 1).
Downloads/copies raw dataset to data/raw/, cleans data, samples 1000 balanced rows,
and performs stratified train/val/test split (70/15/15) with random_state=42.
"""

import os
import shutil
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

def main():
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    # 1. Download raw dataset
    print("Downloading raw dataset...")
    parquet_path = hf_hub_download(
        repo_id="alexok2006/ai-vs-human-texts",
        filename="data/train-00000-of-00001.parquet",
        repo_type="dataset"
    )
    
    raw_csv_path = os.path.join(raw_dir, "raw_ai_vs_human.csv")
    df_raw = pd.read_parquet(parquet_path)
    df_raw.to_csv(raw_csv_path, index=False)
    print(f"Raw data saved to {raw_csv_path} with {len(df_raw)} rows.")
    
    # 2. Keep only two columns: text, label (0 = human, 1 = AI)
    # Note: In alexok2006/ai-vs-human-texts, 0=AI, 1=Human. We remap to 0=human, 1=AI per plan.
    df = df_raw[["text", "label"]].copy()
    df["label"] = 1 - df["label"].astype(int)
    
    # 3. Clean: drop nulls and duplicates, drop texts under 30 words
    before_clean = len(df)
    df = df.dropna(subset=["text", "label"])
    df = df.drop_duplicates(subset=["text"])
    df["word_count"] = df["text"].apply(lambda t: len(str(t).strip().split()))
    df = df[df["word_count"] >= 30].drop(columns=["word_count"])
    print(f"Cleaned: {before_clean} -> {len(df)} rows.")
    print("Class distribution after cleaning:")
    print(df["label"].value_counts())
    
    # 4. Sample balanced 1000 rows (500 human, 500 AI), fixed random_state=42
    human_df = df[df["label"] == 0].sample(n=500, random_state=RANDOM_STATE)
    ai_df = df[df["label"] == 1].sample(n=500, random_state=RANDOM_STATE)
    
    sampled_df = pd.concat([human_df, ai_df]).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Sampled balanced dataset: {len(sampled_df)} rows ({sampled_df['label'].value_counts().to_dict()})")
    
    # 5. Split first: 70% train / 15% val / 15% test, stratified with random_state=42
    # Train: 700 (350 human, 350 AI), Remainder: 300 (150 human, 150 AI)
    train_df, rem_df = train_test_split(
        sampled_df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=sampled_df["label"]
    )
    
    # Val: 150 (75 human, 75 AI), Test: 150 (75 human, 75 AI)
    val_df, test_df = train_test_split(
        rem_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=rem_df["label"]
    )
    
    # Check for text overlap across splits
    train_texts = set(train_df["text"])
    val_texts = set(val_df["text"])
    test_texts = set(test_df["text"])
    
    assert len(train_texts.intersection(val_texts)) == 0, "Leakage between train and val!"
    assert len(train_texts.intersection(test_texts)) == 0, "Leakage between train and test!"
    assert len(val_texts.intersection(test_texts)) == 0, "Leakage between val and test!"
    
    # Check balanced in each
    assert train_df["label"].value_counts().to_dict() == {0: 350, 1: 350}, "Train not balanced!"
    assert val_df["label"].value_counts().to_dict() == {0: 75, 1: 75}, "Val not balanced!"
    assert test_df["label"].value_counts().to_dict() == {0: 75, 1: 75}, "Test not balanced!"
    
    # Save CSVs in data/
    train_path = os.path.join("data", "train.csv")
    val_path = os.path.join("data", "val.csv")
    test_path = os.path.join("data", "test.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved: {train_path} ({len(train_df)} rows, balanced 350/350)")
    print(f"Saved: {val_path} ({len(val_df)} rows, balanced 75/75)")
    print(f"Saved: {test_path} ({len(test_df)} rows, balanced 75/75)")
    print("Done when checks passed successfully!")

if __name__ == "__main__":
    main()
