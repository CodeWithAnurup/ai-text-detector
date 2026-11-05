"""
Model training and probability calibration pipeline (Phase 5).
Compares StandardScaler + LogisticRegression vs RandomForestClassifier,
calibrates probabilities using CalibratedClassifierCV, and saves models/detector.joblib.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report

FEATURE_COLS = ["ppl", "burstiness", "num_tokens", "avg_sent_len", "ttr"]
RANDOM_STATE = 42

def load_data():
    train_df = pd.read_csv("data/processed/features_train.csv")
    val_df = pd.read_csv("data/processed/features_val.csv")
    
    X_train = train_df[FEATURE_COLS]
    y_train = train_df["label"]
    
    X_val = val_df[FEATURE_COLS]
    y_val = val_df["label"]
    
    return X_train, y_train, X_val, y_val

def train_and_evaluate():
    os.makedirs("models", exist_ok=True)
    X_train, y_train, X_val, y_val = load_data()
    
    # 1. Pipeline 1: StandardScaler + LogisticRegression
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))
    ])
    lr_pipe.fit(X_train, y_train)
    lr_val_probs = lr_pipe.predict_proba(X_val)[:, 1]
    lr_val_preds = lr_pipe.predict(X_val)
    lr_auc = roc_auc_score(y_val, lr_val_probs)
    lr_f1 = f1_score(y_val, lr_val_preds)
    lr_acc = accuracy_score(y_val, lr_val_preds)
    
    print("=== Model 1: Logistic Regression ===")
    print(f"Validation ROC-AUC: {lr_auc:.4f}")
    print(f"Validation F1:      {lr_f1:.4f}")
    print(f"Validation Acc:     {lr_acc:.4f}")
    
    # 2. Pipeline 2: RandomForestClassifier
    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, max_depth=6))
    ])
    rf_pipe.fit(X_train, y_train)
    rf_val_probs = rf_pipe.predict_proba(X_val)[:, 1]
    rf_val_preds = rf_pipe.predict(X_val)
    rf_auc = roc_auc_score(y_val, rf_val_probs)
    rf_f1 = f1_score(y_val, rf_val_preds)
    rf_acc = accuracy_score(y_val, rf_val_preds)
    
    print("\n=== Model 2: Random Forest ===")
    print(f"Validation ROC-AUC: {rf_auc:.4f}")
    print(f"Validation F1:      {rf_f1:.4f}")
    print(f"Validation Acc:     {rf_acc:.4f}")
    
    # 3. Model selection: prefer Logistic Regression if scores are close
    # Plan rule: "Pick the better one (prefer Logistic Regression if scores are close, since it's simpler and gives better-behaved probabilities)."
    diff = rf_auc - lr_auc
    if diff > 0.02:
        selected_name = "RandomForest"
        base_pipeline = rf_pipe
    else:
        selected_name = "LogisticRegression"
        base_pipeline = lr_pipe
        
    print(f"\nSelected Model: {selected_name} (difference: {diff:.4f})")
    
    # 4. Calibrate with CalibratedClassifierCV (method='sigmoid', cv=5)
    print("Calibrating model with CalibratedClassifierCV (method='sigmoid', cv=5)...")
    calibrated_clf = CalibratedClassifierCV(
        estimator=base_pipeline,
        method="sigmoid",
        cv=5
    )
    calibrated_clf.fit(X_train, y_train)
    
    cal_val_probs = calibrated_clf.predict_proba(X_val)[:, 1]
    cal_val_preds = calibrated_clf.predict(X_val)
    cal_auc = roc_auc_score(y_val, cal_val_probs)
    cal_f1 = f1_score(y_val, cal_val_preds)
    cal_acc = accuracy_score(y_val, cal_val_preds)
    
    print("\n=== Calibrated Model Performance (Validation) ===")
    print(f"Calibrated ROC-AUC: {cal_auc:.4f}")
    print(f"Calibrated F1:      {cal_f1:.4f}")
    print(f"Calibrated Acc:     {cal_acc:.4f}")
    print("\nClassification Report (Validation):")
    print(classification_report(y_val, cal_val_preds, target_names=["Human", "AI"]))
    
    # Load train imputation medians if available
    medians_path = "data/processed/train_medians.json"
    if os.path.exists(medians_path):
        with open(medians_path, "r") as f:
            train_medians = json.load(f)
    else:
        train_medians = X_train.median().to_dict()
        
    # Save package
    model_artifact = {
        "model": calibrated_clf,
        "base_model_name": selected_name,
        "feature_cols": FEATURE_COLS,
        "train_medians": train_medians,
        "metrics_val": {
            "lr_auc": float(lr_auc),
            "lr_f1": float(lr_f1),
            "rf_auc": float(rf_auc),
            "rf_f1": float(rf_f1),
            "cal_auc": float(cal_auc),
            "cal_f1": float(cal_f1),
            "cal_acc": float(cal_acc)
        }
    }
    
    save_path = os.path.join("models", "detector.joblib")
    joblib.dump(model_artifact, save_path)
    print(f"Successfully saved calibrated model artifact to {save_path}")
    
    return model_artifact

if __name__ == "__main__":
    train_and_evaluate()
