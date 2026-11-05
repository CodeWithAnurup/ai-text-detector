"""
Phase 6: Final Evaluation on the untouched Test Set.
Computes accuracy, precision, recall, F1, ROC-AUC, confusion matrix,
plots calibration curve to docs/, performs error analysis, and creates notebooks/02_modeling.ipynb.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve

def main():
    os.makedirs("docs", exist_ok=True)
    os.makedirs("notebooks", exist_ok=True)
    
    # 1. Load trained calibrated model
    artifact = joblib.load("models/detector.joblib")
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    
    # 2. Load test data
    test_feat_df = pd.read_csv("data/processed/features_test.csv")
    test_raw_df = pd.read_csv("data/test.csv")
    
    X_test = test_feat_df[feature_cols]
    y_test = test_feat_df["label"].values
    
    # Predict probabilities and labels
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    
    # Compute metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)
    
    print("=== FINAL TEST SET EVALUATION ===")
    print(f"Accuracy:        {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision:       {prec:.4f}")
    print(f"Recall:          {rec:.4f}")
    print(f"F1-Score:        {f1:.4f}")
    print(f"ROC-AUC:         {auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"True Human (0):  TN={cm[0, 0]} | FP={cm[0, 1]}")
    print(f"True AI (1):     FN={cm[1, 0]} | TP={cm[1, 1]}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["Human (0)", "AI (1)"]))
    
    # 2. Plot Calibration Curve
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10, strategy="uniform")
    
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.plot(prob_pred, prob_true, "s-", color="royalblue", label=f"Calibrated Model (AUC = {auc:.3f})")
    plt.title("Calibration Curve (Test Set)", fontsize=13)
    plt.xlabel("Mean Predicted Probability", fontsize=11)
    plt.ylabel("Fraction of Positives (Actual AI)", fontsize=11)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    cal_plot_path = os.path.join("docs", "calibration_curve.png")
    plt.savefig(cal_plot_path, dpi=300)
    plt.close()
    print(f"Saved calibration curve to {cal_plot_path}")
    
    # Save metrics table markdown
    metrics_md = f"""# Test Set Performance Metrics

| Metric | Score |
|---|---|
| **Accuracy** | {acc*100:.2f}% |
| **ROC-AUC** | {auc:.4f} |
| **Precision** | {prec:.4f} |
| **Recall** | {rec:.4f} |
| **F1 Score** | {f1:.4f} |

### Confusion Matrix
- **True Negative (Human detected as Human)**: {cm[0, 0]} / {cm[0, 0] + cm[0, 1]}
- **False Positive (Human detected as AI)**: {cm[0, 1]} / {cm[0, 0] + cm[0, 1]}
- **False Negative (AI detected as Human)**: {cm[1, 0]} / {cm[1, 0] + cm[1, 1]}
- **True Positive (AI detected as AI)**: {cm[1, 1]} / {cm[1, 0] + cm[1, 1]}
"""
    with open("docs/test_metrics.md", "w") as f:
        f.write(metrics_md)
        
    # 3. Error Analysis
    # Combine raw texts and predictions
    eval_df = test_raw_df.copy()
    eval_df["predicted_prob"] = probs
    eval_df["prediction"] = preds
    for col in feature_cols:
        eval_df[col] = test_feat_df[col]
        
    fps = eval_df[(eval_df["label"] == 0) & (eval_df["prediction"] == 1)]
    fns = eval_df[(eval_df["label"] == 1) & (eval_df["prediction"] == 0)]
    
    print(f"\nTotal False Positives (Human -> AI): {len(fps)}")
    print(f"Total False Negatives (AI -> Human): {len(fns)}")
    
    # Create 02_modeling.ipynb
    create_modeling_notebook(acc, auc, prec, rec, f1, cm, fps, fns)
    
def create_modeling_notebook(acc, auc, prec, rec, f1, cm, fps, fns):
    import nbformat as nbf
    
    fp_text_sample = ""
    for idx, (_, row) in enumerate(fps.head(5).iterrows()):
        fp_text_sample += f"\n**FP #{idx+1} (prob={row['predicted_prob']:.2f}, ppl={row['ppl']:.1f}, burst={row['burstiness']:.1f})**:\n> {row['text'][:200]}...\n"
        
    fn_text_sample = ""
    for idx, (_, row) in enumerate(fns.head(5).iterrows()):
        fn_text_sample += f"\n**FN #{idx+1} (prob={row['predicted_prob']:.2f}, ppl={row['ppl']:.1f}, burst={row['burstiness']:.1f})**:\n> {row['text'][:200]}...\n"

    error_analysis_text = f"""## 02 - Model Evaluation and Error Analysis

### 1. Test Set Metrics
- **Accuracy**: {acc*100:.2f}%
- **ROC-AUC**: {auc:.4f}
- **Precision**: {prec:.4f}
- **Recall**: {rec:.4f}
- **F1 Score**: {f1:.4f}

### 2. Error Analysis (Patterns & Failure Modes)
1. **False Positives (Human text flagged as AI, total={len(fps)})**:
   - Occurs when human writing is unusually smooth, formal, or lacks idiosyncratic colloquialisms, resulting in lower GPT-2 perplexity.
   - Short sentences with repetitive structure lower the burstiness metric, making the text statistically mimic model completions.
{fp_text_sample}

2. **False Negatives (AI text flagged as Human, total={len(fns)})**:
   - Occurs when AI generates creative fiction with rare vocabulary, unconventional character names, or dialogue punctuation that raises token surprise.
   - Long, compound sentences with dialogue increase sentence-to-sentence perplexity variance (higher burstiness).
{fn_text_sample}
"""

    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# 02 - Modeling and Test Evaluation\n\nFinal evaluation on the untouched test set and detailed failure mode analysis."),
        nbf.v4.new_code_cell("""import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

artifact = joblib.load('../models/detector.joblib')
model = artifact['model']
test_df = pd.read_csv('../data/processed/features_test.csv')
X_test = test_df[artifact['feature_cols']]
y_test = test_df['label']

probs = model.predict_proba(X_test)[:, 1]
preds = (probs >= 0.5).astype(int)

print(classification_report(y_test, preds, target_names=['Human', 'AI']))"""),
        nbf.v4.new_markdown_cell(error_analysis_text)
    ]
    with open("notebooks/02_modeling.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created notebooks/02_modeling.ipynb")

if __name__ == "__main__":
    main()
