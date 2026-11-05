"""
Phase 4: EDA and Single Threshold Baseline.
Analyzes features on train set, finds optimal perplexity threshold, evaluates on val set,
and saves diagnostic plots to docs/.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    os.makedirs("docs", exist_ok=True)
    os.makedirs("notebooks", exist_ok=True)
    
    train_df = pd.read_csv("data/processed/features_train.csv")
    val_df = pd.read_csv("data/processed/features_val.csv")
    
    print("Train dataset summary by class:")
    print(train_df.groupby("label")[["ppl", "burstiness", "num_tokens", "avg_sent_len", "ttr"]].mean())
    print("\nTrain dataset medians by class:")
    print(train_df.groupby("label")[["ppl", "burstiness", "num_tokens", "avg_sent_len", "ttr"]].median())
    
    # 1. Plot histograms of ppl and burstiness
    plt.figure(figsize=(12, 5))
    
    # Perplexity (log scale due to long tail)
    plt.subplot(1, 2, 1)
    human_ppl = train_df[train_df["label"] == 0]["ppl"]
    ai_ppl = train_df[train_df["label"] == 1]["ppl"]
    
    bins_ppl = np.logspace(np.log10(max(1, train_df["ppl"].min())), np.log10(train_df["ppl"].max()), 30)
    plt.hist(human_ppl, bins=bins_ppl, alpha=0.6, label="Human (0)", color="royalblue", density=True)
    plt.hist(ai_ppl, bins=bins_ppl, alpha=0.6, label="AI (1)", color="crimson", density=True)
    plt.xscale("log")
    plt.title("Perplexity Distribution (Train)", fontsize=12)
    plt.xlabel("Perplexity (log scale)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Burstiness
    plt.subplot(1, 2, 2)
    human_burst = train_df[train_df["label"] == 0]["burstiness"]
    ai_burst = train_df[train_df["label"] == 1]["burstiness"]
    
    max_burst = np.percentile(train_df["burstiness"], 98)
    bins_burst = np.linspace(0, max_burst, 30)
    plt.hist(human_burst, bins=bins_burst, alpha=0.6, label="Human (0)", color="royalblue", density=True)
    plt.hist(ai_burst, bins=bins_burst, alpha=0.6, label="AI (1)", color="crimson", density=True)
    plt.title("Burstiness Distribution (Train)", fontsize=12)
    plt.xlabel("Burstiness (std of sentence ppl)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join("docs", "eda_ppl_burstiness.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved EDA plot to {plot_path}")
    
    # 2. Baseline: pick the best single threshold on ppl using train set
    # AI text typically has lower ppl (predict AI if ppl < threshold)
    thresholds = np.linspace(train_df["ppl"].min(), np.percentile(train_df["ppl"], 95), 1000)
    best_thresh = None
    best_train_acc = 0.0
    
    y_train = train_df["label"].values
    for t in thresholds:
        # predict 1 (AI) if ppl < t
        preds = (train_df["ppl"].values < t).astype(int)
        acc = (preds == y_train).mean()
        if acc > best_train_acc:
            best_train_acc = acc
            best_thresh = t
            
    print(f"Best PPL threshold on train: {best_thresh:.2f} with train accuracy: {best_train_acc*100:.2f}%")
    
    # Evaluate best threshold on validation set
    y_val = val_df["label"].values
    val_preds = (val_df["ppl"].values < best_thresh).astype(int)
    val_acc = (val_preds == y_val).mean()
    val_human_acc = (val_preds[y_val == 0] == 0).mean()
    val_ai_acc = (val_preds[y_val == 1] == 1).mean()
    
    print(f"Validation accuracy with baseline threshold: {val_acc*100:.2f}%")
    print(f"  Human accuracy: {val_human_acc*100:.2f}%")
    print(f"  AI accuracy: {val_ai_acc*100:.2f}%")
    
    # Generate 01_eda.ipynb notebook
    create_eda_notebook(best_thresh, best_train_acc, val_acc)
    
def create_eda_notebook(best_thresh, best_train_acc, val_acc):
    import nbformat as nbf
    
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# 01 - Exploratory Data Analysis & Baseline\n\nAnalyzes perplexity and burstiness across human and AI texts and establishes a single-threshold baseline."),
        nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_df = pd.read_csv('../data/processed/features_train.csv')
val_df = pd.read_csv('../data/processed/features_val.csv')
train_df.head()"""),
        nbf.v4.new_code_cell("""# Summary statistics by class
train_df.groupby('label').agg(['mean', 'median', 'std'])"""),
        nbf.v4.new_code_cell("""# Plot perplexity and burstiness
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(train_df[train_df['label']==0]['ppl'], bins=30, alpha=0.6, label='Human', color='blue', density=True)
plt.hist(train_df[train_df['label']==1]['ppl'], bins=30, alpha=0.6, label='AI', color='red', density=True)
plt.legend()
plt.title('Perplexity Distribution')

plt.subplot(1, 2, 2)
plt.hist(train_df[train_df['label']==0]['burstiness'], bins=30, alpha=0.6, label='Human', color='blue', density=True)
plt.hist(train_df[train_df['label']==1]['burstiness'], bins=30, alpha=0.6, label='AI', color='red', density=True)
plt.legend()
plt.title('Burstiness Distribution')
plt.show()"""),
        nbf.v4.new_markdown_cell(f"### Baseline PPL Threshold Results\n- Best Threshold on Train: **{best_thresh:.2f}**\n- Train Accuracy: **{best_train_acc*100:.2f}%**\n- Val Accuracy: **{val_acc*100:.2f}%**")
    ]
    with open("notebooks/01_eda.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created notebooks/01_eda.ipynb")

if __name__ == "__main__":
    main()
