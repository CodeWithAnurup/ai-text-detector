# AI Text Detector - Progress Log

## Phase 0: Setup
- Date: 2026-11-05
- What was done: Initialized git repository, created folder structure (`data/raw`, `data/processed`, `notebooks`, `src`, `models`, `docs`), added `.gitignore`, created `requirements.txt`, created Python virtual environment (`venv`), and successfully installed dependencies.
- Result / metric: `python -c "import torch, transformers, sklearn"` executed with code 0 (All imports successful!).
- Issues and decisions: Used uv with Python 3.13 for fast virtualenv creation and package installation.

## Phase 1: Data
- Date: 2026-11-05
- What was done: Downloaded labeled human-vs-AI text dataset into `data/raw/`. Cleaned nulls, duplicates, and texts with < 30 words. Sampled a balanced 1000 rows (500 human, 500 AI) with random_state=42. Stratified split into train (700 rows, 350/350), val (150 rows, 75/75), and test (150 rows, 75/75).
- Result / metric: 3 CSVs created in `data/` (`train.csv`, `val.csv`, `test.csv`) with zero overlap across splits. Test set isolated.
- Issues and decisions: Verified word lengths (>=30) and strict 50:50 balance across all splits.

## Phase 2: Perplexity feature
- Date: 2026-11-05
- What was done: Implemented `perplexity(text, max_tokens=256)` in `src/features.py` using GPT-2 in eval mode. Handles short text (<10 tokens) safely by returning None.
- Result / metric: Verified on sample texts; returns valid float for standard text and None for <10 tokens.
- Issues and decisions: Set torch.no_grad() and model caching globally for performance.

## Phase 3: Burstiness and extra features
- Date: 2026-11-05
- What was done: Implemented `burstiness` (std of sentence-level perplexities) and cheap features (`num_tokens`, `avg_sent_len`, `ttr`). Ran feature extraction over all three splits with progress tracking. Computed feature medians on train set (`ppl`: 25.89, `burstiness`: 35.24, `num_tokens`: 155.5, `avg_sent_len`: 14.43, `ttr`: 0.55) to impute any missing values without leakage. Cached final feature CSVs in `data/processed/`.
- Result / metric: `features_train.csv` (700, 6), `features_val.csv` (150, 6), `features_test.csv` (150, 6). 0 NaNs across all splits.
- Issues and decisions: Imputed any missing values with train set medians; cached in `data/processed/train_medians.json`.
