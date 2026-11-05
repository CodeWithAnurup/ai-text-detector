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
