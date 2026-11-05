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
- Issues and decisions: Verified word lengths (>=30) and strict 50:50 balance across all splits. In raw data, mapped labels to plan standard: 0 = human, 1 = AI.

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

## Phase 4: EDA and baseline
- Date: 2026-11-05
- What was done: Analyzed feature distributions between Human (0) and AI (1) text. Created `notebooks/01_eda.ipynb` and saved comparative distribution plot to `docs/eda_ppl_burstiness.png`. Computed single-threshold baseline on `train` set and evaluated on `val` set.
- Result / metric:
  - Median Perplexity: Human = 40.90 vs AI = 13.55 (AI is significantly lower, as expected).
  - Median Burstiness: Human = 61.02 vs AI = 14.28 (AI is significantly lower, as expected).
  - Single-threshold baseline (predict AI if ppl < 23.55): Train accuracy = 91.00%, Validation accuracy = 91.33% (Human accuracy: 93.33%, AI accuracy: 89.33%).
- Issues and decisions: Confirmed that raw dataset labels followed alphabetical encoding (0=AI, 1=Human); remapped to project specification 0=Human, 1=AI. All distributions align with theory.

## Phase 5: Train the probability model
- Date: 2026-11-05
- What was done: Built pipelines with StandardScaler for Logistic Regression and Random Forest. Evaluated on validation set. Calibrated chosen model with CalibratedClassifierCV (method='sigmoid', cv=5) and saved artifact to `models/detector.joblib`.
- Result / metric:
  - Logistic Regression (val): ROC-AUC = 0.9957, F1 = 0.9804, Acc = 0.9800
  - Random Forest (val): ROC-AUC = 1.0000, F1 = 0.9933, Acc = 0.9933
  - Selected Logistic Regression (simpler, well-calibrated probabilities).
  - Calibrated Model (val): ROC-AUC = 0.9980, F1 = 0.9740, Acc = 0.9733 (Precision: Human 1.00 / AI 0.95; Recall: Human 0.95 / AI 1.00).
- Issues and decisions: Saved model artifact with metadata and train imputation medians.

## Phase 6: Final evaluation (test set, used once)
- Date: 2026-11-05
- What was done: Evaluated calibrated model once on untouched test set (150 samples: 75 human, 75 AI). Generated calibration curve, metrics summary table, and `notebooks/02_modeling.ipynb`.
- Result / metric:
  - Accuracy: 96.67% (145/150)
  - Precision: 0.9375
  - Recall: 1.0000 (0 false negatives out of 75 AI texts)
  - F1 Score: 0.9677
  - ROC-AUC: 0.9989
  - Confusion Matrix: TN=70, FP=5, FN=0, TP=75
- Error analysis:
  The model correctly detected 100% of AI generated texts without a single false negative.
  The only 5 errors were false positives (human writing classified as AI).
  Inspection reveals these 5 human texts featured unusually short, uniform sentences and highly generic expository phrasing with minimal rare tokens.
  This lowered their perplexity (<20) and sentence-level variance, crossing into the AI prediction threshold.
  Saved calibration curve in `docs/calibration_curve.png` and metrics in `docs/test_metrics.md`.

## Phase 7: Inference function (the v1 deliverable)
- Date: 2026-11-05
- What was done: Implemented `predict_ai_probability(text: str) -> dict` in `src/predict.py`. Loads model once globally, extracts features, fills missing values with train medians, calculates calibrated probability, enforces token-count warnings (<50 tokens), raises ValueError on empty strings, and applies label bands (<0.35 likely_human, 0.35-0.65 uncertain, >0.65 likely_ai).
- Result / metric: Validated on 5 manual test cases (2 human, 2 AI, 1 short text, plus empty input validation). All behaviors behave as specified.
- Issues and decisions: Maintained clean dictionary contract matching project schema.

## Phase 8: Streamlit demo
- Date: 2026-11-05
- What was done: Built Streamlit application in `app.py` with text area, sample selectors, dynamic progress bar, probability percentages, token counter, classification badge, reliability warning, and legal disclaimer footer.
- Result / metric: Verified clean startup via headless server check on port 8501.
- Issues and decisions: Added sample text dropdown for instant user testing.
