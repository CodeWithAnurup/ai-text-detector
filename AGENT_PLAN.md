# AI Text Detector: Agent Roadmap (v1)

> **For the AI agent:** Follow phases in order. Do not start a phase until the previous phase's **Done when** checks pass. Make minimal changes, commit after each phase, and log results in `docs/progress.md`.

---

## 1. Goal (v1 only)

Given any input text, return **one number: the probability (0-100%) that it is AI-generated.**

```json
{ "ai_probability": 0.87, "label": "likely_ai", "tokens": 142, "warning": null }
```

### In scope (v1)
- GPT-2 perplexity + burstiness features
- Logistic Regression / Random Forest on those features
- Calibrated probability output
- Python function + simple Streamlit app

### Out of scope (future versions, do NOT build now)
- Sentence-level highlighting
- Multiple scoring models (GPT-2 + Llama etc.)
- Fine-tuned transformer classifier (DeBERTa/RoBERTa)
- API (FastAPI), Docker, deployment
- Multi-language support

---

## 2. Tech stack

| Part | Tool |
|---|---|
| Language | Python 3.10+ |
| Scoring LM | `gpt2` (Hugging Face, free, runs on CPU) |
| ML | scikit-learn |
| Data | pandas, numpy |
| App | Streamlit |
| Env | VS Code + Jupyter, `venv` |

`requirements.txt`: `torch transformers scikit-learn pandas numpy streamlit matplotlib joblib tqdm`

---

## 3. Repo structure

```
ai-text-detector/
├── data/
│   ├── raw/            # downloaded dataset (git-ignored)
│   └── processed/      # features.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── src/
│   ├── features.py     # perplexity, burstiness, text stats
│   ├── train.py        # train + calibrate + save model
│   ├── predict.py      # predict_ai_probability(text)
│   └── utils.py
├── models/             # detector.joblib
├── app.py              # Streamlit demo
├── docs/progress.md
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 4. Phases

### Phase 0: Setup
**Tasks**
1. Create folder, `git init`, venv, install `requirements.txt`.
2. Create the folder structure above, add `.gitignore` (data/raw, models, venv, `__pycache__`).

**Done when**
- `python -c "import torch, transformers, sklearn"` runs without error.

---

### Phase 1: Data
**Tasks**
1. Download a labeled human-vs-AI text dataset from Kaggle (e.g., "LLM - Detect AI Generated Text" or "AI vs Human Text") into `data/raw/`.
2. Keep only two columns: `text`, `label` (0 = human, 1 = AI).
3. Clean: drop nulls and duplicates, drop texts under 30 words.
4. Sample **balanced 1000 rows** (500 human, 500 AI) for v1. Fixed `random_state=42`.
5. **Split first**: 70% train / 15% val / 15% test, stratified. Save as `train.csv`, `val.csv`, `test.csv`. Never touch test until Phase 6.

**Done when**
- Three CSVs exist, classes are balanced in each, no text appears in more than one split.

**Watch out**
- If the dataset has multiple essays per prompt, split by prompt so the same topic doesn't leak across splits.

---

### Phase 2: Perplexity feature
**Tasks**
1. In `src/features.py`, load GPT-2 once (global), set `.eval()`.
2. Implement `perplexity(text, max_tokens=256)`:
   - tokenize, truncate to `max_tokens`
   - `loss = model(ids, labels=ids).loss`
   - return `exp(loss)`
3. Test on 2 hand-written texts (one robotic, one casual).

**Done when**
- Function returns a float for any non-empty string, and handles very short text without crashing (return `None` or `NaN` if under 10 tokens).

---

### Phase 3: Burstiness and extra features
**Tasks**
1. `burstiness(text)`: split into sentences (>3 words), compute perplexity of each, return **std** (and optionally mean).
2. Add cheap features: `num_tokens`, `avg_sentence_len`, `type_token_ratio` (unique words / total words).
3. `extract_features(text) -> dict` returning all features.
4. Run on train/val/test with `tqdm`, save to `data/processed/features_{split}.csv`. Cache results so it's never recomputed.

**Done when**
- Feature CSVs exist with columns: `ppl, burstiness, num_tokens, avg_sent_len, ttr, label`.
- No NaN left (drop or impute with train median, document which).

**Tip:** GPT-2 on CPU is slow. Test with 50 rows first, then run the full set.

---

### Phase 4: EDA and baseline
**Tasks**
1. `01_eda.ipynb`: plot histograms of `ppl` and `burstiness` for human vs AI (log scale for ppl if skewed).
2. Baseline: pick the best **single threshold on `ppl`** using the **train** set, report accuracy on **val**.

**Done when**
- Plots saved to `docs/`, baseline accuracy recorded in `docs/progress.md`.
- Expectation: AI text has lower ppl and lower burstiness. If not, investigate before moving on.

---

### Phase 5: Train the probability model
**Tasks**
1. In `src/train.py`, build a pipeline: `StandardScaler` + `LogisticRegression`.
2. Also try `RandomForestClassifier`. Compare on val with **ROC-AUC and F1**.
3. Pick the better one (prefer Logistic Regression if scores are close, since it's simpler and gives better-behaved probabilities).
4. **Calibrate** with `CalibratedClassifierCV` (method `sigmoid`, `cv=5`) so `predict_proba` reflects real likelihood.
5. Save with `joblib` to `models/detector.joblib`.

**Done when**
- Model file saved, val ROC-AUC reported for both models.

---

### Phase 6: Final evaluation (test set, used once)
**Tasks**
1. Report: accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
2. Plot a **calibration curve** (predicted prob vs actual fraction AI).
3. **Error analysis** in `02_modeling.ipynb`: list 10 false positives (human flagged as AI) and 10 false negatives. Note patterns (formal writing, short text, etc.).

**Done when**
- Metrics table and calibration plot saved in `docs/`.
- Error analysis written up in 5-8 lines.

---

### Phase 7: Inference function (the v1 deliverable)
**Tasks**
1. In `src/predict.py`:

```python
def predict_ai_probability(text: str) -> dict:
    """Return {'ai_probability', 'label', 'tokens', 'warning'}"""
```

2. Rules:
   - Under 50 tokens: still return a score but set `warning = "Text too short, result unreliable"`.
   - Empty or whitespace: raise a clear `ValueError`.
   - Label bands: `<0.35` likely_human, `0.35-0.65` uncertain, `>0.65` likely_ai.
3. Load model once, not per call.

**Done when**
- 5 manual test calls (2 human, 2 AI, 1 short) return sensible output.

---

### Phase 8: Streamlit demo
**Tasks**
1. `app.py`: text area, "Analyze" button, show probability as a progress bar plus percentage and label.
2. Show the warning if present.
3. Add a footer: *"Probabilistic estimate, not proof. Do not use alone to accuse anyone."*

**Done when**
- `streamlit run app.py` works and pasting text shows a result in a few seconds.

---

### Phase 9: README and polish
**README sections**
1. What it does (1 paragraph + screenshot)
2. How it works (perplexity and burstiness in simple words, with the formula)
3. Results table (metrics from Phase 6)
4. How to run
5. **Limitations** (false positives on formal or non-native English writing, memorized text, paraphrased AI text evades detection, short text unreliable, GPT-2 is old)
6. Future work (the out-of-scope list above)

**Done when**
- A stranger can clone, install, and run the app from the README alone.

---

### Phase 10: Publish
1. Push to GitHub (`CodeWithAnurup/ai-text-detector`).
2. Add the repo to resume and LinkedIn with one line: *"Built a perplexity + burstiness based detector that outputs a calibrated probability of AI authorship (ROC-AUC X)."*

---

## 5. Agent working rules

1. **One phase at a time.** Stop and report if a "Done when" check fails.
2. **Minimal fixes.** On errors, fix the smallest thing. Don't rewrite files.
3. **No leakage.** Test set is untouched until Phase 6. Thresholds and model choice use train/val only.
4. **Reproducible.** `random_state=42` everywhere. Pin versions in `requirements.txt`.
5. **Cache expensive work.** Never recompute GPT-2 features if the CSV exists.
6. **Commit after every phase** with message `phase-N: <what was done>`.
7. **Log honestly** in `docs/progress.md`: metrics, problems, decisions. Don't inflate results.
8. **Ask the user** before adding anything from the out-of-scope list.

### progress.md template
```
## Phase N: <name>
- Date:
- What was done:
- Result / metric:
- Issues and decisions:
```

---

## 6. Future roadmap (v2+)

| Version | Add |
|---|---|
| v1.1 | Sentence-level highlighting of "most AI-like" sentences |
| v1.2 | Compare more scoring models (distilgpt2, Llama-based) and ensemble |
| v2.0 | Fine-tuned DeBERTa/RoBERTa classifier, compare with the perplexity approach |
| v2.1 | FastAPI endpoint + Docker |
| v3.0 | Robustness tests (paraphrased AI text, mixed human+AI), fairness check on non-native writing |

---

## 7. Quick dry-run of final output

| Input | ppl | burstiness | Output |
|---|---|---|---|
| Polished, uniform paragraph | 12 | 4 | `ai_probability: 0.91` → likely_ai |
| Casual, varied writing | 55 | 30 | `ai_probability: 0.08` → likely_human |
| 20-word snippet | 30 | n/a | score + warning "too short" |

(Values are illustrative; real numbers come from your trained model.)
