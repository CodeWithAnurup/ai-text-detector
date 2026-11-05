"""
Inference module for AI Text Detector (Phase 7).
Provides predict_ai_probability(text) with calibrated probability and label categorization.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import extract_features, get_gpt2

_MODEL_ARTIFACT = None

def get_model():
    """Load model artifact once globally."""
    global _MODEL_ARTIFACT
    if _MODEL_ARTIFACT is None:
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "detector.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please train model first.")
        _MODEL_ARTIFACT = joblib.load(model_path)
    return _MODEL_ARTIFACT

def predict_ai_probability(text: str) -> dict:
    """
    Given any input text, return the probability (0.0 to 1.0) that it is AI-generated,
    along with label band, token count, and reliability warning.
    
    Args:
        text: Input string to analyze.
        
    Returns:
        dict: {
            "ai_probability": float (0.0 - 1.0),
            "label": "likely_human" | "uncertain" | "likely_ai",
            "tokens": int,
            "warning": str | None
        }
        
    Raises:
        ValueError: If input is empty, None, or whitespace only.
    """
    if text is None or not str(text).strip():
        raise ValueError("Input text cannot be empty or whitespace.")
        
    clean_text = str(text).strip()
    
    # 1. Load model artifact
    artifact = get_model()
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    train_medians = artifact.get("train_medians", {})
    
    # 2. Extract features
    raw_feats = extract_features(clean_text)
    num_tokens = raw_feats.get("num_tokens", 0)
    
    # Check warning condition
    warning = "Text too short, result unreliable" if num_tokens < 50 else None
    
    # 3. Impute any NaNs (e.g. ppl if under 10 tokens) with train medians
    feature_vector = {}
    for col in feature_cols:
        val = raw_feats.get(col, np.nan)
        if val is None or np.isnan(val):
            val = train_medians.get(col, 0.0)
        feature_vector[col] = float(val)
        
    X = pd.DataFrame([feature_vector])[feature_cols]
    
    # 4. Predict calibrated probability
    prob_ai = float(model.predict_proba(X)[0, 1])
    prob_ai_rounded = round(prob_ai, 2)
    
    # 5. Label bands
    # < 0.35: likely_human, 0.35-0.65: uncertain, > 0.65: likely_ai
    if prob_ai < 0.35:
        label = "likely_human"
    elif prob_ai <= 0.65:
        label = "uncertain"
    else:
        label = "likely_ai"
        
    return {
        "ai_probability": prob_ai_rounded,
        "label": label,
        "tokens": num_tokens,
        "warning": warning
    }

if __name__ == "__main__":
    print("Testing predict_ai_probability on 5 manual test cases...")
    
    # 1. Human text 1 (casual, conversational, slang)
    human_text_1 = (
        "So last weekend my brother and I tried fixing our old lawnmower that's been sitting in the shed for like five years. "
        "We got covered in grease and spilled half a can of WD-40 before realizing the spark plug wire was just unplugged the whole time. "
        "Classic us honestly, but we had a good laugh about it over cold beers afterward."
    )
    
    # 2. Human text 2 (creative, descriptive)
    human_text_2 = (
        "The kitchen smelled of burnt toast and yesterday's rain dripping through the cracked skylight. "
        "Grandma sat by the wooden radio, humming some disjointed jazz tune that had no start and no finish, "
        "her crooked fingers drumming steadily on the worn vinyl tabletop."
    )
    
    # 3. AI text 1 (structured, formal explanatory essay)
    ai_text_1 = (
        "Artificial intelligence plays a transformative role in modern society by enhancing efficiency, automating repetitive tasks, "
        "and facilitating complex data analysis across healthcare, finance, and education. Furthermore, the integration of algorithmic systems "
        "enables decision-makers to optimize resource allocation and accelerate scientific discovery in unprecedented ways."
    )
    
    # 4. AI text 2 (typical neutral LLM prompt response)
    ai_text_2 = (
        "In conclusion, renewable energy sources such as solar and wind power offer significant environmental benefits over conventional fossil fuels. "
        "By substantially reducing carbon dioxide emissions and minimizing ecological disruption, these sustainable technologies provide a viable pathway "
        "toward mitigating the catastrophic impacts of global climate change."
    )
    
    # 5. Short text (<50 tokens)
    short_text = "The cat slept quietly on the warm sunlit windowsill all afternoon."
    
    test_cases = [
        ("Human Text 1", human_text_1),
        ("Human Text 2", human_text_2),
        ("AI Text 1", ai_text_1),
        ("AI Text 2", ai_text_2),
        ("Short Text", short_text),
    ]
    
    for name, text in test_cases:
        res = predict_ai_probability(text)
        print(f"\n--- {name} ---")
        print(f"Tokens:         {res['tokens']}")
        print(f"AI Probability: {res['ai_probability']}")
        print(f"Label:          {res['label']}")
        print(f"Warning:        {res['warning']}")
        
    # Test empty string exception
    try:
        predict_ai_probability("   ")
        assert False, "Should have raised ValueError on empty string"
    except ValueError as e:
        print(f"\nEmpty text check passed with expected ValueError: {e}")
        
    print("\nPhase 7 Done when checks passed successfully!")
