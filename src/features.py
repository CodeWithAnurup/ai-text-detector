"""
Feature extraction module for AI Text Detector (GPT-2 Perplexity, Burstiness, & Linguistic stats).
"""

import math
import os
import re
import numpy as np
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Optimize CPU threads for inference speed
if hasattr(os, "cpu_count") and os.cpu_count():
    torch.set_num_threads(min(6, os.cpu_count()))

_TOKENIZER = None
_MODEL = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_gpt2():
    """Load GPT-2 tokenizer and model once and set to eval mode."""
    global _TOKENIZER, _MODEL
    if _TOKENIZER is None or _MODEL is None:
        model_name = "gpt2"
        _TOKENIZER = GPT2TokenizerFast.from_pretrained(model_name)
        _MODEL = GPT2LMHeadModel.from_pretrained(model_name)
        _MODEL.to(_DEVICE)
        _MODEL.eval()
    return _TOKENIZER, _MODEL

def perplexity(text: str, max_tokens: int = 256) -> float | None:
    """
    Compute GPT-2 perplexity for a given text.
    
    Args:
        text: Input string.
        max_tokens: Maximum tokens to evaluate.
        
    Returns:
        Perplexity as a float, or None if text is empty/under 10 tokens.
    """
    if not text or not str(text).strip():
        return None

    tokenizer, model = get_gpt2()
    
    encodings = tokenizer(
        str(text).strip(),
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens
    )
    
    input_ids = encodings.input_ids.to(_DEVICE)
    num_tokens = input_ids.shape[1]
    
    if num_tokens < 10:
        return None

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        ppl = torch.exp(loss).item()
        
    if math.isnan(ppl) or math.isinf(ppl):
        return None
        
    return float(ppl)

def split_sentences(text: str) -> list[str]:
    """Split text into sentences with at least 4 words (>3 words)."""
    raw_sentences = re.split(r'(?<=[.!?])\s+', str(text).strip())
    valid_sentences = []
    for s in raw_sentences:
        clean = s.strip()
        words = clean.split()
        if len(words) > 3:
            valid_sentences.append(clean)
    return valid_sentences

def burstiness(text: str, max_sentences: int = 8) -> float:
    """
    Compute burstiness (standard deviation of sentence-level perplexities).
    Limits to max_sentences for computational efficiency on CPU.
    """
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return 0.0

    eval_sentences = sentences[:max_sentences]
    sent_ppls = []
    for s in eval_sentences:
        p = perplexity(s, max_tokens=128)
        if p is not None:
            sent_ppls.append(p)
            
    if len(sent_ppls) < 2:
        return 0.0
        
    return float(np.std(sent_ppls))

def extract_features(text: str) -> dict:
    """
    Extract all features from text:
    - ppl: overall GPT-2 perplexity (truncated to 256 tokens)
    - burstiness: standard deviation of sentence perplexities
    - num_tokens: token count using GPT-2 tokenizer
    - avg_sent_len: average sentence length in words
    - ttr: type-token ratio (unique words / total words)
    """
    if not text or not str(text).strip():
        return {
            "ppl": np.nan,
            "burstiness": 0.0,
            "num_tokens": 0,
            "avg_sent_len": 0.0,
            "ttr": 0.0
        }
        
    tokenizer, _ = get_gpt2()
    
    # 1. Words and token stats
    words = re.findall(r'\b\w+\b', str(text).lower())
    total_words = len(words)
    unique_words = len(set(words))
    ttr = (unique_words / total_words) if total_words > 0 else 0.0
    
    tokens = tokenizer.encode(str(text).strip(), truncation=True, max_length=1024)
    num_tokens = len(tokens)
    
    # 2. Sentence lengths
    sentences = split_sentences(text)
    if sentences:
        avg_sent_len = float(np.mean([len(s.split()) for s in sentences]))
    else:
        avg_sent_len = float(total_words) if total_words > 0 else 0.0
        
    # 3. Overall Perplexity
    ppl = perplexity(text, max_tokens=256)
    if ppl is None:
        ppl = np.nan
        
    # 4. Burstiness
    b_val = burstiness(text, max_sentences=8)
    
    return {
        "ppl": ppl,
        "burstiness": b_val,
        "num_tokens": num_tokens,
        "avg_sent_len": avg_sent_len,
        "ttr": ttr
    }
