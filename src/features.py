"""
Feature extraction module for AI Text Detector (GPT-2 Perplexity & Linguistic stats).
"""

import math
import re
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Global model and tokenizer instances (loaded lazily on first access or explicitly)
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
    if not text or not text.strip():
        return None

    tokenizer, model = get_gpt2()
    
    # Tokenize and truncate to max_tokens
    encodings = tokenizer(
        text.strip(),
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens
    )
    
    input_ids = encodings.input_ids.to(_DEVICE)
    num_tokens = input_ids.shape[1]
    
    # Handle very short text (< 10 tokens)
    if num_tokens < 10:
        return None

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        ppl = torch.exp(loss).item()
        
    if math.isnan(ppl) or math.isinf(ppl):
        return None
        
    return float(ppl)

if __name__ == "__main__":
    # Test on 2 hand-written texts (one robotic, one casual)
    robotic_text = (
        "In contemporary society, artificial intelligence systems demonstrate substantial efficacy "
        "across multifaceted operational domains. Consequently, systematic computational paradigms "
        "facilitate optimized strategic deployment and structured algorithmic execution."
    )
    casual_text = (
        "Hey so yesterday I was grabbing some iced coffee at that tiny spot down the street, "
        "and this dog literally jumped up on my table and tried to eat my cinnamon roll! It was hilarious."
    )
    short_text = "Hello there friend."

    print("Loading model and testing perplexity...")
    ppl_robotic = perplexity(robotic_text)
    ppl_casual = perplexity(casual_text)
    ppl_short = perplexity(short_text)

    print(f"Robotic text perplexity: {ppl_robotic:.2f}" if ppl_robotic is not None else "None")
    print(f"Casual text perplexity: {ppl_casual:.2f}" if ppl_casual is not None else "None")
    print(f"Short text (<10 tokens) perplexity: {ppl_short}")
    
    assert isinstance(ppl_robotic, float), "Robotic text must return float perplexity"
    assert isinstance(ppl_casual, float), "Casual text must return float perplexity"
    assert ppl_short is None, "Short text (<10 tokens) must return None"
    print("Phase 2 test checks passed successfully!")
