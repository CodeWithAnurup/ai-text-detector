"""
Streamlit Web Application for AI Text Detector (Phase 8).
Provides an interactive UI to evaluate text authorship probability.
"""

import sys
import os
import streamlit as st

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.predict import predict_ai_probability

st.set_page_config(
    page_title="AI Text Detector",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 AI Text Detector")
st.markdown(
    "Detect whether a passage of text was authored by an AI or a human using "
    "GPT-2 perplexity, sentence-level burstiness, and calibrated machine learning."
)

# Sample texts for quick testing
SAMPLE_TEXTS = {
    "Select a sample text...": "",
    "AI Generated (Explanatory Essay)": (
        "Artificial intelligence plays a transformative role in modern society by enhancing efficiency, "
        "automating repetitive tasks, and facilitating complex data analysis across healthcare, finance, "
        "and education. Furthermore, the integration of algorithmic systems enables decision-makers to "
        "optimize resource allocation and accelerate scientific discovery in unprecedented ways."
    ),
    "Human Written (Casual & Narrative)": (
        "So last weekend my brother and I tried fixing our old lawnmower that's been sitting in the shed "
        "for like five years. We got covered in grease and spilled half a can of WD-40 before realizing "
        "the spark plug wire was just unplugged the whole time. Classic us honestly, but we had a good laugh "
        "about it over cold beers afterward."
    ),
    "Short Text (<50 tokens)": (
        "The quick brown fox jumps gracefully over the lazy dog resting under the shaded oak tree."
    )
}

selected_sample = st.selectbox("Quick Load Example:", list(SAMPLE_TEXTS.keys()))
default_input = SAMPLE_TEXTS[selected_sample] if selected_sample != "Select a sample text..." else ""

user_text = st.text_area(
    "Paste text to analyze:",
    value=default_input,
    height=200,
    placeholder="Enter at least 50 words for optimal reliability..."
)

col1, col2 = st.columns([1, 4])
with col1:
    analyze_button = st.button("🚀 Analyze", type="primary", use_container_width=True)

if analyze_button:
    if not user_text.strip():
        st.error("Please enter some text before analyzing.")
    else:
        with st.spinner("Analyzing text patterns and calculating perplexity..."):
            try:
                result = predict_ai_probability(user_text)
                prob = result["ai_probability"]
                percentage = int(round(prob * 100))
                label = result["label"]
                tokens = result["tokens"]
                warning = result["warning"]
                
                st.divider()
                st.subheader("Analysis Results")
                
                # Display progress bar
                st.progress(float(prob))
                
                # Probability Metric & Label Banner
                mcol1, mcol2 = st.columns(2)
                with mcol1:
                    st.metric(label="AI Probability", value=f"{percentage}%")
                with mcol2:
                    st.metric(label="Token Count", value=tokens)
                    
                # Formatted Label
                if label == "likely_ai":
                    st.error(f"🔴 **Classification: Likely AI-Generated** (Confidence: {percentage}%)")
                elif label == "uncertain":
                    st.warning(f"🟡 **Classification: Uncertain / Mixed** (Confidence: {percentage}%)")
                else:
                    st.success(f"🟢 **Classification: Likely Human-Written** (Confidence: {100 - percentage}%)")
                    
                # Warning notice
                if warning:
                    st.warning(f"⚠️ **Notice**: {warning}")
                    
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

st.divider()
st.caption(
    "⚖️ *Disclaimer: Probabilistic estimate, not proof. Do not use alone to accuse anyone.*"
)
