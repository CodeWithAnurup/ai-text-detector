"""AI Text Detector - Streamlit UI.

Drop-in replacement for app.py. Uses the existing contract:
    predict_ai_probability(text) -> {ai_probability, label, tokens, warning}
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import streamlit as st  # noqa: E402

from src.predict import predict_ai_probability  # noqa: E402

st.set_page_config(page_title="AI Text Detector", page_icon="🔎", layout="wide")

# ---------------------------------------------------------------- styling
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Sans:wght@400;500&display=swap');

html, body, .stApp, .stApp p, .stApp label, .stApp span { font-family: 'IBM Plex Sans', sans-serif; }
[data-testid="stIconMaterial"], .material-symbols-rounded, .material-icons { font-family: 'Material Symbols Rounded', 'Material Icons' !important; }
.stApp { background: #EEF1F5; color: #14213D; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 1100px; padding-top: 2.4rem; padding-bottom: 3rem; }

.title { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 2.3rem; line-height: 1.1; margin: 0; color: #14213D; }
.lede { color: #4A5873; font-size: 1.02rem; margin: .5rem 0 1.6rem 0; max-width: 56ch; }

.stTextArea textarea {
  background: #FFFFFF; color: #14213D; border: 1px solid #C9D1DC; border-radius: 10px;
  font-size: 1rem; line-height: 1.6; padding: 14px 16px;
}
.stTextArea textarea:focus { border-color: #14213D; box-shadow: 0 0 0 2px rgba(20,33,61,.15); }

button[kind="primary"], button[data-testid="stBaseButton-primary"] {
  background: #14213D; color: #FFFFFF; border: 0; border-radius: 8px; font-weight: 500; padding: .55rem 1.3rem;
}
button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover { background: #22345F; color: #FFFFFF; }
button[kind="secondary"], button[data-testid="stBaseButton-secondary"] {
  background: transparent; color: #14213D; border: 1px solid #B9C2D0; border-radius: 8px;
}
button[kind="secondary"]:hover, button[data-testid="stBaseButton-secondary"]:hover { border-color: #14213D; color: #14213D; }

.meter-card { background: #FFFFFF; border: 1px solid #D5DBE5; border-radius: 14px; padding: 26px 28px 22px 28px; }
.verdict { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 500; font-size: 1.15rem; color: #14213D; }
.pct { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 5.2rem; line-height: 1; margin: .35rem 0 .2rem 0; }
.pct span { font-size: 2.2rem; margin-left: 2px; }
.sub { color: #4A5873; font-size: .95rem; margin-bottom: 1.6rem; }

.track { position: relative; height: 12px; border-radius: 6px;
  background: linear-gradient(90deg, #1F9D8B 0%, #CBD1DA 50%, #7B2CBF 100%); }
.marker { position: absolute; top: -9px; width: 6px; height: 30px; background: #14213D;
  border: 2px solid #FFFFFF; border-radius: 4px; transform: translateX(-50%); }
.ends { display: flex; justify-content: space-between; margin-top: 12px; color: #4A5873; font-size: .85rem; }
.facts { margin-top: 1.3rem; padding-top: 1rem; border-top: 1px solid #E3E8EF; color: #4A5873; font-size: .9rem; }

.empty-card { background: #FFFFFF; border: 1px dashed #B9C2D0; border-radius: 14px; padding: 34px 28px; color: #4A5873; }
.empty-card b { color: #14213D; font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.15rem; font-weight: 500; }

.note { background: #FFF6E0; border: 1px solid #F0D9A0; color: #6B4E0A; border-radius: 10px; padding: 10px 14px; margin-top: 12px; font-size: .92rem; }

.hist-row { display: flex; align-items: center; gap: 12px; padding: 9px 0; border-bottom: 1px solid #E3E8EF; font-size: .93rem; }
.hist-row:last-child { border-bottom: 0; }
.dot { width: 10px; height: 10px; border-radius: 50%; flex: none; }
.hist-pct { font-weight: 500; width: 3.2rem; flex: none; }
.hist-text { color: #4A5873; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.disclaimer { color: #4A5873; font-size: .88rem; margin-top: 2.2rem; max-width: 80ch; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------- content
AI_SAMPLE = (
    "Artificial intelligence has transformed numerous industries by enabling organizations to "
    "analyze large volumes of data efficiently. Furthermore, it is important to note that "
    "machine learning models can identify patterns that would otherwise remain hidden. "
    "In conclusion, the adoption of artificial intelligence offers significant benefits, "
    "including improved productivity, enhanced decision-making, and greater innovation."
)
HUMAN_SAMPLE = (
    "Honestly I didn't expect the train to be this late, so I just sat on the platform eating "
    "peanuts and watching a dog chase pigeons. A guy next to me started complaining about the "
    "chai, which, fair, it was terrible. Anyway, I got home around midnight and my mom had "
    "kept dinner warm, which made the whole mess worth it."
)

LABELS = {
    "likely_human": ("Reads as human-written", "#1F9D8B"),
    "uncertain": ("Could go either way", "#B7791F"),
    "likely_ai": ("Reads as AI-generated", "#7B2CBF"),
}

# ---------------------------------------------------------------- state
st.session_state.setdefault("text_input", "")
st.session_state.setdefault("result", None)
st.session_state.setdefault("history", [])


def load_sample(text: str) -> None:
    st.session_state.text_input = text


def clear_all() -> None:
    st.session_state.text_input = ""
    st.session_state.result = None


# ---------------------------------------------------------------- header
st.markdown('<p class="title">AI Text Detector</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="lede">Paste a piece of writing and get an estimated chance that an AI wrote it, '
    "based on how predictable its wording is to GPT-2.</p>",
    unsafe_allow_html=True,
)

left, right = st.columns([1.05, 1], gap="large")

# ---------------------------------------------------------------- input
with left:
    st.text_area(
        "Text to check",
        key="text_input",
        height=300,
        placeholder="Paste at least a paragraph. Short snippets give weak signals.",
        label_visibility="collapsed",
    )
    n_words = len(st.session_state.text_input.split())
    st.caption(f"{n_words} words")

    b1, b2, b3, b4 = st.columns([1.1, 1.2, 1.2, 0.8])
    run = b1.button("Check text", type="primary")
    b2.button("Try AI-style text", on_click=load_sample, args=(AI_SAMPLE,))
    b3.button("Try casual text", on_click=load_sample, args=(HUMAN_SAMPLE,))
    b4.button("Clear", on_click=clear_all)

if run:
    text = st.session_state.text_input.strip()
    if not text:
        st.session_state.result = {"error": "Paste some text first, then select Check text."}
    else:
        try:
            with st.spinner("Scoring with GPT-2..."):
                res = predict_ai_probability(text)
            st.session_state.result = res
            st.session_state.history.insert(
                0,
                {
                    "snippet": text[:90].replace("\n", " "),
                    "p": res["ai_probability"],
                    "label": res["label"],
                },
            )
            st.session_state.history = st.session_state.history[:5]
        except ValueError as exc:
            st.session_state.result = {"error": str(exc)}

# ---------------------------------------------------------------- result
with right:
    res = st.session_state.result
    if res is None:
        st.markdown(
            '<div class="empty-card"><b>No result yet</b><br><br>'
            "Add some text on the left, or load a sample, then select Check text.</div>",
            unsafe_allow_html=True,
        )
    elif "error" in res:
        st.error(res["error"])
    else:
        p = float(res["ai_probability"])
        pct = min(99, max(1, round(p * 100)))
        verdict, color = LABELS.get(res["label"], ("Result", "#14213D"))
        html = (
            '<div class="meter-card">'
            f'<div class="verdict">{verdict}</div>'
            f'<div class="pct" style="color:{color}">{pct}<span>%</span></div>'
            '<div class="sub">estimated chance this text was written by an AI</div>'
            f'<div class="track"><div class="marker" style="left:{pct}%"></div></div>'
            '<div class="ends"><span>Human</span><span>AI</span></div>'
            f'<div class="facts">{res["tokens"]} tokens analysed</div>'
            "</div>"
        )
        st.markdown(html, unsafe_allow_html=True)
        if res.get("warning"):
            st.markdown(f'<div class="note">{res["warning"]}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- history
if st.session_state.history:
    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown("**Recent checks**")
    rows = ""
    for h in st.session_state.history:
        c = LABELS.get(h["label"], ("", "#14213D"))[1]
        hp = min(99, max(1, round(h["p"] * 100)))
        rows += (
            '<div class="hist-row">'
            f'<span class="dot" style="background:{c}"></span>'
            f'<span class="hist-pct">{hp}%</span>'
            f'<span class="hist-text">{h["snippet"]}</span>'
            "</div>"
        )
    st.markdown(rows, unsafe_allow_html=True)

# ---------------------------------------------------------------- how it works
with st.expander("How the score is made"):
    st.markdown(
        "- **Perplexity:** GPT-2 reads the text and measures how surprised it is by each word. "
        "AI writing tends to be more predictable, so it scores lower.\n"
        "- **Burstiness:** human writers vary between simple and complex sentences. "
        "AI writing is often more even.\n"
        "- **Probability:** a calibrated classifier turns these signals into a single percentage."
    )

st.markdown(
    '<p class="disclaimer">This is a statistical estimate, not proof. It can be wrong, especially '
    "on short, formal, or non-native English writing, and on text that has been edited or "
    "paraphrased. Do not use it on its own to accuse anyone.</p>",
    unsafe_allow_html=True,
)
