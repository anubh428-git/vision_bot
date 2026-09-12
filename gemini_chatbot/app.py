"""
app.py
------
A Gemini-powered chatbot, running locally via Streamlit, with integrated
sentiment analysis: every user message is scored positive/neutral/negative,
the model's tone is dynamically adjusted to respond appropriately, and a
running customer-satisfaction estimate is tracked and shown in the sidebar.

Note: Google's PaLM API was fully decommissioned in August 2024 and can no
longer be installed or called -- it was replaced by the Gemini API, which is
what this app uses (via Google's current `google-genai` SDK; the older
`google-generativeai` package is itself now deprecated too).

Setup:
    pip install -r requirements.txt
    streamlit run app.py

Then paste your Gemini API key in the sidebar (get one free at
https://aistudio.google.com/apikey), or set it in a .env file
(see .env.example).
"""
import os
import sys

import streamlit as st
from PIL import Image
from dotenv import load_dotenv
import google
from google.genai import types
from google.genai import errors

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.sentiment_analysis import SentimentAnalyzer, should_escalate
from src.response_strategy import compose_system_instruction

load_dotenv()

MODEL_OPTIONS = {
    "Gemini Flash (latest, recommended)": "gemini-flash-latest",
    "Gemini 2.5 Flash-Lite (cheapest)": "gemini-2.5-flash-lite",
    "Gemini 3.7 Flash (newest, strongest Flash)": "gemini-3.7-flash",
    "Gemini 3.1 Pro (most capable, slower/pricier)": "gemini-3.1-pro",
}

SENTIMENT_BADGE = {
    "positive": ("🟢", "#2f9e44"),
    "neutral": ("⚪", "#868e96"),
    "negative": ("🔴", "#e03131"),
}

st.set_page_config(page_title="Gemini Chatbot", page_icon="✨", layout="wide")


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
if "display_log" not in st.session_state:
    st.session_state.display_log = []       # [(role, text, [PIL images], SentimentResult_or_None)]
if "chat_session" not in st.session_state:
    st.session_state.chat_session = None
if "chat_model_key" not in st.session_state:
    st.session_state.chat_model_key = None
if "sentiment_history" not in st.session_state:
    st.session_state.sentiment_history = []   # list of SentimentResult, one per user turn


@st.cache_resource(show_spinner=False)
def get_analyzer() -> SentimentAnalyzer:
    return SentimentAnalyzer()


def get_or_create_chat(client: google.genai.Client, model: str):
    """The Chat object keeps conversation history internally; we only need to
    recreate it if the *model* changes (system instruction/temperature are
    now set per-turn instead, since they depend on that turn's sentiment)."""
    if st.session_state.chat_session is None or st.session_state.chat_model_key != model:
        st.session_state.chat_session = client.chats.create(model=model)
        st.session_state.chat_model_key = model
    return st.session_state.chat_session


def render_sentiment_badge(result) -> str:
    emoji, color = SENTIMENT_BADGE[result.label]
    return (
        f'<span style="color:{color}; font-size:0.85em;">{emoji} '
        f'<b>{result.label}</b> ({result.intensity}, compound={result.compound:+.2f}, '
        f'satisfaction={result.satisfaction_score:.0f}/100)</span>'
    )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("✨ Gemini Chatbot")
    st.caption("With integrated sentiment analysis, powered by Google's Gemini API.")

    default_key = os.environ.get("GEMINI_API_KEY", "")
    api_key = st.text_input("Gemini API key", value=default_key, type="password",
                             help="Get one free at https://aistudio.google.com/apikey")

    model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()), index=0)
    model = MODEL_OPTIONS[model_label]

    temperature = st.slider("Temperature", 0.0, 2.0, 1.0, 0.1)
    system_instruction = st.text_area(
        "System instruction (optional)",
        placeholder="e.g. You are a support agent for Acme Inc.",
    )

    st.markdown("---")
    st.subheader("📊 Customer sentiment")
    analyzer = get_analyzer()
    st.caption(f"Detection backend: `{analyzer.backend}`"
               + (" (NLTK VADER)" if analyzer.backend == "vader" else " (offline lexicon fallback)"))

    history = st.session_state.sentiment_history
    if history:
        avg_satisfaction = sum(r.satisfaction_score for r in history) / len(history)
        latest = history[-1]
        col1, col2 = st.columns(2)
        col1.metric("Avg. satisfaction", f"{avg_satisfaction:.0f}/100")
        col2.metric("Latest turn", f"{latest.satisfaction_score:.0f}/100", delta=f"{latest.label}")

        st.line_chart([r.satisfaction_score for r in history], height=150)

        counts = {"positive": 0, "neutral": 0, "negative": 0}
        for r in history:
            counts[r.label] += 1
        st.bar_chart(counts, height=150)

        if should_escalate(history):
            st.error(
                "⚠️ Sustained negative sentiment detected across recent messages. "
                "Consider escalating to a human agent.",
                icon="🚨",
            )
    else:
        st.caption("No messages yet -- sentiment stats will appear here once you start chatting.")

    st.markdown("---")
    if st.button("🗑️ Clear conversation"):
        st.session_state.display_log = []
        st.session_state.chat_session = None
        st.session_state.chat_model_key = None
        st.session_state.sentiment_history = []
        st.rerun()

    st.markdown("---")
    st.caption(
        "This app sends your text and images to Google's Gemini API using the "
        "key above. Nothing is stored anywhere except in this browser session."
    )


# --------------------------------------------------------------------------- #
# Main chat UI
# --------------------------------------------------------------------------- #
st.header("Chat")

for role, text, images, sentiment in st.session_state.display_log:
    with st.chat_message("user" if role == "user" else "assistant"):
        if images:
            cols = st.columns(min(len(images), 4))
            for i, im in enumerate(images):
                cols[i % len(cols)].image(im, use_container_width=True)
        if text:
            st.markdown(text)
        if role == "user" and sentiment is not None:
            st.markdown(render_sentiment_badge(sentiment), unsafe_allow_html=True)

uploaded_images = st.file_uploader(
    "Attach image(s) (optional)", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True
)
user_text = st.chat_input("Ask a question, or ask about the attached image(s)...")

if user_text is not None and user_text.strip():
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar first.")
        st.stop()

    # --- Sentiment analysis on the incoming message ---
    sentiment_result = analyzer.analyze(user_text)
    st.session_state.sentiment_history.append(sentiment_result)
    escalate = should_escalate(st.session_state.sentiment_history)

    pil_images = [Image.open(f).convert("RGB") for f in uploaded_images] if uploaded_images else []
    st.session_state.display_log.append(("user", user_text, pil_images, sentiment_result))

    # --- Build this turn's config: base settings + sentiment-adapted tone ---
    dynamic_system_instruction = compose_system_instruction(system_instruction, sentiment_result, escalate=escalate)
    turn_config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=dynamic_system_instruction,
    )

    client = google.genai.Client(api_key=api_key)
    chat = get_or_create_chat(client, model)
    message_parts = [user_text] + pil_images

    with st.spinner(f"Asking {model_label}..."):
        try:
            response = chat.send_message(message_parts, config=turn_config)
            reply = response.text
        except errors.APIError as e:
            reply = f"⚠️ Gemini API error: {e}"
        except Exception as e:
            reply = f"⚠️ Unexpected error: {e}"

    st.session_state.display_log.append(("assistant", reply, [], None))
    st.rerun()
