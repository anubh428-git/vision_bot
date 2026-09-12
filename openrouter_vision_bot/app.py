"""
app.py
------
A simple text + image analyzer chatbot powered by the OpenRouter API,
running locally via Streamlit.

Setup:
    pip install -r requirements.txt
    streamlit run app.py

Then paste your OpenRouter API key in the sidebar (get one free at
https://openrouter.ai/keys), or set it in a .env file (see .env.example).
"""
import os
import base64
from io import BytesIO

import requests
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# A handful of solid vision-capable models available on OpenRouter.
# (Prices/availability change -- check https://openrouter.ai/models for the
# current list and swap in whatever you like; any "vision"-tagged model works.)
MODEL_OPTIONS = {
    "Gemini 2.0 Flash (fast, cheap)": "google/gemini-2.0-flash-001",
    "GPT-4o mini": "openai/gpt-4o-mini",
    "GPT-4o": "openai/gpt-4o",
    "Claude 3.5 Sonnet": "anthropic/claude-3.5-sonnet",
    "Llama 3.2 11B Vision (free)": "meta-llama/llama-3.2-11b-vision-instruct:free",
    "Qwen2.5 VL 72B": "qwen/qwen2.5-vl-72b-instruct",
}

st.set_page_config(page_title="OpenRouter Vision Bot", page_icon="🤖", layout="wide")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def image_to_data_url(image: Image.Image) -> str:
    """Encode a PIL image as a base64 data URL, the format OpenRouter/OpenAI-style vision APIs expect."""
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def build_user_content(text: str, images: list) -> list:
    """Builds the OpenAI-style multi-part content list: text + zero or more images."""
    content = []
    if text:
        content.append({"type": "text", "text": text})
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(img)}})
    return content


def call_openrouter(api_key: str, model: str, messages: list, temperature: float, max_tokens: int) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional but recommended by OpenRouter for attribution / rate-limit tiers:
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "OpenRouter Vision Bot (Streamlit)",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
    except requests.exceptions.RequestException as e:
        return f"⚠️ Network error reaching OpenRouter: {e}"

    if resp.status_code != 200:
        try:
            err = resp.json().get("error", {}).get("message", resp.text)
        except Exception:
            err = resp.text
        return f"⚠️ OpenRouter API error ({resp.status_code}): {err}"

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return f"⚠️ Unexpected response format from OpenRouter: {data}"


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []       # full OpenAI-format history sent to the API
if "display_log" not in st.session_state:
    st.session_state.display_log = []    # [(role, text, [PIL images])] for rendering


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🤖 OpenRouter Vision Bot")
    st.caption("Text + image analyzer, running locally, powered by OpenRouter.")

    default_key = os.environ.get("OPENROUTER_API_KEY", "")
    api_key = st.text_input("OpenRouter API key", value=default_key, type="password",
                             help="Get one free at https://openrouter.ai/keys")

    model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()), index=0)
    model = MODEL_OPTIONS[model_label]

    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    max_tokens = st.slider("Max response tokens", 128, 4096, 1024, 128)

    st.markdown("---")
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.display_log = []
        st.rerun()

    st.markdown("---")
    st.caption(
        "This app sends your text and images to OpenRouter's API using the "
        "key above. Nothing is stored anywhere except in this browser session."
    )


# --------------------------------------------------------------------------- #
# Main chat UI
# --------------------------------------------------------------------------- #
st.header("Chat")

for role, text, images in st.session_state.display_log:
    with st.chat_message("user" if role == "user" else "assistant"):
        if images:
            cols = st.columns(min(len(images), 4))
            for i, im in enumerate(images):
                cols[i % len(cols)].image(im, use_container_width=True)
        if text:
            st.markdown(text)

uploaded_images = st.file_uploader(
    "Attach image(s) (optional)", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True
)
user_text = st.chat_input("Ask a question, or ask about the attached image(s)...")

if user_text is not None and user_text.strip():
    if not api_key:
        st.error("Please enter your OpenRouter API key in the sidebar first.")
        st.stop()

    pil_images = [Image.open(f).convert("RGB") for f in uploaded_images] if uploaded_images else []

    # Update display log immediately so the user's message appears right away.
    st.session_state.display_log.append(("user", user_text, pil_images))

    # Update the real API message history (full multi-turn context).
    st.session_state.messages.append({
        "role": "user",
        "content": build_user_content(user_text, pil_images),
    })

    with st.spinner(f"Asking {model_label}..."):
        reply = call_openrouter(
            api_key=api_key, model=model, messages=st.session_state.messages,
            temperature=temperature, max_tokens=max_tokens,
        )

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.display_log.append(("assistant", reply, []))

    st.rerun()
