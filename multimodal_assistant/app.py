"""
app.py
------
Streamlit front-end for the multi-modal AI assistant. Alongside the chat
itself, each assistant turn exposes an inspectable reasoning trace (ambiguity
assessment, plan, evidence gathered, validation result) so the multi-stage
decision-making is visible rather than a black box.

Run:
    streamlit run app.py
"""
import os
import sys

import streamlit as st
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.orchestrator import MultimodalAssistant


st.set_page_config(page_title="Multi-Modal AI Assistant", page_icon="🖼️", layout="wide")


def get_assistant() -> MultimodalAssistant:
    if "assistant" not in st.session_state:
        st.session_state.assistant = MultimodalAssistant(
            vision_backend=config.VISION_BACKEND,
            llm_backend_name=config.LLM_BACKEND,
        )
        st.session_state.display_log = []  # [(role, text, response_obj_or_None, images)]
    return st.session_state.assistant


with st.sidebar:
    st.title("🖼️ Multi-Modal Assistant")
    st.caption("Text + image understanding with contextual reasoning, ambiguity handling, and response validation.")

    st.markdown("---")
    st.subheader("Backends")
    vision_choice = st.selectbox(
        "Vision backend", ["hf", "mock"],
        index=0 if config.VISION_BACKEND == "hf" else 1,
        help="hf = real BLIP/CLIP/OCR models (needs model downloads). mock = deterministic offline stand-in for testing the reasoning pipeline.",
    )
    llm_choice = st.selectbox(
        "LLM backend", ["ollama", "hf", "mock"],
        index=["ollama", "hf", "mock"].index(config.LLM_BACKEND) if config.LLM_BACKEND in ("ollama", "hf", "mock") else 0,
        help="ollama = local Ollama server (use a vision-capable model like `llava` to let the LLM see the raw image too).",
    )

    if st.button("Apply backend settings (resets conversation)"):
        st.session_state.assistant = MultimodalAssistant(vision_backend=vision_choice, llm_backend_name=llm_choice)
        st.session_state.display_log = []
        st.rerun()

    st.markdown("---")
    show_trace = st.checkbox("Show reasoning trace per response", value=True)

    if st.button("🗑️ Reset conversation"):
        if "assistant" in st.session_state:
            st.session_state.assistant.reset()
            st.session_state.display_log = []
        st.rerun()

    st.markdown("---")
    st.caption(
        "Pipeline: context tracking → ambiguity assessment → planning → "
        "evidence gathering (vision + retrieval) → grounded generation → "
        "response validation (with regeneration-on-failure)."
    )


assistant = get_assistant()

st.header("Chat")

for entry in st.session_state.display_log:
    role, text, resp, imgs = entry
    with st.chat_message("user" if role == "user" else "assistant"):
        if imgs:
            cols = st.columns(min(len(imgs), 4))
            for i, im in enumerate(imgs):
                cols[i % len(cols)].image(im, use_container_width=True)
        st.markdown(text)

        if role == "assistant" and resp is not None and show_trace:
            badge = {
                "answer": "✅ Grounded answer",
                "hedged_answer": "⚠️ Hedged answer (validation flagged some claims)",
                "clarification": "❓ Clarifying question",
            }.get(resp.response_type, resp.response_type)
            with st.expander(f"🔍 Reasoning trace — {badge}"):
                st.markdown("**Ambiguity assessment**")
                st.write(
                    f"- Type: `{resp.ambiguity.ambiguity_type.value}`  \n"
                    f"- Confidence (can answer without clarifying): `{resp.ambiguity.confidence:.2f}`  \n"
                    f"- {resp.ambiguity.explanation}"
                )

                if resp.plan:
                    st.markdown("**Plan (decided actions, in order)**")
                    for i, step in enumerate(resp.plan.steps, 1):
                        st.write(f"{i}. **{step.action.value}** — {step.reason}")

                if not resp.evidence.is_empty():
                    st.markdown("**Evidence gathered**")
                    for img_ev in resp.evidence.image_evidence:
                        st.write(f"- 🖼️ `{img_ev.image_id}`: {img_ev.as_evidence_text()}")
                    for t_ev in resp.evidence.text_evidence:
                        preview = t_ev.content if len(t_ev.content) < 300 else t_ev.content[:300] + "..."
                        st.write(f"- 📄 [{t_ev.source}] {preview}")

                if resp.validation:
                    st.markdown("**Response validation**")
                    st.write(
                        f"- Overall grounding score: `{resp.validation.grounding_score:.2f}`  \n"
                        f"- Passed: `{resp.validation.passed}`  \n"
                        f"- Regeneration attempts used: `{resp.regeneration_attempts}`"
                    )
                    if resp.validation.unsupported_claims:
                        st.write("- Unsupported claims flagged:")
                        for c in resp.validation.unsupported_claims:
                            st.write(f"  - {c}")

st.markdown("---")
uploaded_images = st.file_uploader(
    "Attach image(s) (optional)", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True
)
user_text = st.chat_input("Ask about an image, or ask a follow-up...")

if user_text is not None and user_text.strip():
    pil_images = [Image.open(f).convert("RGB") for f in uploaded_images] if uploaded_images else []

    st.session_state.display_log.append(("user", user_text, None, pil_images))

    with st.spinner("Reasoning through the request..."):
        response = assistant.process_turn(user_text, images=pil_images)

    st.session_state.display_log.append(("assistant", response.text, response, []))
    st.rerun()
