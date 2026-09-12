"""
llm_backend.py
---------------
Generates the final natural-language response from the assembled evidence
bundle + conversation history. Backend-agnostic: works with a local Ollama
server (default -- use a vision-capable model like `llava` if you want the
LLM itself to also see the raw image bytes in addition to the structured
evidence; the structured evidence is passed either way so validation always
has something concrete to check claims against), in-process HF transformers,
or a deterministic "mock" backend for offline pipeline testing.
"""
import sys, os
import base64
from io import BytesIO
from functools import lru_cache
from typing import List, Optional

import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

SYSTEM_PROMPT = (
    "You are a careful multi-modal assistant. You answer using ONLY the "
    "evidence provided below (image analysis results and any retrieved "
    "knowledge) plus the conversation history for continuity. "
    "Do not invent details that aren't supported by the evidence. "
    "If the evidence is insufficient to fully answer, say what you can "
    "confirm and explicitly flag what you're unsure about rather than "
    "guessing. Be concise and specific."
)


def build_prompt(question: str, evidence_block: str, history_block: str, feedback: str = "") -> str:
    feedback_section = f"\n\n### REVISION FEEDBACK (address this in your answer)\n{feedback}" if feedback else ""
    return f"""{SYSTEM_PROMPT}

### EVIDENCE
{evidence_block}

### CONVERSATION HISTORY
{history_block if history_block else "(no prior turns)"}

### CURRENT QUESTION
{question}{feedback_section}

### ANSWER"""


# --------------------------------------------------------------------------- #
# Backend: Ollama
# --------------------------------------------------------------------------- #
def _image_to_b64(image) -> Optional[str]:
    if image is None:
        return None
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _generate_ollama(prompt: str, image=None, model: str = config.OLLAMA_MODEL, host: str = config.OLLAMA_HOST) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}}
    b64 = _image_to_b64(image)
    if b64:
        payload["images"] = [b64]
    try:
        resp = requests.post(f"{host}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return (
            f"⚠️ Could not reach the local Ollama server at {host}. Start it with "
            f"`ollama serve` and pull a model (`ollama pull {model}`, ideally a "
            "vision-capable one like `llava`), or set MM_ASSISTANT_LLM_BACKEND=hf / mock."
        )
    except Exception as e:
        return f"⚠️ LLM generation failed via Ollama backend: {e}"


# --------------------------------------------------------------------------- #
# Backend: HF transformers (text-only reasoning over the structured evidence)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _get_hf_pipeline():
    from transformers import pipeline
    return pipeline("text-generation", model=config.HF_LLM_MODEL, device_map="auto")


def _generate_hf(prompt: str) -> str:
    try:
        gen = _get_hf_pipeline()
        out = gen(prompt, max_new_tokens=400, do_sample=True, temperature=0.2, return_full_text=False)
        return out[0]["generated_text"].strip()
    except Exception as e:
        return f"⚠️ LLM generation failed via HF backend ({config.HF_LLM_MODEL}): {e}"


# --------------------------------------------------------------------------- #
# Backend: mock (deterministic, offline -- summarizes evidence directly)
# --------------------------------------------------------------------------- #
def _generate_mock(question: str, evidence_block: str) -> str:
    return (
        f"[mock backend -- no LLM configured] Based on the gathered evidence:\n{evidence_block}\n"
        f"This would be used to answer: '{question}'."
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def generate(
    question: str,
    evidence_block: str,
    history_block: str,
    image=None,
    feedback: str = "",
    backend: str = config.LLM_BACKEND,
) -> str:
    prompt = build_prompt(question, evidence_block, history_block, feedback=feedback)
    if backend == "ollama":
        return _generate_ollama(prompt, image=image)
    elif backend == "hf":
        return _generate_hf(prompt)
    elif backend == "mock":
        return _generate_mock(question, evidence_block)
    else:
        raise ValueError(f"Unknown LLM_BACKEND '{backend}'. Use 'ollama', 'hf', or 'mock'.")
