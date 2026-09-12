"""
llm_backend.py
---------------
Thin wrapper around an open-source LLM used for free-form explanation /
Q&A generation. Two interchangeable backends:

  * "ollama" (default) -- calls a local Ollama server (`ollama serve`,
    `ollama pull mistral`). Zero Python-side model loading, works well on
    CPU-only laptops, easiest way to run a 7B-class open model locally.

  * "hf" -- loads an open-weight instruct model (default Mistral-7B-Instruct)
    in-process via transformers/accelerate. Needs a GPU with enough VRAM
    (or heavy quantization) for reasonable latency.

Switch backends via config.LLM_BACKEND or the ARXIV_BOT_LLM_BACKEND env var.
"""
import sys, os
import json
from functools import lru_cache
from typing import List, Dict

import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


SYSTEM_PROMPT = (
    "You are an expert research assistant specializing in Computer Science, "
    "grounded in a corpus of arXiv papers. Answer the user's question using ONLY "
    "the CONTEXT passages provided below when they are relevant -- cite paper "
    "titles inline like (Title, arXiv:ID) when you use a specific passage. "
    "If the context doesn't fully cover the question, say so and then use your "
    "general knowledge, clearly flagging which parts are not grounded in the "
    "provided context. Explain concepts clearly and precisely, at the level of "
    "a knowledgeable grad student, and answer follow-up questions using the "
    "conversation history for continuity."
)


def _format_context(context_chunks: List[dict]) -> str:
    if not context_chunks:
        return "(no retrieved context)"
    blocks = []
    for c in context_chunks:
        blocks.append(f"[{c.get('title','')} | arXiv:{c.get('paper_id','')}]\n{c.get('text','')}")
    return "\n\n".join(blocks)


def _format_history(history: List[Dict[str, str]], max_turns: int = config.MAX_HISTORY_TURNS) -> str:
    trimmed = history[-max_turns:]
    lines = []
    for turn in trimmed:
        role = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


def build_prompt(question: str, context_chunks: List[dict], history: List[Dict[str, str]]) -> str:
    context_block = _format_context(context_chunks)
    history_block = _format_history(history)
    prompt = f"""{SYSTEM_PROMPT}

### CONTEXT
{context_block}

### CONVERSATION SO FAR
{history_block}

### NEW QUESTION
User: {question}
Assistant:"""
    return prompt


# --------------------------------------------------------------------------- #
# Backend: Ollama (local server, recommended default)
# --------------------------------------------------------------------------- #
def _generate_ollama(prompt: str, model: str = config.OLLAMA_MODEL, host: str = config.OLLAMA_HOST) -> str:
    try:
        resp = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.3}},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return (
            "⚠️ Could not reach the local Ollama server at "
            f"{host}. Start it with `ollama serve` and make sure the model is pulled "
            f"(`ollama pull {model}`), or switch ARXIV_BOT_LLM_BACKEND=hf in config.py."
        )
    except Exception as e:
        return f"⚠️ LLM generation failed via Ollama backend: {e}"


# --------------------------------------------------------------------------- #
# Backend: Hugging Face transformers (in-process)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _get_hf_pipeline():
    from transformers import pipeline
    return pipeline(
        "text-generation",
        model=config.HF_LLM_MODEL,
        device_map="auto",
    )


def _generate_hf(prompt: str) -> str:
    try:
        gen = _get_hf_pipeline()
        out = gen(prompt, max_new_tokens=512, do_sample=True, temperature=0.3,
                   return_full_text=False)
        return out[0]["generated_text"].strip()
    except Exception as e:
        return f"⚠️ LLM generation failed via HF backend ({config.HF_LLM_MODEL}): {e}"


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def generate_answer(
    question: str,
    context_chunks: List[dict],
    history: List[Dict[str, str]],
    backend: str = config.LLM_BACKEND,
) -> str:
    prompt = build_prompt(question, context_chunks, history)
    if backend == "ollama":
        return _generate_ollama(prompt)
    elif backend == "hf":
        return _generate_hf(prompt)
    else:
        raise ValueError(f"Unknown LLM_BACKEND '{backend}'. Use 'ollama' or 'hf'.")
