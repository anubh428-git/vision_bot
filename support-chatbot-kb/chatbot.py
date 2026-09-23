"""
Multilingual retrieval-augmented chatbot.

Pipeline per turn:
  1. Identify language(s) of the incoming message (whole-message +
     per-segment, to catch mixed-language/code-switched input).
  2. If detection is unreliable (very short/ambiguous text), fall back to
     the language the user was most recently, reliably, speaking in this
     session -- this is what lets "sí" or "¿y el envío?" resolve correctly
     as a follow-up rather than as a fresh, under-specified guess.
  3. Translate the message into the knowledge base's pivot language
     (English) for retrieval. This makes retrieval language-agnostic: the
     KB only needs to exist in one language, and any supported language can
     query it.
  4. Retrieve top-k chunks from the vector KB (unchanged from the
     single-language version).
  5. Generate an answer:
       - with ANTHROPIC_API_KEY set: ask the model to answer in the user's
         detected language, grounded only in the retrieved context, using
         the glossary's canonical terminology so answers stay consistent
         regardless of which language they're delivered in.
       - otherwise: take the best-matching chunk and machine-translate it
         into the user's language (extractive fallback, same as the
         single-language version, now language-aware).
  6. Record the turn (text + detected language) in the session's
     conversation memory for continuity on the next turn.
"""
import os
import uuid

from knowledge_base import KnowledgeBase
import language
import translation
import conversation

kb = KnowledgeBase()

SYSTEM_PROMPT = (
    "You are a helpful, multilingual customer support assistant. Answer the "
    "user's question using ONLY the provided context. If the context doesn't "
    "contain the answer, say so and suggest contacting support. "
    "Respond in {target_language}, regardless of what language the context or "
    "conversation history is in. Keep terminology consistent with how it was "
    "used earlier in the conversation. Be concise."
)


def _build_context(hits):
    return "\n\n---\n\n".join(
        f"[source: {h['metadata']['source_id']}]\n{h['text']}" for h in hits
    )


def _build_history_block(session_id):
    pairs = conversation.recent_context_pairs(session_id)
    if not pairs:
        return ""
    lines = [f"{t['role']}: {t['text']}" for t in pairs]
    return "Recent conversation (for context only, may span multiple languages):\n" + "\n".join(lines)


def _extractive_answer(hits, target_lang):
    if not hits:
        text = (
            "I don't have information about that in my knowledge base yet. "
            "It may not have been added, or the relevant source hasn't been "
            "indexed. I'd recommend contacting our support team directly."
        )
    else:
        best = hits[0]
        snippet = best["text"].strip()
        if len(snippet) > 600:
            snippet = snippet[:600].rsplit(" ", 1)[0] + "..."
        src = best["metadata"]["source_id"]
        text = f"Based on our knowledge base ({src}):\n\n{snippet}"
    if target_lang != language.PIVOT_LANGUAGE:
        text = translation.translate(text, language.PIVOT_LANGUAGE, target_lang)
    return text


def _llm_answer(question_pivot: str, hits, target_lang: str, session_id: str):
    """Generate a synthesized answer via an LLM. Tries ANTHROPIC_API_KEY
    (direct Anthropic API) first, then OPENROUTER_API_KEY (OpenRouter's
    OpenAI-compatible endpoint, which can route to Claude, GPT, Llama, etc.
    depending on OPENROUTER_MODEL)."""
    import requests

    context = _build_context(hits)
    history_block = _build_history_block(session_id)
    target_name = language.SUPPORTED_LANGUAGES.get(target_lang, target_lang)
    system = SYSTEM_PROMPT.format(target_language=target_name)
    user_content = f"{history_block}\n\nContext:\n{context}\n\nQuestion (translated to English for retrieval): {question_pivot}"

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")

    if anthropic_key:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": anthropic_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 500,
                "system": system,
                "messages": [{"role": "user", "content": user_content}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return "".join(b.get("text", "") for b in data.get("content", []))

    elif openrouter_key:
        model = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.6")
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 500,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    raise RuntimeError("No LLM API key configured (ANTHROPIC_API_KEY or OPENROUTER_API_KEY)")


def answer(question: str, session_id: str = None, top_k: int = 4) -> dict:
    session_id = session_id or uuid.uuid4().hex

    # 1-2: identify language, resolving ambiguity via conversation history
    lang_info = language.detect_message_languages(question)
    detected_lang = lang_info["dominant_lang"]
    if not lang_info["dominant_reliable"]:
        detected_lang = conversation.last_user_language(session_id, default=detected_lang)

    # 3: translate to pivot language for retrieval (no-op if already pivot)
    pivot_query = translation.translate(question, detected_lang, language.PIVOT_LANGUAGE)

    # 4: retrieve
    hits = kb.query(pivot_query, top_k=top_k)

    # 5: generate
    use_llm = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENROUTER_API_KEY"))
    try:
        if use_llm:
            text = _llm_answer(pivot_query, hits, detected_lang, session_id)
        else:
            text = _extractive_answer(hits, detected_lang)
    except Exception as e:
        text = _extractive_answer(hits, detected_lang)
        text += f"\n\n(Note: LLM generation failed, showing retrieved excerpt instead: {e})"

    # 6: remember this turn
    conversation.add_turn(session_id, "user", question, lang=detected_lang)
    conversation.add_turn(session_id, "assistant", text, lang=detected_lang)

    return {
        "answer": text,
        "session_id": session_id,
        "detected_language": detected_lang,
        "detected_language_name": language.SUPPORTED_LANGUAGES.get(detected_lang, detected_lang),
        "language_confidence": max((s["confidence"] for s in lang_info["segments"]), default=0.0),
        "is_mixed_language": lang_info["is_mixed"],
        "languages_detected": lang_info["languages_detected"],
        "translation_backend": translation.backend_name(),
        "pivot_query": pivot_query,
        "sources": sorted({h["metadata"]["source_id"] for h in hits}),
        "hits": hits,
    }
