"""
Retrieval-augmented answering. Retrieves the most relevant chunks from the
vector DB for a question, then either:
  - calls the Anthropic API to synthesize a grounded answer, if
    ANTHROPIC_API_KEY is set in the environment, or
  - falls back to a plain extractive answer (best matching chunk), so the
    chatbot still works with zero external dependencies / API keys.
"""
import os
from knowledge_base import KnowledgeBase

kb = KnowledgeBase()

SYSTEM_PROMPT = (
    "You are a helpful customer support assistant. Answer the user's question "
    "using ONLY the provided context. If the context doesn't contain the "
    "answer, say you don't have that information yet and suggest contacting "
    "support. Be concise."
)


def _build_context(hits):
    parts = []
    for h in hits:
        parts.append(f"[source: {h['metadata']['source_id']}]\n{h['text']}")
    return "\n\n---\n\n".join(parts)


def _extractive_answer(hits):
    if not hits:
        return (
            "I don't have information about that in my knowledge base yet. "
            "It may not have been added, or the relevant source hasn't been "
            "indexed. I'd recommend contacting our support team directly."
        )
    best = hits[0]
    snippet = best["text"].strip()
    if len(snippet) > 600:
        snippet = snippet[:600].rsplit(" ", 1)[0] + "..."
    src = best["metadata"]["source_id"]
    return f"Based on our knowledge base ({src}):\n\n{snippet}"


def _llm_answer(question: str, hits):
    import requests

    context = _build_context(hits)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 500,
            "system": SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return "".join(b.get("text", "") for b in data.get("content", []))


def answer(question: str, top_k: int = 4) -> dict:
    hits = kb.query(question, top_k=top_k)
    use_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))
    try:
        text = _llm_answer(question, hits) if use_llm else _extractive_answer(hits)
    except Exception as e:
        text = _extractive_answer(hits)
        text += f"\n\n(Note: LLM generation failed, showing retrieved excerpt instead: {e})"
    return {
        "answer": text,
        "sources": sorted({h["metadata"]["source_id"] for h in hits}),
        "hits": hits,
    }
