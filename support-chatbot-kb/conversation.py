"""
Lightweight in-memory conversation store, keyed by session_id.

This is what lets the chatbot preserve context, intent, and continuity when
a user switches languages mid-conversation: each turn records what language
it was in, so a short/ambiguous follow-up (e.g. "sí", "and international
orders?") can (a) resolve its own language from the recent conversation
instead of guessing from a 1-2 word string, and (b) have retrieval informed
by what the conversation has been about, not just the current message in
isolation.

In-memory + per-process, which is fine for a demo/single-instance deploy.
For multi-worker production, back this with Redis (same interface).
"""
import time
import threading
from collections import defaultdict

_sessions = defaultdict(list)
_lock = threading.Lock()
MAX_TURNS = 12  # per session, keeps prompts/memory bounded


def add_turn(session_id: str, role: str, text: str, lang: str = None, sources=None):
    with _lock:
        _sessions[session_id].append(
            {
                "role": role,  # "user" | "assistant"
                "text": text,
                "lang": lang,
                "sources": sources or [],
                "ts": time.time(),
            }
        )
        _sessions[session_id] = _sessions[session_id][-MAX_TURNS:]


def get_history(session_id: str) -> list:
    with _lock:
        return list(_sessions.get(session_id, []))


def last_user_language(session_id: str, default: str) -> str:
    """Used when the current message's language can't be reliably detected
    (very short input, ambiguous script overlap, etc.) -- fall back to
    whatever language the user was most recently, reliably, speaking."""
    for turn in reversed(get_history(session_id)):
        if turn["role"] == "user" and turn.get("lang"):
            return turn["lang"]
    return default


def recent_context_pairs(session_id: str, max_pairs: int = 3):
    """Return the last few (user, assistant) turn texts for prompt context."""
    history = get_history(session_id)
    return history[-(max_pairs * 2):]


def clear_session(session_id: str):
    with _lock:
        _sessions.pop(session_id, None)
