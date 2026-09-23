"""
Language identification for incoming chat messages.

Uses langdetect (open-source, offline, no model download / no API calls) for
per-message language ID, and a sentence-level pass to catch mixed-language /
code-switched input (e.g. a message that starts in English and switches to
Spanish mid-sentence).
"""
import re
import json
import os
from langdetect import detect_langs, DetectorFactory

# langdetect's detector is non-deterministic across runs unless seeded.
DetectorFactory.seed = 0

# The base chatbot's knowledge base is in English; these are the additional
# languages this extension adds support for. Add more by extending this dict
# and (optionally) the glossary in glossary.json / an Argos language package.
SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
}
PIVOT_LANGUAGE = "en"  # language the knowledge base / retrieval index is stored in

# Split on sentence terminators AND commas/semicolons. Support-chat messages
# are short, so the finer granularity is worth it: code-switching often
# happens mid-sentence ("Hello, necesito ayuda con mi pedido"), and splitting
# only on periods would miss it entirely.
_SEGMENT_SPLIT_RE = re.compile(r"(?<=[.!?।,;])\s+")

_GLOSSARY_PATH = os.path.join(os.path.dirname(__file__), "glossary.json")


def _load_glossary_keywords():
    """{lang: [known phrase, ...]} used as a lexical cross-check: statistical
    sentence-level language ID (langdetect) will happily label a clause like
    "mi pedido never arrived" as pure English, since English tokens dominate
    the n-gram statistics. Spotting a handful of known other-language words
    inside it catches code-switching that the classifier alone misses."""
    try:
        with open(_GLOSSARY_PATH, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except Exception:
        return {}
    # Words that are spelled identically/near-identically across several
    # supported languages (or collide with common English words) aren't a
    # reliable code-switch signal on their own -- exclude them from this
    # cross-check specifically (they're still used normally for translation).
    _AMBIGUOUS_SHORT_WORDS = {"no", "si", "sí", "oui", "ok"}

    keywords = {}
    for entry in entries:
        for lang, phrase in entry.items():
            if lang == "concept":
                continue
            if phrase.lower() in _AMBIGUOUS_SHORT_WORDS:
                continue
            keywords.setdefault(lang, []).append(phrase.lower())
    return keywords


_GLOSSARY_KEYWORDS = _load_glossary_keywords()


def _contains_phrase(lower_text: str, phrase: str) -> bool:
    """Word-boundary substring match -- plain `phrase in text` would match
    e.g. the glossary keyword "no" inside "know", which is wrong."""
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", lower_text) is not None


def _keyword_languages_in(text: str, exclude_lang: str = None):
    lower = text.lower()
    found = set()
    for lang, phrases in _GLOSSARY_KEYWORDS.items():
        if lang == exclude_lang:
            continue
        if any(_contains_phrase(lower, phrase) for phrase in phrases):
            found.add(lang)
    return found


def split_segments(text: str):
    """Best-effort clause split, used to catch code-switching within one message."""
    segs = [s.strip() for s in _SEGMENT_SPLIT_RE.split(text) if s.strip()]
    return segs or ([text.strip()] if text.strip() else [])


def detect_language(text: str, min_confidence: float = 0.5, fallback: str = PIVOT_LANGUAGE) -> dict:
    """Detect the language of a single string. Short strings (<3 words) are
    inherently unreliable for statistical language ID, so they're flagged as
    `reliable: False` -- callers should fall back to conversation context
    (see conversation.last_user_language) rather than trusting the guess."""
    text = text.strip()
    if not text:
        return {"lang": fallback, "confidence": 0.0, "reliable": False}
    try:
        candidates = detect_langs(text)
        # Pick the highest-ranked candidate that's actually one of our
        # supported languages, rather than defaulting straight to English
        # whenever the #1 guess happens to be an unsupported language (e.g.
        # Italian misdetected on a short, lexically-similar Spanish clause).
        supported = [c for c in candidates if c.lang in SUPPORTED_LANGUAGES]
        chosen = supported[0] if supported else candidates[0]
        lang = chosen.lang if chosen.lang in SUPPORTED_LANGUAGES else fallback
        reliable = (chosen.prob >= min_confidence) and (len(text.split()) >= 3) and bool(supported)
        return {"lang": lang, "confidence": round(chosen.prob, 3), "reliable": reliable}
    except Exception:
        return {"lang": fallback, "confidence": 0.0, "reliable": False}


def detect_message_languages(text: str) -> dict:
    """Segment-level detection so a single mixed-language message is handled
    correctly instead of being forced into one language bucket."""
    # Whole-message detection first: splitting into clauses (needed below to
    # catch code-switching) can artificially depress confidence on a
    # perfectly ordinary single-language sentence, since each fragment on
    # its own is shorter and less statistically informative. Prefer the
    # whole-message read when it's confident; only fall back to the
    # largest-segment heuristic when the message as a whole is ambiguous.
    whole = detect_language(text)
    segments = split_segments(text)
    results = [{"text": seg, **detect_language(seg)} for seg in segments]
    reliable_langs = {r["lang"] for r in results if r["reliable"]}

    if whole["reliable"]:
        dominant_lang = whole["lang"]
        dominant_reliable = True
    else:
        dominant = max(results, key=lambda r: len(r["text"])) if results else None
        dominant_lang = dominant["lang"] if dominant else PIVOT_LANGUAGE
        dominant_reliable = dominant["reliable"] if dominant else False

    # Lexical cross-check for code-switched words the statistical classifier
    # missed (see _keyword_languages_in docstring above).
    keyword_langs = _keyword_languages_in(text, exclude_lang=dominant_lang)

    # The dominant language is always included in the result set, even when
    # its own confidence is low -- it's still the single best guess, and
    # excluding it could leave it missing from `languages_detected` entirely
    # while a minority keyword-matched language remained, which is backwards.
    all_langs = reliable_langs | {dominant_lang} | keyword_langs

    return {
        "segments": results,
        "is_mixed": len(all_langs) > 1,
        "dominant_lang": dominant_lang,
        "dominant_reliable": dominant_reliable,
        "languages_detected": sorted(all_langs),
    }
