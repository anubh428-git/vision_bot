"""
Translation layer used to pivot user queries into the knowledge base's
language (English) for retrieval, and to translate answers back into the
user's language.

Two backends, tried in order:
  1. ArgosTranslator  -- Argos Translate (https://www.argosopentech.com/),
     a fully open-source, offline neural MT engine. This is the intended
     production backend. Install language packages once (see
     `argos_setup.py`); after that it needs no network access at all.
  2. GlossaryTranslator -- a small, dependency-free phrase-substitution
     translator built from glossary.json. It's a safety net so the pipeline
     still runs end-to-end (and is demoable/testable) in environments where
     Argos packages haven't been installed yet -- NOT a replacement for real
     MT in production.

`translate()` always goes through this module's factory so callers never
need to know which backend is active.
"""
import json
import os
import re
import logging

logger = logging.getLogger("translation")

GLOSSARY_PATH = os.path.join(os.path.dirname(__file__), "glossary.json")


class Translator:
    def translate(self, text: str, src: str, tgt: str) -> str:
        raise NotImplementedError


class ArgosTranslator(Translator):
    """Wraps Argos Translate. Raises on init if the library or the needed
    language packages aren't installed, so the factory can fall back."""

    def __init__(self):
        import argostranslate.translate as argos_translate

        self._argos = argos_translate
        installed = self._argos.get_installed_languages()
        if not installed:
            raise RuntimeError("No Argos Translate language packages installed")
        self._installed_by_code = {l.code: l for l in installed}

    def translate(self, text: str, src: str, tgt: str) -> str:
        if src == tgt or not text.strip():
            return text
        src_lang = self._installed_by_code.get(src)
        tgt_lang = self._installed_by_code.get(tgt)
        if not src_lang or not tgt_lang:
            raise RuntimeError(
                f"Argos Translate package for {src}->{tgt} not installed. "
                f"Run `python argos_setup.py` to install language packages."
            )
        translation = src_lang.get_translation(tgt_lang)
        if translation is None:
            raise RuntimeError(f"No direct/pivoted Argos translation path for {src}->{tgt}")
        return translation.translate(text)


class GlossaryTranslator(Translator):
    """Offline, zero-dependency phrase-substitution translator built from
    glossary.json. Looks up the longest matching known phrase and swaps it;
    anything outside the glossary is left untranslated (a safe no-op rather
    than a wrong guess). Good enough to keep retrieval/response pipelines
    demonstrably working without internet access; install Argos Translate
    packages for real free-text translation in production."""

    def __init__(self, path: str = GLOSSARY_PATH):
        with open(path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        self._tables = {}  # {lang: {phrase_lower: {other_lang: phrase}}}
        for entry in entries:
            langs = {k: v for k, v in entry.items() if k != "concept"}
            for src_lang, src_phrase in langs.items():
                table = self._tables.setdefault(src_lang, {})
                table[src_phrase.lower()] = {k: v for k, v in langs.items() if k != src_lang}

    @staticmethod
    def _find_phrase(lower_text: str, phrase: str):
        # word-boundary match -- a naive `phrase in text` would match e.g.
        # the glossary keyword "no" inside the English word "know".
        m = re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", lower_text)
        return m.span() if m else None

    def translate(self, text: str, src: str, tgt: str) -> str:
        if src == tgt or not text.strip():
            return text
        lower = text.lower()
        table = self._tables.get(src, {})
        best_phrase, best_span = None, None
        for phrase, translations in table.items():
            if tgt not in translations:
                continue
            span = self._find_phrase(lower, phrase)
            if span and (best_phrase is None or len(phrase) > len(best_phrase)):
                best_phrase, best_span = phrase, span
        if best_phrase:
            replacement = table[best_phrase][tgt]
            start, end = best_span
            return text[:start] + replacement + text[end:]
        return text  # untranslated fallback rather than a wrong guess


_translator = None


def get_translator() -> Translator:
    global _translator
    if _translator is not None:
        return _translator
    try:
        _translator = ArgosTranslator()
        logger.info("Translation backend: Argos Translate (open-source neural MT).")
    except Exception as e:
        logger.warning("Argos Translate unavailable (%s). Falling back to glossary translator.", e)
        _translator = GlossaryTranslator()
    return _translator


def translate(text: str, src: str, tgt: str) -> str:
    translator = get_translator()
    try:
        return translator.translate(text, src, tgt)
    except Exception as e:
        logger.warning("Translation failed (%s); returning original text unchanged.", e)
        return text


def backend_name() -> str:
    return type(get_translator()).__name__
