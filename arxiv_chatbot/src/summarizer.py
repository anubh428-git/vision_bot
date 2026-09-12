"""
summarizer.py
-------------
Abstractive summarization of paper abstracts / retrieved passages using an
open-source seq2seq model (default: facebook/bart-large-cnn). Also provides
a fast extractive fallback (TextRank-style) for when a heavy model isn't
available, and a multi-document summarizer that condenses several retrieved
chunks into one synthesis paragraph.
"""
import sys, os
from functools import lru_cache
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


@lru_cache(maxsize=1)
def _get_summarization_pipeline():
    from transformers import pipeline
    return pipeline("summarization", model=config.SUMMARIZATION_MODEL)


def summarize_text(text: str, max_length: int = 130, min_length: int = 30) -> str:
    """Summarize a single passage with the HF summarization pipeline."""
    if not text or len(text.split()) < 40:
        return text  # too short to bother summarizing
    try:
        summarizer = _get_summarization_pipeline()
        # BART has a 1024-token limit; truncate defensively.
        truncated = " ".join(text.split()[:800])
        out = summarizer(truncated, max_length=max_length, min_length=min_length, do_sample=False)
        return out[0]["summary_text"].strip()
    except Exception as e:
        # Graceful fallback if the model isn't downloaded / no internet / OOM.
        return extractive_fallback_summary(text) + f"\n\n_(fallback summary — summarization model unavailable: {e})_"


def extractive_fallback_summary(text: str, n_sentences: int = 3) -> str:
    """
    Cheap, dependency-light extractive summary: score sentences by word
    frequency (à la TextRank/LexRank lite) and return the top N in original order.
    """
    import re
    from collections import Counter

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= n_sentences:
        return text

    words = re.findall(r"[a-zA-Z']+", text.lower())
    stop = {"the", "a", "an", "of", "to", "in", "and", "is", "for", "on", "that", "with",
            "as", "by", "we", "this", "are", "be", "or", "it", "from", "our"}
    freq = Counter(w for w in words if w not in stop)

    scored = []
    for i, sent in enumerate(sentences):
        sent_words = re.findall(r"[a-zA-Z']+", sent.lower())
        score = sum(freq.get(w, 0) for w in sent_words) / (len(sent_words) + 1)
        scored.append((score, i, sent))

    top = sorted(scored, key=lambda x: -x[0])[:n_sentences]
    top_in_order = [s for _, _, s in sorted(top, key=lambda x: x[1])]
    return " ".join(top_in_order)


def summarize_paper(title: str, abstract: str) -> str:
    return summarize_text(f"{title}. {abstract}")


def synthesize_multi_doc(passages: List[str], question: str = "") -> str:
    """
    Condense several retrieved passages into one synthesis paragraph.
    Used to compress RAG context before it's handed to the LLM, and to
    power the 'summarize these papers together' feature.
    """
    joined = " ".join(passages)
    return summarize_text(joined, max_length=180, min_length=60)
