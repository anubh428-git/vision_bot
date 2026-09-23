"""
Lightweight, fully offline embedding function used to turn text chunks into
fixed-size vectors for the vector store.

Why not call an embeddings API? This project needs to run standalone without
requiring API keys or internet access to a model host. It uses a
hashing-trick + log-scaled term-frequency scheme (a deterministic
bag-of-words vector), which is enough to power keyword/topic-level retrieval
for a support KB. For higher semantic quality in production, swap
`HashingEmbeddingFunction` for an OpenAI / Anthropic / HuggingFace embedding
call -- the rest of the system (knowledge_base.py) only depends on this
class's __call__ interface, so the swap is a one-file change.
"""
import re
import math
from collections import Counter

from chromadb.api.types import EmbeddingFunction

DIM = 512

_word_re = re.compile(r"[a-zA-Z0-9']+")


def _tokenize(text: str):
    return [t.lower() for t in _word_re.findall(text)]


def _hash_token(token: str, dim: int = DIM) -> int:
    # FNV-1a style hash. Deterministic across runs/processes (unlike Python's
    # built-in hash(), which is randomized per-process for strings).
    h = 2166136261
    for ch in token:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h % dim


def embed_text(text: str, dim: int = DIM):
    tokens = _tokenize(text)
    vec = [0.0] * dim
    if not tokens:
        return vec
    counts = Counter(tokens)
    for token, count in counts.items():
        idx = _hash_token(token, dim)
        vec[idx] += 1.0 + math.log(count)
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


class HashingEmbeddingFunction(EmbeddingFunction):
    """Chroma-compatible embedding function: callable on a list of strings,
    returns a list of fixed-size float vectors. Subclasses Chroma's
    EmbeddingFunction protocol so both document embedding and query
    embedding (embed_query, defaults to __call__) work correctly."""

    def __init__(self):
        pass

    def name(self):
        return "hashing-bow-v1"

    def __call__(self, input):
        return [embed_text(t) for t in input]
