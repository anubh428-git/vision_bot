"""
Loads raw content from a configured source (URL / local file / inline text)
and splits it into overlapping chunks suitable for embedding.
"""
import re
import requests
from bs4 import BeautifulSoup

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Elements that are structurally part of the page but not "content" --
# navigation, scripts, and (especially on sites like Wikipedia) citation /
# reference apparatus. Left in, these get chunked and indexed right next to
# real content and can easily out-rank it for short queries, since a
# citation-heavy chunk is dense with proper nouns and dates.
_NOISE_SELECTORS = [
    "script", "style", "nav", "footer", "header", "noscript",
    "sup.reference", "ol.references", "div.reflist", "div.refbegin",
    "div.navbox", "div.vertical-navbox", "table.infobox",
    ".mw-editsection", ".mw-cite-backlink", "#catlinks",
    ".hatnote", ".ambox", ".metadata",
]

# After stripping tags, a handful of citation-style lines can still slip
# through as plain text (e.g. "Retrieved June 10, 2025."). Chunks made
# almost entirely of these are worth dropping rather than indexing.
_CITATION_LINE_RE = re.compile(
    r"^(retrieved|isbn|doi:|archived from|\^|\").{0,20}$|"
    r"^(the\s+)?(wall street journal|new york times|reuters|associated press)\b",
    re.IGNORECASE,
)


def fetch_url_text(url: str, timeout: int = 15) -> str:
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "SupportKB-Bot/1.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for selector in _NOISE_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _is_mostly_citation_noise(chunk_text: str, threshold: float = 0.5) -> bool:
    """Defense in depth for content that survives selector stripping (e.g.
    citation text embedded inline rather than in a recognizable container)."""
    lines = [l.strip() for l in chunk_text.split("\n") if l.strip()]
    if not lines:
        return False
    noisy = sum(1 for l in lines if _CITATION_LINE_RE.match(l))
    return (noisy / len(lines)) >= threshold


def read_file_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    idx = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            boundary = text.rfind("\n", start, end)
            if boundary == -1 or boundary <= start + chunk_size * 0.5:
                boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + chunk_size * 0.3:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk and not _is_mostly_citation_noise(chunk):
            chunks.append({"text": chunk, "chunk_index": idx})
            idx += 1
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_source_content(source: dict) -> str:
    s_type = source["type"]
    if s_type == "url":
        return fetch_url_text(source["location"])
    elif s_type == "file":
        return read_file_text(source["location"])
    elif s_type == "text":
        return source["location"]
    else:
        raise ValueError(f"Unknown source type: {s_type}")
