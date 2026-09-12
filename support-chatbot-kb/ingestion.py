"""
Loads raw content from a configured source (URL / local file / inline text)
and splits it into overlapping chunks suitable for embedding.
"""
import re
import requests
from bs4 import BeautifulSoup

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def fetch_url_text(url: str, timeout: int = 15) -> str:
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "SupportKB-Bot/1.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


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
        if chunk:
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
