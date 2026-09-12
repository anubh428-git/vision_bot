"""
data_pipeline.py
-----------------
Loads the arXiv metadata dump (newline-delimited JSON, same schema as the
Kaggle "Cornell-University/arxiv" dataset), filters it down to a domain
subset (default: Computer Science categories), cleans text, and splits long
abstracts/bodies into overlapping chunks ready for embedding.

The Kaggle file has one JSON object per line with fields including:
    id, title, abstract, categories, authors, update_date, ...

This module works unchanged on either the full ~4GB snapshot or the small
bundled sample in data/sample_arxiv_cs.json.
"""
import json
import re
from typing import Iterator, List, Optional

import pandas as pd
try:
    from tqdm import tqdm
except ImportError:  # tqdm is a nicety, not a hard requirement
    def tqdm(iterable, **kwargs):
        return iterable

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def iter_raw_records(json_path: str) -> Iterator[dict]:
    """Stream records one line at a time so we never load the full multi-GB file into memory."""
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def record_matches_categories(record: dict, categories: List[str]) -> bool:
    cats = record.get("categories", "") or ""
    cat_set = set(cats.split())
    return len(cat_set.intersection(categories)) > 0


def load_domain_subset(
    json_path: str = config.RAW_ARXIV_JSON,
    categories: List[str] = None,
    max_papers: int = config.MAX_PAPERS,
) -> pd.DataFrame:
    """
    Stream the raw arXiv dump, keep only records whose `categories` field
    intersects `categories`, clean fields, and return a DataFrame.
    Falls back to the bundled sample file if the full dump isn't present.
    """
    categories = categories or config.DOMAIN_CATEGORIES

    if not os.path.exists(json_path):
        print(f"[data_pipeline] {json_path} not found -- using bundled sample dataset instead.")
        json_path = config.SAMPLE_ARXIV_JSON

    rows = []
    for record in tqdm(iter_raw_records(json_path), desc="Scanning arXiv records"):
        if not record_matches_categories(record, categories):
            continue
        rows.append({
            "id": record.get("id", ""),
            "title": clean_text(record.get("title", "")),
            "abstract": clean_text(record.get("abstract", "")),
            "categories": record.get("categories", ""),
            "authors": record.get("authors", ""),
            "update_date": record.get("update_date", ""),
        })
        if len(rows) >= max_papers:
            break

    df = pd.DataFrame(rows).drop_duplicates(subset="id").reset_index(drop=True)
    return df


def chunk_text(text: str, chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> List[str]:
    """Word-based sliding-window chunking (simple, fast, model-agnostic)."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if text else []
    chunks = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


def build_chunk_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode each paper's abstract into overlapping chunks (one row per chunk)
    for embedding + retrieval. We chunk the abstract (the full text body isn't
    included in the Kaggle metadata dump -- only title/abstract/metadata are).
    """
    records = []
    for _, row in df.iterrows():
        full_text = f"{row['title']}. {row['abstract']}"
        chunks = chunk_text(full_text)
        for i, chunk in enumerate(chunks):
            records.append({
                "paper_id": row["id"],
                "chunk_id": f"{row['id']}::{i}",
                "title": row["title"],
                "categories": row["categories"],
                "authors": row["authors"],
                "update_date": row["update_date"],
                "text": chunk,
            })
    return pd.DataFrame(records)


if __name__ == "__main__":
    df = load_domain_subset()
    print(f"Loaded {len(df)} papers in categories {config.DOMAIN_CATEGORIES}")
    chunks = build_chunk_table(df)
    print(f"Produced {len(chunks)} text chunks")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    df.to_parquet(config.PROCESSED_PARQUET, index=False)
    print(f"Saved paper table to {config.PROCESSED_PARQUET}")
