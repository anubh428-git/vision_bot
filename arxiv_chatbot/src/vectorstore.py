"""
vectorstore.py
--------------
Embeds text chunks with a sentence-transformers model and builds/queries a
FAISS index for fast semantic (dense) retrieval. Also exposes a lightweight
keyword search fallback (TF-IDF) used by the "Paper Search" tab so users can
search by exact terms as well as by meaning.
"""
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ArxivVectorStore:
    def __init__(self, embedding_model_name: str = config.EMBEDDING_MODEL):
        self.embedding_model_name = embedding_model_name
        self._model = None  # lazy-loaded (heavy import / download)
        self.index: faiss.Index = None
        self.doc_store: pd.DataFrame = None
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #
    def build(self, chunk_df: pd.DataFrame, text_col: str = "text", batch_size: int = 64) -> None:
        """Embed every chunk and build a flat inner-product FAISS index (cosine sim on normalized vectors)."""
        texts = chunk_df[text_col].tolist()
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype("float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner product == cosine sim on normalized vectors
        index.add(embeddings)

        self.index = index
        self.doc_store = chunk_df.reset_index(drop=True)

        # Also fit a TF-IDF index over the same chunks for exact keyword search.
        self._tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=50_000)
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, index_path: str = config.FAISS_INDEX_PATH, doc_store_path: str = config.DOC_STORE_PATH) -> None:
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        self.doc_store.to_parquet(doc_store_path, index=False)

    def load(self, index_path: str = config.FAISS_INDEX_PATH, doc_store_path: str = config.DOC_STORE_PATH) -> None:
        self.index = faiss.read_index(index_path)
        self.doc_store = pd.read_parquet(doc_store_path)
        # Rebuild TF-IDF in memory (cheap) since FAISS doesn't store it.
        self._tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=50_000)
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(self.doc_store["text"].tolist())

    def is_ready(self) -> bool:
        return self.index is not None and self.doc_store is not None

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def semantic_search(self, query: str, top_k: int = config.TOP_K_RETRIEVAL) -> pd.DataFrame:
        query_vec = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, idxs = self.index.search(query_vec, top_k)
        results = self.doc_store.iloc[idxs[0]].copy()
        results["score"] = scores[0]
        return results.reset_index(drop=True)

    def keyword_search(self, query: str, top_k: int = config.TOP_K_RETRIEVAL) -> pd.DataFrame:
        query_vec = self._tfidf_vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        top_idx = np.argsort(-sims)[:top_k]
        results = self.doc_store.iloc[top_idx].copy()
        results["score"] = sims[top_idx]
        return results.reset_index(drop=True)

    def hybrid_search(self, query: str, top_k: int = config.TOP_K_RETRIEVAL, alpha: float = 0.6) -> pd.DataFrame:
        """alpha weights semantic score vs keyword score (0=pure keyword, 1=pure semantic)."""
        sem = self.semantic_search(query, top_k=top_k * 3)
        kw = self.keyword_search(query, top_k=top_k * 3)
        merged = pd.concat([sem.assign(sem_score=sem["score"]), kw.assign(kw_score=kw["score"])])
        merged = merged.groupby("chunk_id").agg({
            "paper_id": "first", "title": "first", "categories": "first",
            "authors": "first", "update_date": "first", "text": "first",
            "sem_score": "max", "kw_score": "max",
        }).fillna(0.0)
        merged["score"] = alpha * merged["sem_score"] + (1 - alpha) * merged["kw_score"]
        return merged.sort_values("score", ascending=False).head(top_k).reset_index()

    def paper_level_results(self, chunk_results: pd.DataFrame) -> pd.DataFrame:
        """Collapse chunk-level hits to one row per paper (best-scoring chunk wins), for the Search tab."""
        return (
            chunk_results.sort_values("score", ascending=False)
            .drop_duplicates(subset="paper_id")
            .reset_index(drop=True)
        )
