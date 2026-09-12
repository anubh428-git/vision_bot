"""
Persistent vector store for the support KB, backed by ChromaDB.

Each ingested source is split into chunks; each chunk is stored with an id of
the form "<source_id>::<chunk_index>" and metadata that records which source
it came from and a content hash (used elsewhere to detect unchanged sources
and skip re-embedding).
"""
import os
import hashlib
import time

import chromadb

from embeddings import HashingEmbeddingFunction

DB_DIR = os.path.join(os.path.dirname(__file__), "data", "chroma")
COLLECTION_NAME = "support_kb"


class KnowledgeBase:
    def __init__(self, persist_dir: str = DB_DIR):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = HashingEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def upsert_chunks(self, source_id: str, chunks: list) -> int:
        """chunks: list of {"text": str, "chunk_index": int}"""
        if not chunks:
            return 0
        ids = [f"{source_id}::{c['chunk_index']}" for c in chunks]
        docs = [c["text"] for c in chunks]
        metadatas = [
            {
                "source_id": source_id,
                "chunk_index": c["chunk_index"],
                "content_hash": self._hash(c["text"]),
                "updated_at": time.time(),
            }
            for c in chunks
        ]
        self.collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
        return len(ids)

    def delete_source(self, source_id: str):
        try:
            self.collection.delete(where={"source_id": source_id})
        except Exception:
            # collection may simply have no chunks for this source yet
            pass

    def get_source_chunk_count(self, source_id: str) -> int:
        res = self.collection.get(where={"source_id": source_id})
        return len(res["ids"])

    def query(self, text: str, top_k: int = 4, source_filter=None):
        where = {"source_id": source_filter} if source_filter else None
        res = self.collection.query(query_texts=[text], n_results=top_k, where=where)
        hits = []
        if res["ids"] and res["ids"][0]:
            for i in range(len(res["ids"][0])):
                hits.append(
                    {
                        "id": res["ids"][0][i],
                        "text": res["documents"][0][i],
                        "metadata": res["metadatas"][0][i],
                        "distance": res["distances"][0][i] if res.get("distances") else None,
                    }
                )
        return hits

    def stats(self):
        return {"total_chunks": self.collection.count()}
