"""
rag_engine.py
-------------
Orchestrates the full pipeline for a chat turn:
  1. Retrieve relevant chunks from the vector store (hybrid semantic+keyword).
  2. Optionally compress/summarize the retrieved context if it's long.
  3. Build a grounded prompt (with conversation history for follow-ups).
  4. Call the open-source LLM to generate the answer.
  5. Return the answer plus the source papers used, for citation display.

Also exposes a "summarize this paper" helper used directly by the Search tab.
"""
import sys, os
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.vectorstore import ArxivVectorStore
from src import summarizer, llm_backend


class ConversationMemory:
    """Simple rolling chat history, kept per Streamlit session."""

    def __init__(self):
        self.turns: List[Dict[str, str]] = []

    def add(self, role: str, content: str):
        self.turns.append({"role": role, "content": content})

    def as_list(self) -> List[Dict[str, str]]:
        return self.turns

    def clear(self):
        self.turns = []


class RagEngine:
    def __init__(self, vector_store: ArxivVectorStore):
        self.vs = vector_store

    def retrieve(self, query: str, top_k: int = config.TOP_K_RETRIEVAL, expand_with_history: str = "") -> "pd.DataFrame":
        effective_query = f"{expand_with_history} {query}".strip() if expand_with_history else query
        return self.vs.hybrid_search(effective_query, top_k=top_k)

    def answer(self, question: str, memory: ConversationMemory, top_k: int = config.TOP_K_RETRIEVAL) -> dict:
        """
        Full RAG turn. Uses the last user turn (if any) to help disambiguate
        pronoun-heavy follow-ups ("what about its limitations?") during retrieval.
        """
        last_user_turn = next(
            (t["content"] for t in reversed(memory.as_list()) if t["role"] == "user"), ""
        )
        retrieved = self.retrieve(question, top_k=top_k, expand_with_history=last_user_turn)
        context_chunks = retrieved.to_dict(orient="records")

        answer_text = llm_backend.generate_answer(
            question=question,
            context_chunks=context_chunks,
            history=memory.as_list(),
        )

        memory.add("user", question)
        memory.add("assistant", answer_text)

        sources = self.vs.paper_level_results(retrieved)[
            ["paper_id", "title", "categories", "authors", "update_date", "score"]
        ]
        return {"answer": answer_text, "sources": sources, "retrieved_chunks": retrieved}

    def summarize_paper_by_id(self, paper_id: str) -> str:
        rows = self.vs.doc_store[self.vs.doc_store["paper_id"] == paper_id]
        if rows.empty:
            return "Paper not found in the index."
        title = rows.iloc[0]["title"]
        full_text = " ".join(rows["text"].tolist())
        return summarizer.summarize_text(full_text, max_length=150, min_length=40)

    def search_papers(self, query: str, top_k: int = 10, mode: str = "hybrid"):
        if mode == "semantic":
            hits = self.vs.semantic_search(query, top_k=top_k * 3)
        elif mode == "keyword":
            hits = self.vs.keyword_search(query, top_k=top_k * 3)
        else:
            hits = self.vs.hybrid_search(query, top_k=top_k * 3)
        return self.vs.paper_level_results(hits).head(top_k)
