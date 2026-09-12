"""
knowledge_retrieval.py
------------------------
Optional tool the planner can invoke when a question needs facts beyond what
the image + conversation provide (e.g. "who painted this style?", "what year
did this event happen?"). Uses a no-API-key web search library by default;
degrades gracefully to "no external evidence" if unavailable/offline so the
rest of the pipeline still works (the validator will then simply not have
external evidence to ground such claims in, and should flag them).
"""
import sys, os
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.schemas import TextEvidence


def retrieve(query: str, max_results: int = 3) -> List[TextEvidence]:
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                snippet = r.get("body") or r.get("snippet") or ""
                title = r.get("title", "")
                if snippet:
                    results.append(TextEvidence(
                        source="external_knowledge",
                        content=f"{title}: {snippet}",
                        relevance=1.0,
                    ))
        return results
    except Exception as e:
        return [TextEvidence(
            source="external_knowledge",
            content=f"(External knowledge retrieval unavailable: {e}. "
                     "Answer should rely only on the image and conversation, "
                     "and should say so if outside facts were requested.)",
            relevance=0.0,
        )]
