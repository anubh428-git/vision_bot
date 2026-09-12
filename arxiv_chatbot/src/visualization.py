"""
visualization.py
-----------------
Two complementary "concept visualization" views for the Streamlit app:

1. embedding_scatter(): projects paper embeddings to 2D (UMAP, falls back to
   PCA/TSNE) and plots them colored by primary arXiv category, so users can
   see how subfields cluster and where a given paper/topic sits relative to
   others.

2. concept_graph(): builds a co-occurrence network of frequent keywords
   extracted (via simple TF-IDF n-gram scoring) from a set of papers, so
   users can see which concepts tend to appear together in the corpus --
   a lightweight "concept map" for a topic or search result set.
"""
import sys, os
from collections import Counter
from itertools import combinations
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _project_2d(embeddings: np.ndarray) -> np.ndarray:
    """Try UMAP first (best for this kind of data), fall back to PCA if unavailable."""
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=min(15, max(2, len(embeddings) - 1)), random_state=42)
        return reducer.fit_transform(embeddings)
    except Exception:
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=42).fit_transform(embeddings)


def embedding_scatter(doc_store: pd.DataFrame, model, sample_n: int = 2000) -> go.Figure:
    """
    doc_store: chunk-level table with 'text', 'title', 'categories', 'paper_id'.
    model: a loaded SentenceTransformer (reused from the vector store to avoid
           re-downloading/re-loading weights).
    """
    papers = doc_store.drop_duplicates(subset="paper_id").reset_index(drop=True)
    if len(papers) > sample_n:
        papers = papers.sample(sample_n, random_state=42).reset_index(drop=True)

    texts = (papers["title"].fillna("") + ". " + papers["text"].fillna("")).tolist()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    coords = _project_2d(np.asarray(embeddings))

    papers = papers.copy()
    papers["x"] = coords[:, 0]
    papers["y"] = coords[:, 1]
    papers["primary_category"] = papers["categories"].fillna("").apply(
        lambda c: c.split()[0] if c else "unknown"
    )

    fig = px.scatter(
        papers, x="x", y="y", color="primary_category",
        hover_data={"title": True, "x": False, "y": False, "primary_category": True},
        title="Paper Embedding Landscape (2D projection, colored by primary category)",
        opacity=0.8,
    )
    fig.update_layout(legend_title_text="Primary category", height=550)
    return fig


def _extract_keywords(texts: List[str], top_n: int = 25) -> List[str]:
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), max_features=2000, min_df=1
    )
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return []
    scores = np.asarray(matrix.sum(axis=0)).flatten()
    vocab = np.array(vectorizer.get_feature_names_out())
    top_idx = np.argsort(-scores)[:top_n]
    return vocab[top_idx].tolist()


def concept_graph(texts: List[str], top_n_keywords: int = 20, co_occurrence_window: bool = True) -> go.Figure:
    """
    Builds a keyword co-occurrence graph across a set of documents (e.g. all
    chunks returned for a search query or topic) and renders it with Plotly.
    Two keywords are linked if they appear in the same document; edge weight
    = number of documents in which they co-occur.
    """
    keywords = _extract_keywords(texts, top_n=top_n_keywords)
    if len(keywords) < 2:
        fig = go.Figure()
        fig.update_layout(title="Not enough text to build a concept graph -- try a broader query.")
        return fig

    co_counts = Counter()
    node_counts = Counter()
    lowered_docs = [t.lower() for t in texts]
    for doc in lowered_docs:
        present = [kw for kw in keywords if kw in doc]
        for kw in present:
            node_counts[kw] += 1
        for a, b in combinations(sorted(set(present)), 2):
            co_counts[(a, b)] += 1

    G = nx.Graph()
    for kw, count in node_counts.items():
        if count > 0:
            G.add_node(kw, weight=count)
    for (a, b), w in co_counts.items():
        if w > 0:
            G.add_edge(a, b, weight=w)

    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(title="Not enough text to build a concept graph -- try a broader query.")
        return fig

    pos = nx.spring_layout(G, seed=42, k=0.8)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=1, color="#aaaaaa"), hoverinfo="none")

    node_x, node_y, node_text, node_size = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_size.append(12 + 6 * G.nodes[node]["weight"])

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=node_text, textposition="top center",
        marker=dict(size=node_size, color="#6c63ff", line=dict(width=1, color="white")),
        hoverinfo="text",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Concept Co-occurrence Graph",
        showlegend=False, height=550,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


def category_distribution(df: pd.DataFrame) -> go.Figure:
    """Simple bar chart of how many papers fall into each primary category -- useful corpus overview."""
    primary = df["categories"].fillna("").apply(lambda c: c.split()[0] if c else "unknown")
    counts = primary.value_counts().reset_index()
    counts.columns = ["category", "count"]
    fig = px.bar(counts, x="category", y="count", title="Papers per Primary Category")
    fig.update_layout(height=400)
    return fig
