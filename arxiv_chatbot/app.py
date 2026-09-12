"""
app.py
------
Streamlit front-end for the arXiv Domain-Expert Chatbot.

Tabs:
  * Chat                 -- RAG conversation with follow-up support & citations
  * Paper Search          -- semantic / keyword / hybrid search over the indexed subset
  * Concept Visualization -- embedding landscape + keyword co-occurrence graph
  * Corpus Overview       -- basic stats about the indexed domain subset

Run:
    streamlit run app.py
"""
import os
import sys

import streamlit as st
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.vectorstore import ArxivVectorStore
from src.rag_engine import RagEngine, ConversationMemory
from src import summarizer, visualization
from src.data_pipeline import load_domain_subset, build_chunk_table


st.set_page_config(page_title="arXiv Domain-Expert Chatbot", page_icon="🔬", layout="wide")


# --------------------------------------------------------------------------- #
# Cached resource loading
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_vector_store() -> ArxivVectorStore:
    store = ArxivVectorStore()
    if os.path.exists(config.FAISS_INDEX_PATH) and os.path.exists(config.DOC_STORE_PATH):
        store.load()
    else:
        # Build on the fly from whatever data is available (full dump if present,
        # otherwise the bundled sample) so the app works the first time it's run.
        df = load_domain_subset()
        chunk_df = build_chunk_table(df)
        store.build(chunk_df)
        store.save()
    return store


def get_engine() -> RagEngine:
    return RagEngine(get_vector_store())


if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()
if "chat_display" not in st.session_state:
    st.session_state.chat_display = []  # [(role, text, sources_df_or_None)]


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🔬 arXiv Domain Expert")
    st.caption("RAG chatbot over a Computer Science subset of the arXiv corpus.")

    with st.spinner("Loading index..."):
        vs = get_vector_store()
    n_papers = vs.doc_store["paper_id"].nunique()
    n_chunks = len(vs.doc_store)
    st.success(f"Index ready: {n_papers} papers / {n_chunks} chunks")

    if vs.doc_store is not None and n_papers <= 30:
        st.info(
            "Running on the small bundled **sample dataset**. Download the full "
            "Kaggle arXiv dump and run `python scripts/build_index.py` for the "
            "real corpus.",
            icon="ℹ️",
        )

    st.markdown("---")
    st.subheader("Settings")
    top_k = st.slider("Chunks retrieved per query", min_value=2, max_value=10, value=config.TOP_K_RETRIEVAL)
    search_mode = st.selectbox("Search Tab mode", ["hybrid", "semantic", "keyword"], index=0)
    llm_backend_choice = st.selectbox(
        "LLM backend", ["ollama", "hf"],
        index=0 if config.LLM_BACKEND == "ollama" else 1,
        help="ollama = call a local Ollama server (recommended). hf = load an open model in-process via transformers.",
    )
    config.LLM_BACKEND = llm_backend_choice

    st.markdown("---")
    if st.button("🗑️ Clear conversation"):
        st.session_state.memory.clear()
        st.session_state.chat_display = []
        st.rerun()

    st.markdown("---")
    st.caption(
        "Data: [Cornell-University/arxiv](https://www.kaggle.com/datasets/Cornell-University/arxiv) · "
        f"Embeddings: `{config.EMBEDDING_MODEL}` · Summarizer: `{config.SUMMARIZATION_MODEL}`"
    )


engine = get_engine()

tab_chat, tab_search, tab_viz, tab_overview = st.tabs(
    ["💬 Chat", "🔎 Paper Search", "🧠 Concept Visualization", "📊 Corpus Overview"]
)


# --------------------------------------------------------------------------- #
# Tab 1: Chat
# --------------------------------------------------------------------------- #
with tab_chat:
    st.subheader("Ask a question about the field")
    st.caption(
        "Ask about a concept, a technique, or a specific paper. Follow-up questions "
        "(\"what about its limitations?\", \"how does that compare to X?\") use the "
        "conversation so far to stay grounded."
    )

    for role, text, sources in st.session_state.chat_display:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(text)
            if sources is not None and not sources.empty:
                with st.expander("📚 Sources used"):
                    for _, row in sources.iterrows():
                        st.markdown(f"- **{row['title']}** (`arXiv:{row['paper_id']}`, score={row['score']:.2f})")

    question = st.chat_input("e.g. Explain how retrieval-augmented generation reduces hallucination")
    if question:
        st.session_state.chat_display.append(("user", question, None))
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant papers and generating a grounded answer..."):
                result = engine.answer(question, st.session_state.memory, top_k=top_k)
            st.markdown(result["answer"])
            if not result["sources"].empty:
                with st.expander("📚 Sources used"):
                    for _, row in result["sources"].iterrows():
                        st.markdown(f"- **{row['title']}** (`arXiv:{row['paper_id']}`, score={row['score']:.2f})")

        st.session_state.chat_display.append(("assistant", result["answer"], result["sources"]))


# --------------------------------------------------------------------------- #
# Tab 2: Paper Search
# --------------------------------------------------------------------------- #
with tab_search:
    st.subheader("Search the indexed papers")
    query = st.text_input("Search by topic, technique, or keywords", key="search_query")
    n_results = st.slider("Number of results", 3, 20, 8)

    if query:
        results = engine.search_papers(query, top_k=n_results, mode=search_mode)
        st.write(f"**{len(results)} results** (mode: `{search_mode}`)")

        for _, row in results.iterrows():
            with st.container(border=True):
                st.markdown(f"### {row['title']}")
                st.caption(
                    f"arXiv:{row['paper_id']} · {row['categories']} · {row['authors']} · "
                    f"{row['update_date']} · relevance={row['score']:.2f}"
                )
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("📄 Summarize this paper", key=f"sum_{row['paper_id']}"):
                        with st.spinner("Summarizing..."):
                            summary = engine.summarize_paper_by_id(row["paper_id"])
                        st.info(summary)
                with col2:
                    if st.button("💬 Discuss this paper in Chat", key=f"chat_{row['paper_id']}"):
                        prompt = f"Explain the paper \"{row['title']}\" (arXiv:{row['paper_id']}) and its significance."
                        result = engine.answer(prompt, st.session_state.memory, top_k=top_k)
                        st.session_state.chat_display.append(("user", prompt, None))
                        st.session_state.chat_display.append(("assistant", result["answer"], result["sources"]))
                        st.success("Added to Chat tab -- switch tabs to view.")
    else:
        st.caption("Try: \"efficient attention\", \"federated learning privacy\", \"diffusion models\"...")


# --------------------------------------------------------------------------- #
# Tab 3: Concept Visualization
# --------------------------------------------------------------------------- #
with tab_viz:
    st.subheader("Visualize the concept landscape")

    viz_kind = st.radio(
        "View", ["Embedding landscape (all papers)", "Concept co-occurrence graph (topic-specific)"],
        horizontal=True,
    )

    if viz_kind.startswith("Embedding"):
        st.caption("2D projection of paper embeddings, colored by primary arXiv category. Nearby points discuss similar content.")
        with st.spinner("Projecting embeddings..."):
            fig = visualization.embedding_scatter(vs.doc_store, vs.model)
        st.plotly_chart(fig, use_container_width=True)

    else:
        topic_query = st.text_input("Topic / query to build a concept map for", value="transformer attention")
        n_docs = st.slider("Number of chunks to analyze", 10, 100, 40)
        if topic_query:
            hits = vs.hybrid_search(topic_query, top_k=n_docs)
            texts = hits["text"].tolist()
            with st.spinner("Extracting concepts and building graph..."):
                fig = visualization.concept_graph(texts)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Nodes = frequent keywords/phrases in the retrieved passages. "
                "Edges link concepts that co-occur in the same passage; thicker "
                "node size = more frequent."
            )


# --------------------------------------------------------------------------- #
# Tab 4: Corpus Overview
# --------------------------------------------------------------------------- #
with tab_overview:
    st.subheader("What's in the index")
    papers = vs.doc_store.drop_duplicates(subset="paper_id")
    col1, col2, col3 = st.columns(3)
    col1.metric("Papers indexed", papers["paper_id"].nunique())
    col2.metric("Text chunks", len(vs.doc_store))
    date_range = f"{papers['update_date'].min()} → {papers['update_date'].max()}" if len(papers) else "n/a"
    col3.metric("Date range", date_range)

    fig = visualization.category_distribution(papers)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Browse indexed papers"):
        st.dataframe(
            papers[["paper_id", "title", "categories", "authors", "update_date"]],
            use_container_width=True, hide_index=True,
        )
