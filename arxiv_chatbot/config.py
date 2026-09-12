"""
Central configuration for the arXiv Domain-Expert Chatbot.
Edit paths/model names here rather than scattering magic strings through the code.
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(BASE_DIR, "data", "index")

# The full Kaggle dump (arxiv-metadata-oai-snapshot.json). Download from
# https://www.kaggle.com/datasets/Cornell-University/arxiv and place it here,
# OR point RAW_ARXIV_JSON at wherever you saved it.
RAW_ARXIV_JSON = os.path.join(DATA_DIR, "arxiv-metadata-oai-snapshot.json")

# Small bundled sample (~30 CS papers) so the app runs out of the box without
# the 4GB+ Kaggle download. Swap to the real, filtered subset once you've run
# scripts/build_index.py on the full dataset.
SAMPLE_ARXIV_JSON = os.path.join(DATA_DIR, "sample_arxiv_cs.json")

PROCESSED_PARQUET = os.path.join(DATA_DIR, "arxiv_subset.parquet")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
DOC_STORE_PATH = os.path.join(INDEX_DIR, "doc_store.parquet")

# ---------------------------------------------------------------------------
# Domain subset
# ---------------------------------------------------------------------------
# arXiv category codes to keep. Default = core Computer Science categories.
# See https://arxiv.org/category_taxonomy for the full list.
DOMAIN_CATEGORIES = [
    "cs.AI", "cs.CL", "cs.LG", "cs.CV", "cs.NE", "cs.IR",
    "cs.DC", "cs.CR", "cs.SE", "cs.RO",
]

# Cap on number of papers to index (memory/CPU friendly default; raise this
# on a machine with more RAM once you move past the bundled sample).
MAX_PAPERS = 50_000

# ---------------------------------------------------------------------------
# Models (all open-source / open-weight, pulled from Hugging Face)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # fast, 384-dim
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"               # abstractive summarizer

# LLM used for explanation / free-form Q&A generation. Two backends supported:
#   "hf"      -> loads the model in-process via transformers (needs local GPU/RAM)
#   "ollama"  -> calls a local Ollama server (recommended: `ollama pull mistral`)
LLM_BACKEND = os.environ.get("ARXIV_BOT_LLM_BACKEND", "ollama")
HF_LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
OLLAMA_MODEL = os.environ.get("ARXIV_BOT_OLLAMA_MODEL", "mistral")
OLLAMA_HOST = os.environ.get("ARXIV_BOT_OLLAMA_HOST", "http://localhost:11434")

# ---------------------------------------------------------------------------
# Retrieval / chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = 400          # words per chunk when a doc is split for embedding
CHUNK_OVERLAP = 60
TOP_K_RETRIEVAL = 5       # chunks pulled per query for the RAG prompt
MAX_HISTORY_TURNS = 6      # how many prior chat turns to keep in the LLM context
