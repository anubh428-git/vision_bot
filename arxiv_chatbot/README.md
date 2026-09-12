# arXiv Domain-Expert Chatbot (Computer Science)

A retrieval-augmented chatbot that answers advanced questions, summarizes papers,
and explains concepts grounded in a Computer-Science subset of the
[Cornell-University/arxiv](https://www.kaggle.com/datasets/Cornell-University/arxiv)
dataset — built with open-source models end to end (no proprietary API required).

## What it does

- **Domain-restricted RAG chat** — retrieves relevant arXiv chunks (hybrid dense +
  keyword search) and grounds an open-source LLM's answer in them, with inline
  citations back to paper titles / arXiv IDs.
- **Follow-up handling** — keeps a rolling conversation history and folds the last
  user turn into retrieval, so "what about its limitations?" resolves correctly.
- **Paper summarization** — one-click abstractive summaries of any indexed paper
  (BART), with an extractive fallback if the summarization model isn't available.
- **Paper search** — semantic, keyword (TF-IDF), or hybrid search over the indexed
  subset.
- **Concept visualization** — a 2D embedding landscape of the corpus (UMAP) colored
  by subfield, plus a keyword co-occurrence "concept graph" for any topic/query.
- **Corpus overview** — quick stats and a browsable table of what's indexed.

## Architecture

```
Kaggle arXiv JSON  ──►  data_pipeline.py   (filter by category, clean, chunk)
                              │
                              ▼
                     vectorstore.py         (sentence-transformers + FAISS, + TF-IDF)
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
             rag_engine.py        visualization.py
        (retrieve → prompt → LLM)   (UMAP scatter, concept graph)
                    │
                    ▼
              llm_backend.py     (Ollama local server OR HF transformers)
                    │
                    ▼
                 app.py           (Streamlit UI: Chat / Search / Viz / Overview)
```

## Setup

```bash
pip install -r requirements.txt
```

### 1. Get the data

The app ships with a small **bundled sample** (`data/sample_arxiv_cs.json`, ~24
papers) so it runs immediately in demo mode. For the real experience:

1. Download `arxiv-metadata-oai-snapshot.json` from
   https://www.kaggle.com/datasets/Cornell-University/arxiv
2. Place it at `data/arxiv-metadata-oai-snapshot.json` (or pass `--json_path`)
3. Build the index:

   ```bash
   python scripts/build_index.py --categories cs.AI cs.CL cs.LG cs.CV --max_papers 50000
   ```

   This filters to your chosen categories, chunks the abstracts, embeds them,
   and writes a FAISS index + doc store under `data/index/`. Adjust
   `--max_papers` / `config.MAX_PAPERS` to fit your RAM.

### 2. Set up the LLM backend

**Option A — Ollama (recommended, easiest on CPU):**
```bash
# install from https://ollama.com, then:
ollama pull mistral
ollama serve
```
The app talks to it at `http://localhost:11434` by default (`config.OLLAMA_HOST`).

**Option B — Hugging Face transformers (needs a GPU for good latency):**
```bash
export ARXIV_BOT_LLM_BACKEND=hf
```
Edit `config.HF_LLM_MODEL` to any open-weight instruct model you have access to
(default `mistralai/Mistral-7B-Instruct-v0.3`).

### 3. Run the app

```bash
streamlit run app.py
```

## Project layout

```
config.py                  # all paths / model names / hyperparameters in one place
app.py                      # Streamlit UI
src/
  data_pipeline.py          # load, filter-by-category, clean, chunk
  vectorstore.py            # embeddings + FAISS + TF-IDF hybrid search
  summarizer.py              # abstractive (BART) + extractive fallback summarization
  llm_backend.py              # Ollama / HF transformers generation backends + prompt building
  rag_engine.py                # retrieval → prompt → generation orchestration + chat memory
  visualization.py              # embedding landscape + concept co-occurrence graph
scripts/
  build_index.py             # offline CLI: raw JSON -> filtered, chunked, embedded index
data/
  sample_arxiv_cs.json       # bundled demo data (24 CS papers)
  make_sample.py              # regenerates the sample file
```

## Notes, limits & things to extend

- The Kaggle metadata dump contains **title + abstract**, not full paper text, so
  retrieval/summarization operate at the abstract level. For full-text RAG, fetch
  PDFs (e.g. via the arXiv ID) and extend `data_pipeline.py` to parse and chunk them.
- `MAX_PAPERS` defaults to a laptop-friendly cap — raise it once you've confirmed
  your machine can hold the embeddings in RAM (a `all-MiniLM-L6-v2` embedding is
  384 floats/chunk, so 200k chunks ≈ 300MB of vectors).
- Swap `EMBEDDING_MODEL` / `SUMMARIZATION_MODEL` / `HF_LLM_MODEL` in `config.py`
  for other open-weight models as needed (e.g. a larger `bge` or `e5` embedding
  model for better retrieval quality).
- All generation is grounded via the prompt in `llm_backend.build_prompt` — the
  system prompt instructs the model to cite sources and flag ungrounded claims.
