# Support KB Chatbot — Self-Updating Knowledge Base

A customer support chatbot whose knowledge base **grows and refreshes itself
automatically** from sources you point it at (web pages, pasted text, or
local files) — no Streamlit, just a Flask backend + a plain HTML/JS chat UI.

## How it works

```
sources.json  --(scheduler polls every N min, per-source interval)-->
   ingestion.py (fetch + chunk)  -->  embeddings.py (vectorize)  -->
   knowledge_base.py (ChromaDB, persisted on disk)  -->
   chatbot.py (retrieve top-k chunks + answer)  -->  app.py (Flask API + UI)
```

1. **Sources registry** (`config.py` / `sources.json`) — each source (a URL,
   a block of pasted text, or a local file path) has its own
   `update_interval_minutes`.
2. **Scheduler** (`scheduler.py`, via APScheduler) — runs a background job
   every 60 seconds. For each source whose interval has elapsed, it
   re-fetches the content, hashes it, and only re-embeds/re-indexes if the
   content actually changed (so unchanged sources are cheap no-ops). This is
   the "dynamic expansion" mechanism: add a source once, and it stays current
   without any manual re-upload.
3. **Vector store** (`knowledge_base.py`) — a persistent ChromaDB collection.
   Chunks are stored per-source and can be wholesale replaced when a source
   updates, or deleted when a source is removed.
4. **Embeddings** (`embeddings.py`) — a deterministic, fully offline
   hashing/bag-of-words embedder (no API key or model download required), so
   the whole system runs standalone out of the box. Swap in a real embedding
   API for production-grade semantic search (see "Upgrading" below).
5. **Chatbot** (`chatbot.py`) — retrieves the most relevant chunks for a
   question and answers either extractively (default) or via the Anthropic
   API if you set `ANTHROPIC_API_KEY` (for fluent, synthesized answers
   grounded in the retrieved context).
6. **Web app** (`app.py` + `templates/index.html` + `static/`) — a Flask
   server with a JSON API and a plain HTML/CSS/JS chat interface (two-pane:
   chat on the left, source management on the right). No Streamlit.

## Running it

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000

Optional, for LLM-synthesized answers instead of extractive snippets:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python app.py
```

## Using it

- **Add a source** from the sidebar: paste a URL, some text, or a server
  file path, set how often it should refresh (in minutes), and submit. It's
  ingested immediately and also re-checked automatically going forward.
- **Refresh now** / **Refresh all now** force an immediate re-check,
  bypassing the interval.
- **Remove** deletes a source and all of its vectors from the KB.
- Ask questions in the chat pane — answers cite which source(s) they came
  from.

## API

| Method & Path                      | Purpose                                   |
|-------------------------------------|--------------------------------------------|
| `POST /api/chat`                    | `{message}` → RAG answer + cited sources   |
| `GET /api/sources`                  | List sources, chunk counts, KB stats       |
| `POST /api/sources`                 | Add `{type, location, update_interval_minutes}` and ingest immediately |
| `POST /api/sources/<id>/refresh`    | Force re-ingest one source now             |
| `POST /api/sources/refresh-all`     | Force re-ingest every source now           |
| `DELETE /api/sources/<id>`          | Remove a source and its vectors            |

`type` is one of `url`, `text`, or `file`.

## Upgrading for production

- **Embeddings**: replace `HashingEmbeddingFunction` in `embeddings.py` with
  a call to a real embedding model (OpenAI `text-embedding-3-*`, Anthropic-
  compatible providers, or a locally hosted `sentence-transformers` model).
  Nothing else needs to change — `knowledge_base.py` only depends on the
  `__call__` interface.
- **Answer generation**: `chatbot.py` already calls the Anthropic API when
  `ANTHROPIC_API_KEY` is set; adjust the model name / prompt as needed.
- **More source types**: extend `ingestion.load_source_content()` — e.g. add
  a case for PDFs, Notion pages, Zendesk/Confluence exports, or a database
  query, following the existing `url`/`file`/`text` pattern.
- **Scale**: ChromaDB's `PersistentClient` is fine for small/medium KBs; for
  larger deployments point it at a hosted Chroma server, or swap in
  Pinecone/Weaviate/pgvector behind the same `KnowledgeBase` interface.
- **Multi-process deployment**: run the scheduler as a separate worker
  process (e.g. a small script that just imports `scheduler.run_all_due` on
  a cron/systemd timer) rather than inside the Flask process, so it isn't
  tied to web server restarts.

## Notes & limitations

- The bundled embedding function is lexical (word-hash based), not
  semantic — it matches well on shared keywords but won't catch pure
  paraphrases (e.g. "get my money back" vs. "refund policy") without shared
  vocabulary. This is intentional so the whole project runs with zero
  external dependencies out of the box; swap in a real embedding model (see
  above) for better recall on paraphrased questions.
- `type: "file"` reads from the local filesystem the Flask app is running
  on — only point it at trusted local paths.
