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

Optional, for LLM-synthesized answers instead of extractive snippets — use
**either** of these (not both need to be set; Anthropic is tried first):

```bash
# Direct Anthropic API
export ANTHROPIC_API_KEY=sk-ant-...

# OR OpenRouter (OpenAI-compatible; can route to Claude, GPT, Llama, etc.)
export OPENROUTER_API_KEY=sk-or-v1-...
export OPENROUTER_MODEL=anthropic/claude-sonnet-4.6   # optional, this is the default

python app.py
```

On Windows PowerShell, use `$env:OPENROUTER_API_KEY = "sk-or-v1-..."` instead
of `export`. This only lasts for the current terminal session; set it again
next time, or add it permanently via System Properties → Environment
Variables.

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

## Multilingual support

The chatbot supports English plus three additional languages out of the
box — **Spanish, French, and Hindi** — and is built to extend to more.

### How it works

```
message --> language.py (detect, incl. mixed/code-switched input)
        --> conversation.py (resolve ambiguity from session history)
        --> translation.py (translate to pivot language: English)
        --> knowledge_base.py (retrieve — KB stays single-language)
        --> chatbot.py (generate answer in the user's language)
        --> conversation.py (record turn for the next one)
```

- **`language.py`** — language identification via `langdetect` (open-source,
  fully offline, no model download). Detects the whole message's language
  first (more reliable on a full sentence), and separately splits the
  message into clauses to catch **code-switching** — a message that
  switches languages mid-sentence (e.g. *"Hola, I need help with a refund,
  mi pedido never arrived"*) is flagged `is_mixed_language: true` with both
  languages listed, using a glossary-based lexical cross-check to catch
  minority-language words that a purely statistical sentence classifier
  would miss.
- **`translation.py`** — pivots any supported language to/from English so
  retrieval only has to work in one language. Two backends:
  - `ArgosTranslator`: real open-source neural MT via
    [Argos Translate](https://www.argosopentech.com/), fully offline once
    language packages are installed (`python argos_setup.py`, run in an
    environment with normal internet access — Argos's model host isn't on
    this sandbox's network allow-list, which is why it isn't installed
    here).
  - `GlossaryTranslator`: an offline, zero-dependency phrase-substitution
    fallback (`glossary.json`) that keeps the whole pipeline demonstrably
    working even with no Argos packages installed. It's a safety net, not a
    production translator — install Argos packages for real free-text
    translation quality.
- **`conversation.py`** — per-session turn memory. This is what resolves
  **ambiguous queries across languages**: a short follow-up like *"sí"* or
  *"merci"* is inherently too short for reliable statistical language ID
  (by design, `language.py` flags anything under ~3 words as
  `reliable: false`), so the chatbot instead uses whatever language the
  user was last, reliably, speaking in that session — this is also how
  **context and intent are preserved through a language switch** (ask about
  a refund in English, then ask a Spanish follow-up about international
  orders, and it's still understood as a continuation of the same topic).
- **`chatbot.py`** — ties it together: detect → resolve → translate to
  pivot → retrieve → generate (in the LLM path, the system prompt
  explicitly instructs the model to answer in the detected language using
  consistent terminology) → record the turn.
- The `/api/chat` response includes `detected_language`,
  `is_mixed_language`, `languages_detected`, and `session_id` (pass this
  back on the next request to keep the same conversation's memory); the UI
  shows this as a small tag under each answer.

### Consistent responses across languages

Because retrieval always happens against the English-pivoted query, asking
the same underlying question in any supported language retrieves the *same*
knowledge-base source — verified by asking "What is your refund policy?" in
all four languages and confirming each one's top retrieval hit was
`refund-policy`. Terminology is kept consistent via `glossary.json`, which
both backs the fallback translator and (in the LLM path) is referenced in
the system prompt.

### Adding a 5th+ language

1. Add it to `SUPPORTED_LANGUAGES` in `language.py`.
2. Add its translations to each entry in `glossary.json` (for the offline
   fallback + terminology consistency).
3. Add its Argos Translate pair(s) to `PAIRS` in `argos_setup.py` and rerun
   the script, if you want real MT rather than the glossary fallback for
   that language.

### Known limitations

- Statistical language ID is inherently unreliable on very short text
  (single words, 1-2 word replies) — this is exactly why the
  conversation-memory fallback exists, but a *first* message that's very
  short and in a new language may default to English or the pivot language
  until enough text arrives to detect confidently.
- The glossary fallback translator only translates phrases it knows; text
  outside the glossary passes through unchanged rather than being guessed
  at. Install Argos Translate packages (`argos_setup.py`) for full free-text
  translation coverage.
- Cross-lingual retrieval currently relies on translate-then-match against
  a lexical (hashing) embedder. For genuinely semantic cross-lingual
  retrieval without a translation step, swap `embeddings.py`'s
  `HashingEmbeddingFunction` for a multilingual sentence-transformers model
  (e.g. `paraphrase-multilingual-MiniLM-L12-v2`), which maps semantically
  similar text in different languages to nearby vectors directly.

## Getting a live/public link

Running `python app.py` only serves on `localhost` — no one else can reach
it. Two options:

### Quick & temporary (ngrok)
Good for demos today; the link dies when you close the terminal.
1. Install ngrok (ngrok.com/download) and run `ngrok config add-authtoken <token>` once.
2. Start the app as usual: `python app.py`.
3. In a second terminal: `ngrok http 5000`. It prints a public URL
   (`https://xxxx.ngrok-free.app`) that forwards to your local app.

### Permanent (deploy it) — e.g. Render, Railway, Fly.io
This project ships a `Procfile` so it's ready for any Heroku-style PaaS.
Using [Render](https://render.com) as an example:
1. Push this project to a GitHub repo.
2. On Render: **New → Web Service**, connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: leave it to the `Procfile`, or set explicitly:
   `gunicorn --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT app:app`
5. Add environment variables (`ANTHROPIC_API_KEY` and/or
   `OPENROUTER_API_KEY`) under the service's Environment tab if you want
   LLM-generated answers.
6. Deploy. Render gives you a permanent `https://your-app.onrender.com` URL.

**Important — always run with exactly 1 worker.** Conversation memory
(`conversation.py`) and the background scheduler both live in-process; more
than one gunicorn worker would mean separate, inconsistent copies of each
(a user's session could hit a different worker each request and "forget"
context, and the scheduler could double-ingest sources). If you need to
scale beyond one process, move `conversation.py` to Redis and run the
scheduler as its own separate worker process instead of inside the web
process — see the "Upgrading for production" section above.

**Also worth knowing**: most free PaaS tiers use an *ephemeral* filesystem —
your ChromaDB data in `data/chroma/` can be wiped on redeploy/restart. For a
demo this is usually fine (sources re-ingest automatically on the
scheduler's next pass); for real persistence, mount a persistent disk/volume
(Render and Railway both offer this) at `data/chroma/`, or point
`knowledge_base.py`'s `DB_DIR` at a managed Chroma server instead.
