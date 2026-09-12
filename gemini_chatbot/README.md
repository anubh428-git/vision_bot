# Gemini Chatbot (PaLM's successor)

## ⚠️ About "installing PaLM"

Google's **PaLM API was fully retired in August 2024** — it can no longer be
installed, called, or used in any new project (Google announced this well in
advance: https://ai.google.dev/palm_docs/deprecation). It was replaced by the
**Gemini API**, which is what this app uses.

The good news: the migration is small. Gemini is a strict upgrade (better
quality, multimodal by default, larger context), and the Python SDK is
similarly simple. This project is a ready-to-run chatbot built on the
**current** SDK (`google-genai` — note: the older `google-generativeai`
package is now *also* deprecated, so this doesn't use that either).

## Setup

```bash
pip install -r requirements.txt
```

Get a free API key at **https://aistudio.google.com/apikey** (any Google
account works, no credit card needed for the free tier). Either:
- paste it into the sidebar text box when the app is running, or
- copy `.env.example` to `.env` and put your key there:
  ```bash
  cp .env.example .env
  # then edit .env and paste your key
  ```

## Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Using it

- Type a question and hit enter to chat normally — conversation history is
  kept automatically for the session (the SDK's `Chat` object tracks it).
- Optionally attach image(s) with the uploader before sending a message —
  Gemini is natively multimodal, so no special handling is needed.
- Switch models, adjust temperature, or set a system instruction in the
  sidebar at any time.
- "Clear conversation" resets everything, including sentiment history.

## Sentiment analysis

Every user message is scored **positive / neutral / negative** (plus an
intensity level and a 0–100 satisfaction score) before the model replies,
and the reply's tone is adjusted accordingly:

- **Positive** — the model matches your energy and reinforces what's working.
- **Neutral** — normal, efficient, helpful tone.
- **Negative** — the model leads with a brief, genuine acknowledgment of the
  frustration (not repeated over-apologizing), then stays solution-focused.
  The strength of the empathy language scales with how negative the message is.
- **Sustained negativity** (2+ negative messages in the last 3) — the model
  is additionally instructed to proactively offer a concrete next step, such
  as escalating to a human agent, and the sidebar shows an escalation warning.

The sidebar shows a live dashboard: average satisfaction score, the most
recent turn's score/label, a satisfaction trend line across the conversation,
and a positive/neutral/negative count breakdown — a running proxy for
customer satisfaction impact, not just a per-message label.

Detection uses **VADER** (`nltk.sentiment.vader`), tuned for short informal
text (handles negation, intensifiers, and punctuation emphasis well — the
kind of text a chat message actually is). If the VADER lexicon can't be
downloaded (e.g. no internet access to NLTK's data server), it falls back
automatically to a small built-in lexicon scorer with basic negation
handling, so the app still works either way.

Run `python tests/test_sentiment.py` to see the accuracy report (93% on the
bundled 30-message labeled test set) and confirm the tone/escalation logic.

## How it works

`app.py`:
- `genai.Client(api_key=...)` — the SDK client
- `client.chats.create(model=...)` — starts a chat session; the returned
  `Chat` object keeps the full message history internally
- Each turn, `src/sentiment_analysis.py` scores the message, then
  `src/response_strategy.py` composes a per-turn system instruction (your
  base instruction + sentiment-based tone guidance)
- `chat.send_message([text, *images], config=...)` — sends the turn with
  that turn's dynamic config; the SDK accepts a plain list mixing strings
  and `PIL.Image` objects directly (no manual base64 encoding needed), and
  a per-call `config` overrides temperature/system instruction for just
  that turn while history keeps accumulating
- Streamlit session state holds the `Chat` object, the sentiment history
  (for the dashboard + escalation detection), and a display log for
  rendering the conversation + sentiment badges + image thumbnails

```
src/
  sentiment_analysis.py   # VADER (+ lexicon fallback) -> SentimentResult(label, compound, intensity, satisfaction_score)
  response_strategy.py     # SentimentResult -> tone instruction injected into the model's system instruction
tests/
  test_sentiment.py          # accuracy on a 30-message labeled set + response-strategy correctness checks
```

## Models

The sidebar includes a few current options (as of Aug 2026):

| Option | Model ID | Notes |
|---|---|---|
| Gemini Flash (latest) | `gemini-flash-latest` | Alias for Google's current default fast model — safest pick if you don't want to track version numbers |
| Gemini 2.5 Flash-Lite | `gemini-2.5-flash-lite` | Cheapest option |
| Gemini 3.7 Flash | `gemini-3.7-flash` | Newest, strongest Flash-tier model |
| Gemini 3.1 Pro | `gemini-3.1-pro` | Most capable, slower/pricier |

Google's model lineup moves fast — check
https://ai.google.dev/gemini-api/docs/models for the current full list and
swap in any model ID you like by adding it to `MODEL_OPTIONS` in `app.py`.

## Notes

- Nothing is persisted to disk — history lives only in the browser session
  and resets if you refresh or restart the app.
- If you get an API error, double check the key is valid and pasted in full
  (Google AI Studio only shows it once at creation — regenerate if lost).
