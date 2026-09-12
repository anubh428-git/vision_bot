# OpenRouter Vision Bot

A simple text + image analyzer chatbot, running locally via Streamlit, powered
by the [OpenRouter](https://openrouter.ai) API (one key, access to GPT-4o,
Claude, Gemini, Llama vision models, and more).

## Setup

```bash
pip install -r requirements.txt
```

Get a free API key at https://openrouter.ai/keys. Either:
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

- Type a question and hit enter to chat normally.
- Attach one or more images with the uploader, then ask a question about
  them (e.g. "what's in this image?", "read the text in this photo",
  "compare these two pictures").
- Conversation history is kept for the session, so follow-up questions work.
- Switch models or adjust temperature/max tokens in the sidebar at any time.
- "Clear conversation" resets the chat.

## How it works

`app.py` is a single self-contained file:
- `image_to_data_url()` — encodes an uploaded image as a base64 JPEG data URL
- `build_user_content()` — builds an OpenAI-style multi-part message
  (`[{"type": "text", ...}, {"type": "image_url", ...}]`), which is the
  format OpenRouter's vision-capable models expect
- `call_openrouter()` — POSTs the full conversation history to
  `https://openrouter.ai/api/v1/chat/completions` and returns the reply
- Streamlit session state keeps both the API-format message history (sent
  on every call, for multi-turn context) and a display log (for rendering
  the chat + thumbnails)

## Notes

- Nothing is persisted to disk — history lives only in the browser session
  and resets if you refresh or restart the app.
- Swap in any other vision-capable model from https://openrouter.ai/models
  by adding it to the `MODEL_OPTIONS` dict in `app.py`.
- Some models on OpenRouter are pay-per-token; a couple of free-tier options
  (like the Llama 3.2 Vision `:free` variant) are included in the model list.
