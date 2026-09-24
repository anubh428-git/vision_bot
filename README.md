Common setup notes
Each project has its own requirements.txt — install per-project, not globally, to avoid version conflicts (e.g. different transformers/torch pins).
Projects using a local open-source LLM (arxiv_chatbot, multimodal_assistant) default to a local Ollama server; set LLM_BACKEND=hf in their config.py to use Hugging Face transformers in-process instead.
Projects using a third-party API (openrouter_vision_bot, gemini_chatbot) need an API key, entered via the sidebar or a .env file — never commit API keys to the repo.
For deploying any of these publicly, see each project's README for specifics; in general, Streamlit apps deploy cleanly to Streamlit Community Cloud, and the Flask-based support-chatbot-kb deploys to any standard Python host (Render, Railway, etc.) with a gunicorn app:app start command.
