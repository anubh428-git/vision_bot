"""
Flask web app for the self-updating support chatbot (Streamlit-free UI).

Routes:
  GET  /                         -> chat + knowledge-source management UI
  POST /api/chat                 -> {message} -> RAG answer
  GET  /api/sources              -> list configured sources + chunk counts
  POST /api/sources              -> add a source ({type, location, ...}) and ingest it immediately
  DELETE /api/sources/<id>       -> remove a source and its vectors
  POST /api/sources/<id>/refresh -> force re-ingest a single source now
  POST /api/sources/refresh-all  -> force re-ingest every source now
"""
import time
import uuid

from flask import Flask, request, jsonify, render_template

import config
from scheduler import KBScheduler, ingest_source, run_all_due
from chatbot import answer as chatbot_answer
from knowledge_base import KnowledgeBase

app = Flask(__name__)
kb = KnowledgeBase()
kb_scheduler = KBScheduler(poll_seconds=60)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    question = (data.get("message") or "").strip()
    if not question:
        return jsonify({"error": "message is required"}), 400
    return jsonify(chatbot_answer(question))


@app.route("/api/sources", methods=["GET"])
def list_sources():
    sources = config.load_sources()
    for s in sources:
        s["chunk_count"] = kb.get_source_chunk_count(s["id"])
    return jsonify({"sources": sources, "stats": kb.stats()})


@app.route("/api/sources", methods=["POST"])
def add_source():
    data = request.get_json(force=True) or {}
    s_type = data.get("type")
    location = data.get("location")
    if s_type not in ("url", "text", "file") or not location:
        return jsonify({"error": "type (url|text|file) and location are required"}), 400

    source = {
        "id": data.get("id") or f"{s_type}-{uuid.uuid4().hex[:8]}",
        "type": s_type,
        "location": location,
        "update_interval_minutes": int(data.get("update_interval_minutes", 60)),
        "created_at": time.time(),
        "last_checked_at": 0,
        "last_updated_at": None,
        "last_content_hash": None,
        "last_error": None,
    }
    config.upsert_source(source)
    result = ingest_source(source, force=True)
    return jsonify({"source": source, "ingest_result": result}), 201


@app.route("/api/sources/<source_id>", methods=["DELETE"])
def delete_source(source_id):
    kb.delete_source(source_id)
    config.remove_source(source_id)
    return jsonify({"status": "deleted", "id": source_id})


@app.route("/api/sources/<source_id>/refresh", methods=["POST"])
def refresh_source(source_id):
    source = config.get_source(source_id)
    if not source:
        return jsonify({"error": "not found"}), 404
    return jsonify(ingest_source(source, force=True))


@app.route("/api/sources/refresh-all", methods=["POST"])
def refresh_all():
    return jsonify({"results": run_all_due(force=True)})


if __name__ == "__main__":
    kb_scheduler.start()
    app.run(host="0.0.0.0", port=5000, debug=False)
