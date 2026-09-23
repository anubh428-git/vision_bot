"""
Simple JSON-backed registry of knowledge sources. Each source record tracks
its own update interval and last-ingested state so the scheduler knows when
it's due for a refresh, and so unchanged content can be skipped cheaply.
"""
import json
import os
import threading

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "sources.json")
_lock = threading.Lock()


def load_sources() -> list:
    with _lock:
        if not os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "w") as f:
                json.dump([], f)
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)


def save_sources(sources: list):
    with _lock:
        with open(CONFIG_PATH, "w") as f:
            json.dump(sources, f, indent=2)


def upsert_source(source: dict):
    sources = load_sources()
    for i, s in enumerate(sources):
        if s["id"] == source["id"]:
            sources[i] = source
            save_sources(sources)
            return
    sources.append(source)
    save_sources(sources)


def remove_source(source_id: str):
    sources = [s for s in load_sources() if s["id"] != source_id]
    save_sources(sources)


def get_source(source_id: str):
    for s in load_sources():
        if s["id"] == source_id:
            return s
    return None
