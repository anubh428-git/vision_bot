"""
Periodic knowledge-base updater.

Each source has its own `update_interval_minutes`. A background job polls
every `poll_seconds` and, for each source whose interval has elapsed,
re-fetches the content. If the content hash is unchanged, nothing is
re-embedded (cheap no-op). If it changed (or ingestion is forced), old
chunks for that source are replaced with freshly chunked + embedded ones.
"""
import time
import hashlib
import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

import config
import ingestion
from knowledge_base import KnowledgeBase

logger = logging.getLogger("kb-scheduler")
logging.basicConfig(level=logging.INFO)

kb = KnowledgeBase()


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ingest_source(source: dict, force: bool = False) -> dict:
    """Fetch a source, chunk & embed it if content changed, persist state."""
    result = {"id": source["id"], "status": "skipped", "chunks": 0}
    try:
        text = ingestion.load_source_content(source)
        new_hash = _content_hash(text)

        if not force and source.get("last_content_hash") == new_hash:
            result["status"] = "unchanged"
            source["last_checked_at"] = time.time()
            config.upsert_source(source)
            return result

        chunks = ingestion.chunk_text(text)
        kb.delete_source(source["id"])  # clear stale chunks for this source
        n = kb.upsert_chunks(source["id"], chunks)

        source["last_content_hash"] = new_hash
        source["last_updated_at"] = time.time()
        source["last_checked_at"] = time.time()
        source["last_error"] = None
        config.upsert_source(source)

        result["status"] = "updated"
        result["chunks"] = n
        logger.info("Updated source '%s': %d chunks", source["id"], n)
    except Exception as e:
        logger.exception("Failed to ingest source %s", source["id"])
        source["last_checked_at"] = time.time()
        source["last_error"] = str(e)
        config.upsert_source(source)
        result["status"] = "error"
        result["error"] = str(e)
    return result


def run_all_due(force: bool = False):
    sources = config.load_sources()
    results = []
    now = time.time()
    for source in sources:
        interval_min = source.get("update_interval_minutes", 60)
        last_checked = source.get("last_checked_at", 0)
        due = force or (now - last_checked) >= interval_min * 60
        if due:
            results.append(ingest_source(source, force=force))
    return results


class KBScheduler:
    def __init__(self, poll_seconds: int = 60):
        self.scheduler = BackgroundScheduler()
        self.poll_seconds = poll_seconds

    def start(self):
        self.scheduler.add_job(
            run_all_due,
            trigger=IntervalTrigger(seconds=self.poll_seconds),
            id="kb_poll",
            replace_existing=True,
            max_instances=1,
        )
        self.scheduler.start()
        logger.info("Knowledge base scheduler started (poll every %ss)", self.poll_seconds)

    def shutdown(self):
        self.scheduler.shutdown(wait=False)
