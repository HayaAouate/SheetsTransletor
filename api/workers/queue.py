"""One background worker, one transcription at a time.

A transcription needs 3-4 GB of RAM and every CPU core for minutes (Demucs, CREPE, Beat This!),
so jobs are queued and run strictly one after the other in a single thread of the API process.
On startup, jobs left "queued" or "running" by a previous process are re-queued.
"""
import logging
import queue
import threading
import traceback

from ..storage import database
from . import pipeline

log = logging.getLogger("sheets.queue")

_queue: "queue.Queue[str]" = queue.Queue()
_thread: threading.Thread | None = None


def enqueue(tid: str) -> None:
    _queue.put(tid)


def pending_count() -> int:
    return _queue.qsize()


def _work_forever():
    while True:
        tid = _queue.get()
        try:
            database.update_transcription(tid, status="running", progress=0.0, message="Démarrage")

            def report(fraction, message):
                database.update_transcription(tid, progress=float(fraction), message=message)

            pipeline.run_transcription(tid, report)
            database.update_transcription(tid, status="done", progress=1.0, message="")
        except Exception as exc:  # the job fails, the worker lives on
            log.error("[%s] échec : %s\n%s", tid, exc, traceback.format_exc())
            database.update_transcription(tid, status="error", message=str(exc)[:500])
        finally:
            _queue.task_done()


def start() -> None:
    """Start the worker thread (idempotent) and re-queue what a previous run left unfinished."""
    global _thread
    if _thread is not None:
        return
    for job in reversed(database.list_transcriptions(limit=None)):  # oldest first
        if job["status"] in ("queued", "running"):
            database.update_transcription(job["id"], status="queued", progress=0.0, message="En attente")
            _queue.put(job["id"])
    _thread = threading.Thread(target=_work_forever, name="transcription-worker", daemon=True)
    _thread.start()
