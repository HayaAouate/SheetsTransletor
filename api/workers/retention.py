"""Automatic clean-up of DATA_DIR, so the server does not fill up with audio nobody reopens.

Rules (settings.py):
  - a failed job is deleted entirely after ERROR_RETENTION_DAYS;
  - the audio of a finished job (original, stem, preview = ~40 MB per 3-min song) is dropped after
    AUDIO_RETENTION_DAYS without being opened; its score, MIDI, MusicXML and timeline (a few KB) stay,
    so the history and the PDF remain, only the synced playback needs a new run;
  - if the folder still exceeds STORAGE_MAX_GB, audio of the least recently opened jobs goes first.
Runs in its own thread every RETENTION_INTERVAL_MIN, and once at startup. Never touches a queued or running job.
"""
import logging
import threading
import time

from .. import settings
from ..storage import database, files

log = logging.getLogger("sheets.retention")

_thread: threading.Thread | None = None


def _last_activity(job: dict) -> float:
    return job["last_opened_at"] or job["updated_at"]


def sweep() -> dict:
    """Apply the rules once. Returns counts, for the log and for tests."""
    now = time.time()
    deleted = purged = 0
    freed = 0
    jobs = database.list_transcriptions(limit=None)

    for job in jobs:
        age_days = (now - _last_activity(job)) / 86400
        if job["status"] == "error" and age_days > settings.ERROR_RETENTION_DAYS:
            files.delete_folder(job["id"])
            database.delete_transcription(job["id"])
            deleted += 1
        elif job["status"] == "done" and age_days > settings.AUDIO_RETENTION_DAYS:
            got = files.delete_audio(job["id"])
            if got:
                freed += got
                purged += 1

    total = sum(files.size_bytes(job["id"]) for job in database.list_transcriptions(limit=None))
    limit = settings.STORAGE_MAX_GB * 1024 ** 3
    if total > limit:
        # least recently opened first; audio is what weighs, the score files are negligible
        candidates = sorted((j for j in database.list_transcriptions(limit=None) if j["status"] == "done"),
                            key=_last_activity)
        for job in candidates:
            if total <= limit:
                break
            got = files.delete_audio(job["id"])
            if got:
                total -= got
                freed += got
                purged += 1

    if deleted or purged:
        log.info("Nettoyage : %d échec(s) supprimé(s), audio purgé sur %d transcription(s), %.0f Mo libérés, "
                 "%.2f Go utilisés", deleted, purged, freed / 1024 ** 2, total / 1024 ** 3)
    return {"deleted": deleted, "purged": purged, "freed_bytes": freed, "total_bytes": total}


def _run_forever():
    while True:
        try:
            sweep()
        except Exception:  # never let a clean-up failure kill the thread
            log.exception("Nettoyage : erreur")
        time.sleep(settings.RETENTION_INTERVAL_MIN * 60)


def start() -> None:
    global _thread
    if _thread is None:
        _thread = threading.Thread(target=_run_forever, name="retention", daemon=True)
        _thread.start()
