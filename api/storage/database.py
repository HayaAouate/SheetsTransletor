"""The transcriptions table (SQLite, one file: DATA_DIR/db.sqlite). Files live next to it, see files.py.

One row per transcription: what was asked (title, instrument, source...), where the job is
(status, progress, message) and what came out (bpm, key, note count, duration).
"""
import os
import sqlite3
import threading
import time
import uuid

from .. import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS transcriptions (
    id             TEXT PRIMARY KEY,
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL,
    last_opened_at REAL,                   -- last GET /transcriptions/{id}; NULL = never reopened
    title          TEXT NOT NULL,
    artist         TEXT NOT NULL DEFAULT '',
    instrument     TEXT NOT NULL,          -- Violin | Guitar
    method         TEXT NOT NULL,          -- melody | basic_pitch
    stem_choice    TEXT NOT NULL,          -- label of transcriber.separate.STEM_CHOICES
    source_kind    TEXT NOT NULL,          -- upload | url
    source         TEXT NOT NULL,          -- file name or URL
    status         TEXT NOT NULL,          -- queued | running | done | error
    progress       REAL NOT NULL DEFAULT 0,
    message        TEXT NOT NULL DEFAULT '',
    bpm            REAL,
    key_label      TEXT,
    note_count     INTEGER,
    duration_s     REAL
);
"""
# Columns added after the first release: applied to databases created before them.
_MIGRATIONS = [("last_opened_at", "ALTER TABLE transcriptions ADD COLUMN last_opened_at REAL")]

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _connection() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        conn = sqlite3.connect(os.path.join(settings.DATA_DIR, "db.sqlite"), check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        existing = {row[1] for row in conn.execute("PRAGMA table_info(transcriptions)")}
        for column, statement in _MIGRATIONS:
            if column not in existing:
                conn.execute(statement)
        conn.commit()
        _conn = conn
    return _conn


def create_transcription(title: str, artist: str, instrument: str, method: str, stem_choice: str,
                         source_kind: str, source: str) -> dict:
    tid = uuid.uuid4().hex[:12]
    now = time.time()
    with _lock:
        _connection().execute(
            "INSERT INTO transcriptions (id, created_at, updated_at, title, artist, instrument, method, stem_choice, "
            "source_kind, source, status) VALUES (?,?,?,?,?,?,?,?,?,?,'queued')",
            (tid, now, now, title, artist, instrument, method, stem_choice, source_kind, source),
        )
        _connection().commit()
    return get_transcription(tid)


def update_transcription(tid: str, **fields) -> None:
    fields["updated_at"] = time.time()
    columns = ", ".join(f"{name} = ?" for name in fields)
    with _lock:
        _connection().execute(f"UPDATE transcriptions SET {columns} WHERE id = ?", (*fields.values(), tid))
        _connection().commit()


def mark_opened(tid: str) -> None:
    """Record that someone looked at this transcription (keeps its audio from being purged)."""
    with _lock:
        _connection().execute("UPDATE transcriptions SET last_opened_at = ? WHERE id = ?", (time.time(), tid))
        _connection().commit()


def get_transcription(tid: str) -> dict | None:
    row = _connection().execute("SELECT * FROM transcriptions WHERE id = ?", (tid,)).fetchone()
    return dict(row) if row else None


def list_transcriptions(limit: int | None = 200) -> list:
    """Newest first."""
    sql = "SELECT * FROM transcriptions ORDER BY created_at DESC"
    rows = _connection().execute(sql + (f" LIMIT {int(limit)}" if limit else "")).fetchall()
    return [dict(r) for r in rows]


def delete_transcription(tid: str) -> None:
    with _lock:
        _connection().execute("DELETE FROM transcriptions WHERE id = ?", (tid,))
        _connection().commit()
