"""/transcriptions — create a job, follow it, list the history, download its files, delete it.

    POST   /transcriptions                      multipart: file or url, title, artist, instrument, method, stem_choice
    GET    /transcriptions                      history, newest first
    GET    /transcriptions/{id}                 status, progress, results, timeline, files ready
    GET    /transcriptions/{id}/files/{kind}    original | stem | preview | musicxml | pdf | midi | tab_pdf | timeline
    DELETE /transcriptions/{id}
All behind the API key (see security.py).
"""
import logging
import os
import re
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from transcriber.separate import DEFAULT_STEM, STEM_CHOICES

from .. import settings
from ..schemas import FileKind, TranscriptionOut
from ..security import require_api_key
from ..storage import database, files
from ..workers import queue

log = logging.getLogger("sheets.routes")

router = APIRouter(prefix="/transcriptions", tags=["transcriptions"], dependencies=[Depends(require_api_key)])

INSTRUMENTS = ("Violin", "Guitar")
METHODS = ("melody", "basic_pitch")
DOWNLOAD_SUFFIX = {"tab_pdf": "_tab", "stem": "_piste_isolee", "preview": "_transcription", "original": "_original"}


def _safe_filename(name: str, fallback: str) -> str:
    cleaned = re.sub(r"[^\w\- ]+", "", name or "").strip().replace(" ", "_")
    return cleaned or fallback


def _url_allowed(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == h or host.endswith("." + h) for h in settings.URL_HOSTS)


def _to_public(job: dict) -> dict:
    """The row plus what is derived from the folder: files ready, timeline, place in the queue."""
    out = dict(job)
    out["files"] = files.available_kinds(job["id"])
    out["timeline"] = files.read_timeline(job["id"]) if job["status"] == "done" else None
    out["queue_position"] = queue.pending_count() if job["status"] == "queued" else 0
    return out


def _get_or_404(tid: str) -> dict:
    job = database.get_transcription(tid)
    if not job:
        raise HTTPException(404, "Transcription inconnue.")
    return job


@router.get("", response_model=list[TranscriptionOut])
def list_transcriptions():
    return [_to_public(j) for j in database.list_transcriptions()]


@router.post("", status_code=201, response_model=TranscriptionOut)
async def create_transcription(
    file: UploadFile | None = File(None, description="Audio file (mp3, wav, m4a, ogg, webm) — or give `url`"),
    url: str = Form("", description="YouTube / TikTok / ... link — or give `file`"),
    title: str = Form(""),
    artist: str = Form(""),
    instrument: str = Form("Violin", description="Violin | Guitar"),
    method: str = Form("", description="melody (default for violin) | basic_pitch (default for guitar)"),
    stem_choice: str = Form("", description="Demucs track to isolate; default depends on the instrument"),
):
    # --- validation ---
    if instrument not in INSTRUMENTS:
        raise HTTPException(400, f"instrument doit être parmi {INSTRUMENTS}")
    method = method or ("melody" if instrument == "Violin" else "basic_pitch")
    if method not in METHODS:
        raise HTTPException(400, f"method doit être parmi {METHODS}")
    stem_choice = stem_choice or DEFAULT_STEM[instrument]
    if stem_choice not in STEM_CHOICES:
        raise HTTPException(400, f"stem_choice doit être parmi {list(STEM_CHOICES)}")
    url = url.strip()
    has_file = file is not None and bool(file.filename)
    if not url and not has_file:
        raise HTTPException(400, "Fournis un fichier audio (file) ou un lien (url).")
    if url and not _url_allowed(url):
        raise HTTPException(400, f"Lien refusé : sites autorisés {', '.join(settings.URL_HOSTS)}.")
    if has_file:
        ext = os.path.splitext(file.filename)[1].lower() or ".wav"
        if not files.MEDIA_TYPES.get(ext, "").startswith("audio/"):
            raise HTTPException(400, "Formats acceptés : mp3, wav, m4a, ogg, webm.")
    if queue.pending_count() >= settings.MAX_PENDING:
        raise HTTPException(429, f"File d'attente pleine ({settings.MAX_PENDING} transcriptions en attente) : réessaie plus tard.")

    # --- create the row, store the upload, queue the job ---
    title = title.strip() or ("Transcription violon" if instrument == "Violin" else "Tablature guitare")
    job = database.create_transcription(title, artist.strip(), instrument, method, stem_choice,
                                        "url" if url else "upload", url or file.filename)
    if has_file:
        limit = int(settings.MAX_UPLOAD_MB * 1024 * 1024)
        written = 0
        with open(os.path.join(files.folder(job["id"]), "original" + ext), "wb") as out:
            while chunk := await file.read(1024 * 1024):  # streamed: checked without buffering the whole file
                written += len(chunk)
                if written > limit:
                    break
                out.write(chunk)
        if written > limit:
            files.delete_folder(job["id"])
            database.delete_transcription(job["id"])
            raise HTTPException(413, f"Fichier trop gros : {settings.MAX_UPLOAD_MB:g} Mo maximum.")
    database.update_transcription(job["id"], message="En attente")
    queue.enqueue(job["id"])
    log.info("Job %s créé (%s)", job["id"], job["source"])
    return _to_public(database.get_transcription(job["id"]))


@router.get("/{tid}", response_model=TranscriptionOut)
def get_transcription(tid: str):
    job = _get_or_404(tid)
    if job["status"] == "done":
        database.mark_opened(tid)  # keeps its audio alive (retention)
    return _to_public(job)


@router.get("/{tid}/files/{kind}", response_class=FileResponse)
def download_file(tid: str, kind: FileKind):
    job = _get_or_404(tid)
    path = files.path_of(tid, kind)
    if not path:
        raise HTTPException(404, f"Fichier '{kind}' non disponible.")
    ext = os.path.splitext(path)[1].lower()
    base = _safe_filename(" - ".join(x for x in (job["title"], job["artist"]) if x), "partition")
    return FileResponse(path, media_type=files.MEDIA_TYPES.get(ext, "application/octet-stream"),
                        filename=f"{base}{DOWNLOAD_SUFFIX.get(kind, '')}{ext}")


@router.delete("/{tid}", status_code=204)
def delete_transcription(tid: str):
    job = _get_or_404(tid)
    if job["status"] == "running":
        raise HTTPException(409, "Transcription en cours : attends qu'elle se termine.")
    files.delete_folder(tid)
    database.delete_transcription(tid)
