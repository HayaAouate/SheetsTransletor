"""The files of a transcription: one folder per id under DATA_DIR.

    <id>/original.<ext>   the audio as received (upload) or downloaded (link)
    <id>/stem.flac        the isolated track the notes were read from
    <id>/preview.flac     the transcription synthesized, on the recording's timeline
    <id>/score.musicxml   score.pdf   score.mid   [tab.pdf, tab.txt for guitar]
    <id>/timeline.json    bpm, beat_origin, beat_times (for the cursor of the score player)

Audio (original, stem, preview) is the bulk of the size and is dropped by retention after a while;
the score files are tiny and kept.
"""
import json
import os
import shutil

from .. import settings

# kind -> file names, first one is what we write now; the others are older layouts still readable.
FILE_NAMES = {
    "original": ("original.*",),
    "stem": ("stem.flac", "stem.wav"),
    "preview": ("preview.flac", "preview.wav"),
    "musicxml": ("score.musicxml",),
    "pdf": ("score.pdf",),
    "midi": ("score.mid",),
    "tab_pdf": ("tab.pdf",),
    "timeline": ("timeline.json",),
}
AUDIO_KINDS = ("original", "stem", "preview")
MEDIA_TYPES = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac", ".m4a": "audio/mp4",
               ".ogg": "audio/ogg", ".webm": "audio/webm", ".opus": "audio/ogg",
               ".musicxml": "application/vnd.recordare.musicxml+xml", ".pdf": "application/pdf",
               ".mid": "audio/midi", ".json": "application/json", ".txt": "text/plain"}


def folder(tid: str) -> str:
    path = os.path.join(settings.DATA_DIR, tid)
    os.makedirs(path, exist_ok=True)
    return path


def path_of(tid: str, kind: str) -> str | None:
    """Absolute path of one of the transcription's files, or None if absent."""
    directory = os.path.join(settings.DATA_DIR, tid)
    if not os.path.isdir(directory):
        return None
    for name in FILE_NAMES.get(kind, ()):
        if name.endswith("*"):
            prefix = name[:-1]
            matches = sorted(f for f in os.listdir(directory) if f.startswith(prefix))
            if matches:
                return os.path.join(directory, matches[0])
        elif os.path.exists(os.path.join(directory, name)):
            return os.path.join(directory, name)
    return None


def available_kinds(tid: str) -> list:
    return [kind for kind in FILE_NAMES if path_of(tid, kind)]


def read_timeline(tid: str) -> dict | None:
    path = path_of(tid, "timeline")
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def size_bytes(tid: str) -> int:
    directory = os.path.join(settings.DATA_DIR, tid)
    if not os.path.isdir(directory):
        return 0
    return sum(os.path.getsize(os.path.join(directory, f)) for f in os.listdir(directory))


def delete_audio(tid: str) -> int:
    """Remove the audio files (the heavy ones), keep the score. Returns the bytes freed."""
    freed = 0
    for kind in AUDIO_KINDS:
        path = path_of(tid, kind)
        if path:
            freed += os.path.getsize(path)
            os.remove(path)
    return freed


def delete_folder(tid: str) -> None:
    shutil.rmtree(os.path.join(settings.DATA_DIR, tid), ignore_errors=True)
