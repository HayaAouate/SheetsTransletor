"""Audio input helpers: load a local uploaded file or download audio from a YouTube link."""
import logging
import os
import tempfile

import yt_dlp

log = logging.getLogger("sheets.audio")


def get_audio_path(source: str, is_url: bool, upload_bytes: bytes = None, upload_name: str = None) -> str:
    """
    Return a local filesystem path to an audio file ready for transcription.

    - If is_url is True: `source` is a YouTube (or Instagram/TikTok) URL, downloaded via yt-dlp.
    - If is_url is False: `upload_bytes`/`upload_name` are the bytes/filename of a user-uploaded file,
      written to a temp file and returned.
    """
    if is_url:
        return download_audio_from_url(source)

    if not upload_bytes:
        raise ValueError("No audio provided: pass either a URL or uploaded file bytes.")

    suffix = os.path.splitext(upload_name or "")[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload_bytes)
    tmp.close()
    return tmp.name


def download_audio_from_url(url: str) -> str:
    """Download the best audio track from a YouTube/Instagram/TikTok link and convert it to WAV."""
    out_dir = tempfile.mkdtemp()
    out_template = os.path.join(out_dir, "audio.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        # Sortie yt-dlp (résolution, téléchargement, conversion ffmpeg) visible dans le terminal.
        "quiet": False,
        "no_warnings": False,
        "noprogress": True,
    }
    log.info("Téléchargement audio : %s", url)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    wav_path = os.path.join(out_dir, "audio.wav")
    if not os.path.exists(wav_path):
        # Fallback: postprocessor sometimes keeps a different name; grab whatever landed in out_dir.
        candidates = [f for f in os.listdir(out_dir) if not f.endswith(".part")]
        if not candidates:
            raise RuntimeError("Le téléchargement audio a échoué (aucun fichier produit).")
        wav_path = os.path.join(out_dir, candidates[0])
    log.info("Audio téléchargé : %s", wav_path)
    return wav_path
