"""The transcription pipeline as one function writing into a transcription's folder: what the
Streamlit app does interactively, without the UI. Called by the worker (queue.py).

    audio (upload or link) -> Demucs stem -> notes + beats -> preview audio + MIDI -> score PDF/MusicXML (+ tab)
"""
import json
import logging
import os
import shutil
import time

import numpy as np
import soundfile as sf

from transcriber.audio_input import download_audio_from_url
from transcriber.guitar_tabs import build_tab, export_tab_pdf, render_ascii_tab
from transcriber.notation import CREDIT, configure_lilypond, export_pdf, key_label, transcription_to_score
from transcriber.separate import STEM_CHOICES, isolate_stem
from transcriber.transcribe import transcribe_audio

from .. import settings
from ..storage import database, files

log = logging.getLogger("sheets.pipeline")

_lilypond_ready = False


def _ensure_lilypond():
    global _lilypond_ready
    if not _lilypond_ready:
        configure_lilypond()
        _lilypond_ready = True


def _to_flac(src_wav: str, dst_flac: str) -> None:
    """Store audio as FLAC: lossless, 2-3x smaller than WAV, playable by every browser."""
    data, sr = sf.read(src_wav, dtype="float32")
    sf.write(dst_flac, data, sr, format="FLAC", subtype="PCM_16")


def run_transcription(tid: str, report) -> None:
    """
    Run the whole pipeline for transcription `tid` (row already created, source file in its folder
    for an upload). `report(fraction, message)` is called at each step.
    Files land in the transcription's folder (see storage/files.py); results go into the database.
    """
    _ensure_lilypond()
    t0 = time.time()
    job = database.get_transcription(tid)
    out = files.folder(tid)

    # 1. Audio
    report(0.02, "Récupération de l'audio")
    if job["source_kind"] == "url":
        downloaded = download_audio_from_url(job["source"])
        original = os.path.join(out, "original" + (os.path.splitext(downloaded)[1] or ".wav"))
        shutil.move(downloaded, original)
    else:
        original = files.path_of(tid, "original")
        if not original:
            raise RuntimeError("Fichier audio absent.")
    import librosa

    duration = float(librosa.get_duration(path=original))
    database.update_transcription(tid, duration_s=duration)
    if duration > settings.MAX_AUDIO_MIN * 60:  # a job costs minutes of CPU per minute of audio
        raise RuntimeError(f"Audio trop long ({duration / 60:.1f} min) : {settings.MAX_AUDIO_MIN:g} min maximum.")

    # 2. Source separation
    stem_path = original
    if STEM_CHOICES.get(job["stem_choice"]) is not None:
        report(0.05, "Séparation de sources (Demucs)")

        def on_separation(info):
            total = info.get("audio_length") or 1
            done = info.get("segment_offset", 0)
            report(0.05 + 0.25 * min(done / total, 1.0), "Séparation de sources (Demucs)")

        separated = isolate_stem(original, job["stem_choice"], progress_callback=on_separation)
        stem_path = os.path.join(out, "stem.flac")
        _to_flac(separated, stem_path)
        os.remove(separated)
    log.info("[%s] audio + stem prêts (%.0fs)", tid, time.time() - t0)

    # 3. Notes + rhythm
    report(0.32, "Suivi de mélodie (CREPE) + temps (Beat This!)" if job["method"] == "melody"
           else "Détection des notes (Basic Pitch)")
    tr = transcribe_audio(stem_path, instrument=job["instrument"], tempo_audio_path=original, method=job["method"])
    if not tr.notes:
        raise RuntimeError("Aucune note détectée.")
    log.info("[%s] %d notes, %.0f bpm (%.0fs)", tid, len(tr.notes), tr.bpm, time.time() - t0)

    # 4. Preview audio (synthesized, on the recording's timeline) + MIDI
    report(0.85, "Synthèse de l'aperçu")
    pm = tr.to_pretty_midi(program=40 if job["instrument"] == "Violin" else 24)
    wave = pm.synthesize(fs=22050)
    wave = wave / (np.abs(wave).max() or 1.0) * 0.8
    sf.write(os.path.join(out, "preview.flac"), wave.astype(np.float32), 22050, format="FLAC", subtype="PCM_16")
    pm.write(os.path.join(out, "score.mid"))

    # 5. Score: MusicXML + PDF (+ guitar tab)
    report(0.9, "Gravure de la partition")
    score, detected_key = transcription_to_score(tr, title=job["title"], artist=job["artist"] or None)
    export_pdf(score, os.path.join(out, "score.pdf"))
    if job["instrument"] == "Guitar":
        ascii_tab = render_ascii_tab(build_tab(tr))
        export_tab_pdf(ascii_tab, os.path.join(out, "tab.pdf"), title=job["title"], subtitle=job["artist"], credit=CREDIT,
                       info=f"{tr.bpm:.0f} bpm — 4/4 — une colonne = une double-croche")
        with open(os.path.join(out, "tab.txt"), "w", encoding="utf-8") as f:
            f.write(ascii_tab)

    with open(os.path.join(out, "timeline.json"), "w", encoding="utf-8") as f:
        json.dump({"bpm": tr.bpm, "beat_origin": tr.beat_origin,
                   "beat_times": [float(t) for t in tr.beat_times] if tr.beat_times else None}, f)
    database.update_transcription(tid, bpm=float(tr.bpm), key_label=key_label(detected_key), note_count=len(tr.notes))
    report(1.0, "Terminé")
    log.info("[%s] terminé en %.0fs", tid, time.time() - t0)
