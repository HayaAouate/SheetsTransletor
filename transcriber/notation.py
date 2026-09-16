"""Turn a cleaned Transcription into notated sheet music (MusicXML + PDF), for melodic instruments
(violin, flute, voice, etc). The Transcription is already monophonic and quantized on a beat grid
(see transcribe.py), so this module only has to lay it out: tempo, key, measures, rests.
"""
import gc
import os
import re
import shutil
import subprocess
import time

from music21 import clef, environment, instrument, key, metadata, meter, note, stream, tempo, tie

from .tools import ensure_tools_on_path, find_tool
from .transcribe import Transcription

CREDIT = "Violon d'or"  # printed top-right of every score, like a composer/arranger credit
_lilypond_bin = None


def configure_lilypond(lilypond_path: str = None):
    """
    Point music21 at the Lilypond binary used to render PDFs. Call this once at app startup.
    If lilypond_path is not given, looks in PATH and then in the usual install folders (winget,
    Program Files...), and also puts FFmpeg on PATH the same way (see tools.py).
    """
    ensure_tools_on_path()
    resolved = lilypond_path or find_tool("lilypond")
    if not resolved:
        raise RuntimeError(
            "Lilypond introuvable. Installe-le avec `winget install LilyPond.LilyPond` "
            "(ou https://lilypond.org/download.html) puis relance l'application."
        )
    global _lilypond_bin
    _lilypond_bin = resolved
    us = environment.UserSettings()
    us["lilypondPath"] = resolved


def transcription_to_score(tr: Transcription, title: str = None, artist: str = None):
    """Build a music21 Score from a Transcription. Returns (score, detected_key or None).
    `title` is the song title (big, centered), `artist` the subtitle under it."""
    part = stream.Part()
    part.insert(0, instrument.Violin() if tr.instrument == "Violin" else instrument.Guitar())
    part.insert(0, clef.TrebleClef())
    part.insert(0, meter.TimeSignature("4/4"))
    part.insert(0, tempo.MetronomeMark(number=int(round(tr.bpm))))

    for n in tr.notes:
        for offset, ql, is_last in _standard_pieces(n.offset_beats, n.duration_beats):
            m21 = note.Note(n.pitch, quarterLength=ql)
            if not is_last:
                m21.tie = tie.Tie("start")
            part.insert(offset, m21)

    detected_key = None
    if tr.notes:
        try:
            detected_key = part.analyze("key")
            detected_key = _simplest_enharmonic_key(detected_key)
            part.insert(0, detected_key)
            _respell_pitches(part, detected_key)
        except Exception:
            pass

    part.makeRests(fillGaps=True, inPlace=True)
    _split_rests(part)
    part.makeMeasures(inPlace=True)

    score = stream.Score()
    if title or artist:
        score.insert(0, metadata.Metadata(title=title or " ", alternativeTitle=artist or None))
    score.insert(0, part)
    return score, detected_key


STANDARD_QL = (4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.375, 0.25)  # whole ... 16th, with single dots


def _standard_pieces(offset: float, ql: float):
    """
    Split a duration into standard values tied together, so music21 never has to write a
    double-dotted note or a 5-sixteenths duration: 1.25 beats -> quarter ~ sixteenth.
    Yields (offset, quarterLength, is_last).
    """
    pieces = []
    remaining = ql
    while remaining > 1e-6:
        d = next((v for v in STANDARD_QL if v <= remaining + 1e-6), STANDARD_QL[-1])
        pieces.append(d)
        remaining -= d
    pos = offset
    for i, d in enumerate(pieces):
        yield pos, d, i == len(pieces) - 1
        pos += d


def _split_rests(part: stream.Part, beats_per_bar: int = 4):
    """
    makeRests fills each gap with a single rest of arbitrary length (3.5 beats -> a double-dotted
    half rest). Musicians write rests beat by beat: whole bar, half on beats 1/3, quarters on beats,
    eighths in between. Split every rest into such pieces, in place.
    """
    for r in list(part.getElementsByClass(note.Rest)):
        pos, end = float(r.offset), float(r.offset + r.quarterLength)
        pieces = []
        while end - pos > 1e-6:
            bar_pos, remaining = pos % beats_per_bar, end - pos
            if bar_pos == 0 and remaining >= beats_per_bar:
                d = beats_per_bar
            elif bar_pos % 2 == 0 and remaining >= 2:
                d = 2
            elif bar_pos % 1 == 0 and remaining >= 1:
                d = 1
            elif bar_pos % 0.5 == 0 and remaining >= 0.5:
                d = 0.5
            else:
                d = min(remaining, 0.25 - (bar_pos % 0.25) or 0.25)
            pieces.append((pos, d))
            pos += d
        if len(pieces) > 1:
            part.remove(r)
            for off, d in pieces:
                part.insert(off, note.Rest(quarterLength=d))


def key_label(k) -> str:
    """'e- minor' -> 'Mi♭ mineur', for the UI."""
    if k is None:
        return "—"
    names = {"C": "Do", "D": "Ré", "E": "Mi", "F": "Fa", "G": "Sol", "A": "La", "B": "Si"}
    tonic = names[k.tonic.step] + {"-": "♭", "#": "♯"}.get(k.tonic.accidental.modifier if k.tonic.accidental else "", "")
    return f"{tonic} {'mineur' if k.mode == 'minor' else 'majeur'}"


def _simplest_enharmonic_key(k: key.Key) -> key.Key:
    """
    Prefer the enharmonic key with fewer accidentals (Db major over C# major, etc.). At 6 it is a
    tie (F# / Gb): pick the flat side, the usual choice in written music (Eb minor over D# minor).
    """
    if abs(k.sharps) < 6:
        return k
    alt = key.Key(k.tonic.getEnharmonic(), k.mode)
    if abs(alt.sharps) < abs(k.sharps) or (abs(alt.sharps) == abs(k.sharps) and alt.sharps < 0):
        return alt
    return k


def _respell_pitches(part: stream.Part, k: key.Key):
    """
    Notes come from MIDI numbers, which music21 spells with sharps by default (G#, C#...). In a
    flat key those must read as Ab, Db...: same sound, but they sit in the key signature and no
    longer need an accidental. For each black-key note keep the spelling that is diatonic in the
    key; for chromatic notes, follow the key's own accidental direction (sharps or flats).
    """
    for n in part.recurse().notes:
        p = n.pitch
        if not p.spellingIsInferred or not p.accidental or p.accidental.alter == 0:
            continue  # white keys stay natural (E# / Fb spellings are harder to read, not easier)
        candidates = [p, p.getEnharmonic()]  # G#4 <-> Ab4
        diatonic = [c for c in candidates if _is_diatonic(c, k)]
        if diatonic:
            n.pitch = diatonic[0]
        else:  # chromatic note: flats in flat keys, sharps otherwise
            wanted = -1 if k.sharps < 0 else 1
            n.pitch = next((c for c in candidates if c.accidental.alter == wanted), p)
        n.pitch.spellingIsInferred = False


def _is_diatonic(p, k: key.Key) -> bool:
    """True when `p` is spelled as a degree of the key (its accidental is the key signature's)."""
    expected = k.accidentalByStep(p.step)
    actual = p.accidental
    if expected is None:
        return actual is None or actual.alter == 0
    return actual is not None and actual.alter == expected.alter


def export_pdf(score: stream.Score, out_path: str, credit: str = CREDIT):
    """Write the score as MusicXML and as a PDF (via Lilypond). Returns (pdf_path, musicxml_path)."""
    xml_path = out_path.replace(".pdf", ".musicxml")
    score.write("musicxml", xml_path)

    # music21 only exports title/subtitle to the Lilypond header, so write the .ly ourselves, add the
    # right-aligned credit (Lilypond's `composer` slot) and drop the "engraved by Lilypond" tagline,
    # then run Lilypond directly.
    base_path = out_path[:-4] if out_path.endswith(".pdf") else out_path
    ly_path = str(score.write("lily", fp=base_path))
    with open(ly_path, encoding="utf-8") as f:
        ly = f.read()
    # `score.write("lily")` keeps the lilypond-book preamble (one cropped page per system); the
    # "lily.pdf" writer strips it before running Lilypond, and so do we.
    ly = ly.replace('\\include "lilypond-book-preamble.ly"', "")
    extra = "tagline = ##f\n"
    if credit:
        extra += 'composer = "%s"\n' % credit.replace("\\", "\\\\").replace('"', '\\"')
    if "\\header" in ly:
        ly = re.sub(r"\\header\s*\{", lambda m: m.group(0) + "\n" + extra, ly, count=1)
    else:
        ly = ly.replace("\\score", "\\header {\n" + extra + "}\n\\score", 1)
    with open(ly_path, "w", encoding="utf-8") as f:
        f.write(ly)

    lilypond = _lilypond_bin or shutil.which("lilypond")
    produced_path = base_path + ".pdf"
    # Lilypond (Guile) needs a few hundred MB to start. Right after Demucs / CREPE the process is
    # at its memory peak and the machine may be swapping: free what we can first, and if Lilypond
    # dies without a word (no output, no PDF), give it a second chance after a pause.
    gc.collect()
    for attempt in range(2):
        result = subprocess.run(
            [lilypond, "--pdf", "-o", base_path, ly_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if os.path.exists(produced_path) or result.stderr.strip() or attempt:
            break
        time.sleep(3)
    if result.returncode != 0 or not os.path.exists(produced_path):
        raise RuntimeError(
            f"Lilypond a échoué (code {result.returncode}, PDF attendu : {produced_path}, "
            f"présent : {os.path.exists(produced_path)}, fichier .ly : {ly_path}).\n"
            "stderr :\n" + (result.stderr[-2000:] or "(vide)") + "\nstdout :\n" + (result.stdout[-1000:] or "(vide)")
        )
    if produced_path != out_path:
        os.replace(produced_path, out_path)
    return out_path, xml_path
