"""Core transcription step: audio file -> clean, tempo-aligned note list.

Pipeline:
  1. Basic Pitch (Spotify, open source) detects notes (start/end in seconds, pitch, amplitude),
     restricted to the target instrument's frequency range.
  2. librosa estimates the tempo and the beat phase on the ORIGINAL mix (drums help), so note
     positions can be expressed in beats instead of seconds -> the written rhythm matches the audio.
  3. Cleanup: drop bleed/ghost notes, monophonic reduction (loudest note per grid slot), no overlaps,
     16th-note quantization, re-join notes falsely split by drum transients, bar-line alignment.
"""
import logging
from dataclasses import dataclass, field

import librosa
import numpy as np
import pretty_midi
from basic_pitch.inference import predict

GRID = 0.25  # rhythmic grid, in beats (0.25 = 16th note)

# Playable MIDI range per instrument: (lowest, highest). Anything outside is a transcription artifact.
INSTRUMENT_RANGES = {
    "Violin": (55, 100),  # G3 .. E7
    "Guitar": (40, 88),  # E2 .. E6 (standard tuning, ~24th fret)
}


@dataclass
class NoteEvent:
    start: float  # seconds
    end: float  # seconds
    pitch: int  # MIDI number
    amplitude: float  # Basic Pitch confidence/energy, 0..1
    slot: int = 0  # quantized onset, in GRID units from the first downbeat
    length: int = 1  # quantized duration, in GRID units (>= 1)

    @property
    def offset_beats(self) -> float:
        return self.slot * GRID

    @property
    def duration_beats(self) -> float:
        return self.length * GRID


@dataclass
class Transcription:
    notes: list = field(default_factory=list)  # list[NoteEvent], time-sorted, monophonic
    bpm: float = 120.0
    beat_origin: float = 0.0  # seconds; time of the first written beat (measure 1, beat 1)
    instrument: str = "Violin"

    @property
    def beat_seconds(self) -> float:
        return 60.0 / self.bpm

    def to_pretty_midi(self, program: int = 40) -> pretty_midi.PrettyMIDI:
        """Cleaned notes re-aligned on the grid, as MIDI (used for the audio preview / MIDI download)."""
        pm = pretty_midi.PrettyMIDI(initial_tempo=self.bpm)
        inst = pretty_midi.Instrument(program=program)
        for n in self.notes:
            start = max(0.0, self.beat_origin + n.offset_beats * self.beat_seconds)
            end = start + n.duration_beats * self.beat_seconds  # pretty_midi cannot synthesize t < 0
            velocity = int(np.clip(40 + 87 * n.amplitude, 40, 127))
            inst.notes.append(pretty_midi.Note(velocity=velocity, pitch=n.pitch, start=start, end=end))
        pm.instruments.append(inst)
        return pm


log = logging.getLogger("sheets.transcribe")


def transcribe_audio(
    audio_path: str,
    instrument: str = "Violin",
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length: float = 80.0,
    tempo_audio_path: str = None,
) -> Transcription:
    """
    Run the full transcription on `audio_path` (ideally an isolated stem, see separate.py).

    onset_threshold: confidence required to start a new note (lower = more notes, more false positives).
    frame_threshold: confidence required to keep sustaining a note.
    minimum_note_length: shortest note kept, in milliseconds (filters out noise blips).
    tempo_audio_path: audio used for tempo/beat detection; defaults to `audio_path`. Pass the original
                      mix when transcribing a separated stem: drums make beat tracking far more reliable.
    """
    low, high = INSTRUMENT_RANGES.get(instrument, (36, 96))
    log.info("Basic Pitch sur %s (onset=%.2f, min_note=%.0fms)", audio_path, onset_threshold, minimum_note_length)
    _, _, note_events = predict(
        audio_path,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length,
        minimum_frequency=librosa.midi_to_hz(low),
        maximum_frequency=librosa.midi_to_hz(high),
        melodia_trick=True,
    )

    notes = [
        NoteEvent(start=float(s), end=float(e), pitch=int(p), amplitude=float(a))
        for s, e, p, a, _bends in note_events
        if low <= int(p) <= high
    ]
    notes.sort(key=lambda n: (n.start, -n.amplitude))
    log.info("Basic Pitch : %d notes brutes dans la tessiture", len(notes))
    notes = _drop_weak_notes(notes)

    bpm, beat_origin = detect_tempo(tempo_audio_path or audio_path)
    bpm, beat_origin = _refine_grid(notes, bpm, beat_origin)
    log.info("Tempo : %.1f bpm, origine=%.3fs", bpm, beat_origin)
    mono = _clean_and_quantize(notes, bpm, beat_origin)
    mono = _merge_false_splits(mono, audio_path)
    shift = _align_downbeat(mono)
    beat_origin -= shift * (60.0 / bpm * GRID)  # slots moved by +shift -> origin moves by -shift
    return Transcription(notes=mono, bpm=bpm, beat_origin=beat_origin, instrument=instrument)


def _drop_weak_notes(notes, ratio: float = 0.45):
    """Drop notes far quieter than the rest: bleed from other instruments / octave ghosts."""
    if len(notes) < 8:
        return notes
    floor = ratio * float(np.median([n.amplitude for n in notes]))
    return [n for n in notes if n.amplitude >= floor]


def _merge_false_splits(mono, audio_path: str, dip_ratio: float = 0.5):
    """
    Basic Pitch cuts a sustained note in two whenever its onset detector fires, which drum bleed in
    a separated stem does at every hi-hat. Two contiguous same-pitch notes are re-joined unless the
    audio shows a real re-attack at the split: an energy dip right before the second note starts
    (a sustained note that merely gets a drum hit on top never dips).
    """
    if len(mono) < 2:
        return mono
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    hop = 128  # ~6 ms resolution
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    def rms_between(t0, t1, reduce):
        mask = (times >= t0) & (times <= t1)
        return float(reduce(rms[mask])) if mask.any() else 0.0

    merged = [mono[0]]
    for n in mono[1:]:
        prev = merged[-1]
        if prev.pitch == n.pitch and prev.slot + prev.length == n.slot:
            dip = rms_between(n.start - 0.08, n.start + 0.02, np.min)
            level = rms_between(n.start, n.start + 0.12, np.max)
            if dip >= dip_ratio * level:  # no dip -> same note still ringing -> false split
                prev.length += n.length
                prev.end = n.end
                prev.amplitude = max(prev.amplitude, n.amplitude)
                continue
        merged.append(n)
    return merged


def detect_tempo(audio_path: str):
    """Return (bpm, beat_origin_seconds). bpm is folded into the 70-180 range (readable notation)."""
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    if len(y) < sr:  # < 1 s of audio: not enough to track anything
        return 120.0, 0.0

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    bpm = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0
    if not bpm or not np.isfinite(bpm):
        return 120.0, 0.0
    while bpm < 70:
        bpm *= 2
    while bpm > 180:
        bpm /= 2

    if len(beat_frames) == 0:
        return bpm, 0.0
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    if len(beat_times) >= 8:
        # The global tempo estimate is coarse (~0.5 bpm); a linear fit through the tracked beats
        # gives the mean beat period far more precisely, which matters over a 3-minute song.
        period, _ = np.polyfit(np.arange(len(beat_times)), beat_times, 1)
        fitted = 60.0 / period
        while fitted < 70:
            fitted *= 2
        while fitted > 180:
            fitted /= 2
        if abs(fitted - bpm) / bpm < 0.08:  # sanity: stay close to the global estimate
            bpm = fitted

    beat_sec = 60.0 / bpm
    first_beat = float(beat_times[0])
    # Beat grid phase: earliest point >= 0 that is on the beat.
    origin = first_beat - int(first_beat / beat_sec) * beat_sec
    return bpm, origin


def _refine_grid(notes, bpm: float, coarse_origin: float):
    """
    Fine-tune (bpm, grid phase) so the detected note onsets fall as close as possible to grid points.
    librosa's beat phase is only accurate to a few tens of ms and its tempo to ~0.5 bpm; Basic Pitch
    onsets also run slightly early. Together that is enough to push notes one 16th off, so we search
    a small neighbourhood (±1.5 % tempo, ±half a cell phase) and keep the best-aligned grid.
    """
    if len(notes) < 4:
        return bpm, coarse_origin
    onsets = np.array([n.start for n in notes])
    weights = np.array([n.amplitude for n in notes]) + 1e-3

    def cost(b, phase):
        rel = (onsets - phase) / (60.0 / b * GRID)
        dist = np.abs(rel - np.round(rel))  # 0 = on the grid, 0.5 = exactly between two points
        return float(np.sum(weights * dist**2))

    best = (cost(bpm, coarse_origin), bpm, coarse_origin)
    for b in bpm * np.linspace(0.985, 1.015, 31):
        grid_sec = 60.0 / b * GRID
        for phase in coarse_origin + np.linspace(-grid_sec / 2, grid_sec / 2, 25):
            c = cost(b, phase)
            if c < best[0]:
                best = (c, float(b), float(phase))
    return best[1], best[2]


def _clean_and_quantize(notes, bpm: float, beat_origin: float):
    """Snap notes to the beat grid and force a single, playable melodic line."""
    grid_sec = 60.0 / bpm * GRID
    quantized = []
    for n in notes:
        n.slot = max(0, int(round((n.start - beat_origin) / grid_sec)))
        end_slot = int(round((n.end - beat_origin) / grid_sec))
        n.length = max(1, end_slot - n.slot)
        quantized.append(n)

    # Monophonic: one note per grid slot, keep the loudest (tie -> highest pitch).
    best = {}
    for n in quantized:
        cur = best.get(n.slot)
        if cur is None or (n.amplitude, n.pitch) > (cur.amplitude, cur.pitch):
            best[n.slot] = n
    mono = [best[s] for s in sorted(best)]

    # No overlaps: a note stops when the next one starts.
    for cur, nxt in zip(mono, mono[1:]):
        cur.length = max(1, min(cur.length, nxt.slot - cur.slot))

    return mono


SLOTS_PER_BEAT = int(round(1 / GRID))
SLOTS_PER_BAR = 4 * SLOTS_PER_BEAT  # 4/4


def _align_downbeat(mono) -> int:
    """
    Beat trackers happily lock onto off-beats (hi-hats) and know nothing about bar lines. Re-phase
    the grid from the notes themselves: most onsets fall on beats, and the longest/loudest notes fall
    on downbeats. Shifts every slot in place and returns the shift applied (in grid units).
    """
    if not mono:
        return 0

    def apply(shift):
        for n in mono:
            n.slot += shift

    # 1. Beat phase: rotate so the 16th position with the most onsets becomes "on the beat".
    counts = [sum(1 for n in mono if n.slot % SLOTS_PER_BEAT == r) for r in range(SLOTS_PER_BEAT)]
    r = int(np.argmax(counts))
    total = 0
    if r:
        shift = -r if mono[0].slot >= r else SLOTS_PER_BEAT - r
        apply(shift)
        total += shift

    # 2. Downbeat: without drums there is no reliable way to find bar lines, so use the predictable
    #    convention "measure 1 starts on the beat of the first real note" (a pickup will be off by
    #    a beat, easy to fix in MuseScore). "Real" = loud enough not to be separation bleed.
    threshold = 0.6 * max(n.amplitude for n in mono)
    first = next((n for n in mono if n.amplitude >= threshold), mono[0])
    shift = -(first.slot // SLOTS_PER_BEAT) * SLOTS_PER_BEAT
    if shift:
        apply(shift)
        total += shift
    lowest = min(n.slot for n in mono)
    if lowest < 0:  # notes before the first real one (bleed / pickup): add whole bars in front
        shift = -(lowest // SLOTS_PER_BAR) * SLOTS_PER_BAR
        apply(shift)
        total += shift
    return total
