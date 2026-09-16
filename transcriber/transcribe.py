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
    bpm = _readable_tempo(bpm, notes)
    bpm, beat_origin = _refine_grid(notes, bpm, beat_origin)
    log.info("Tempo : %.1f bpm, origine=%.3fs", bpm, beat_origin)
    mono = _clean_and_quantize(notes, bpm, beat_origin)
    mono = _merge_false_splits(mono, audio_path)
    mono = _drop_leading_bleed(mono)
    shift = _align_downbeat(mono)
    beat_origin -= shift * (60.0 / bpm * GRID)  # slots moved by +shift -> origin moves by -shift
    mono = _simplify_rhythm(mono)
    log.info("Rythme : %d notes, grille %s", len(mono), "croche" if _choose_unit(mono) == 2 else "double-croche")
    return Transcription(notes=mono, bpm=bpm, beat_origin=beat_origin, instrument=instrument)


def _drop_weak_notes(notes, ratio: float = 0.45):
    """Drop notes far quieter than the rest: bleed from other instruments / octave ghosts."""
    if len(notes) < 8:
        return notes
    floor = ratio * float(np.median([n.amplitude for n in notes]))
    return [n for n in notes if n.amplitude >= floor]


def _readable_tempo(bpm: float, notes) -> float:
    """
    Beat trackers sometimes lock on half or double the felt tempo. For notation the difference is
    huge: at half tempo every eighth note is written as a sixteenth. Only the extreme cases are
    corrected, from the spacing between note onsets (not their sounded length, which depends on
    how the note decays): a melody whose typical note-to-note spacing is a sixteenth or less is
    written at double tempo, one whose notes are two beats apart or more at half tempo.
    """
    if len(notes) < 8:
        return bpm
    onsets = np.array(sorted(n.start for n in notes))
    ioi = np.diff(onsets)
    ioi = ioi[ioi > 0.03]  # ignore near-simultaneous detections (octave ghosts)
    if len(ioi) < 6:
        return bpm
    beats = float(np.median(ioi)) * bpm / 60.0
    if beats < 0.35 and bpm * 2 <= 200:
        return bpm * 2
    if beats > 1.75 and bpm / 2 >= 55:
        return bpm / 2
    return bpm


def _drop_leading_bleed(mono, ratio: float = 0.6, lead_beats: float = 2.0):
    """
    Quiet notes long before the first real note are separation bleed (voice, drums...), not music:
    keeping them produces empty-looking bars at the top of the score. Anything within `lead_beats`
    of the first real note is kept as a possible pickup.
    """
    if len(mono) < 4:
        return mono
    threshold = ratio * max(n.amplitude for n in mono)
    real = [n for n in mono if n.amplitude >= threshold]
    # The melody starts where real notes come in a row, not at an isolated loud blip 10 s earlier.
    first = next((n for i, n in enumerate(real)
                  if len([m for m in real[i + 1:] if m.slot - n.slot <= 8 * SLOTS_PER_BEAT]) >= 2),
                 real[0] if real else mono[0])
    limit = first.slot - lead_beats * SLOTS_PER_BEAT
    return [n for n in mono if n.slot >= limit]


def _choose_unit(mono, max_lost: float = 0.20) -> int:
    """Rhythmic unit in grid slots: 2 (eighth notes) unless that would merge > 20 % of the notes."""
    if len(mono) < 4:
        return 2
    seen, lost = set(), 0
    for n in mono:
        s = _snap(n.slot, 2)
        lost += s in seen
        seen.add(s)
    return 2 if lost / len(mono) <= max_lost else 1


def _snap(slot: int, unit: int) -> int:
    """Nearest multiple of `unit`, halves rounded up (Python's round() rounds them to even, which
    would pair slots inconsistently and create phantom collisions)."""
    return int(np.floor(slot / unit + 0.5)) * unit


NICE_LENGTHS = (1, 2, 3, 4, 6, 8, 12, 16)  # in units: 8th, quarter, dotted quarter, half, ... whole


def _simplify_rhythm(mono):
    """
    Make the written rhythm as simple as a musician would write it by hand:
      * onsets on the eighth-note grid when the music allows it (sixteenths only if needed),
      * a gap shorter than a beat before the next note is not a rest, the note just lasts longer
        (Basic Pitch stops a note when it decays, a player would hold it),
      * durations rounded to standard values (no double dots, no 5-sixteenths-tied-to-something).
    """
    if not mono:
        return mono
    unit = _choose_unit(mono)
    best = {}
    for n in mono:  # collisions on the coarser grid: keep the louder / longer note
        s = _snap(n.slot, unit)
        cur = best.get(s)
        if cur is None or (n.amplitude, n.length) > (cur.amplitude, cur.length):
            n.slot = s
            best[s] = n
    mono = [best[s] for s in sorted(best)]

    beat_units = SLOTS_PER_BEAT // unit
    for i, n in enumerate(mono):
        sounded = max(1, int(round(n.length / unit)))
        if i + 1 < len(mono):
            room = (mono[i + 1].slot - n.slot) // unit  # units until the next onset
            if room - sounded < beat_units:  # short gap -> legato, no fiddly rest
                sounded = room
        else:
            room = max(sounded, 1)
        # Always a standard value (music21 would otherwise write a double-dotted note); what is
        # left before the next note becomes a rest, split cleanly on the beats by the notation.
        length = max(v for v in NICE_LENGTHS if v <= max(1, min(sounded, room)))
        n.length = max(1, min(length, room)) * unit
    return mono


def _merge_false_splits(mono, audio_path: str, dip_ratio: float = 0.4):
    """
    Basic Pitch cuts a sustained note in two whenever its onset detector fires, which drum bleed in
    a separated stem does at every hi-hat. Two contiguous same-pitch notes are re-joined unless the
    audio shows a real re-attack at the split: an energy dip right before the second note starts
    (a sustained note that merely gets a drum hit on top never dips). The dip must be deep: vibrato
    or tremolo modulate the level by 10-30 %, a re-bowed / re-plucked note drops well below half.
    """
    if len(mono) < 2:
        return mono
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    hop = 64  # ~3 ms resolution; short frames so that a quick re-attack of the same note shows as a dip
    rms = librosa.feature.rms(y=y, frame_length=256, hop_length=hop)[0]
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
        # The global tempo estimate is coarse (~0.5 bpm); the median beat interval is far more
        # precise (and, unlike a linear fit, immune to the odd skipped/doubled beat), which matters
        # over a 3-minute song where 1 % of error drifts the grid by several beats.
        period = float(np.median(np.diff(beat_times)))
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
    # Coarse pass (±10 %: beat trackers are regularly 5-8 % off, e.g. 121 for a song at 130), then a
    # fine pass around the winner: the onsets are precise to ~10 ms, so the tempo can be pinned to
    # ~0.1 bpm. A wrong tempo spreads the onsets uniformly over the grid, so the true one stands out.
    for span, steps in ((0.10, 401), (0.002, 41)):
        center = best[1]
        for b in center * np.linspace(1 - span, 1 + span, steps):
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

    # 2. Bar lines: of the 4 possible beat offsets, keep the one where long/loud notes start on
    #    downbeats and few notes straddle a bar line (a note tied across the bar is what a wrong
    #    bar line produces). Then start measure 1 on the bar of the first real note ("real" = loud
    #    enough not to be separation bleed).
    threshold = 0.6 * max(n.amplitude for n in mono)

    def bar_score(k):
        score = 0.0
        for n in mono:
            pos = (n.slot - k * SLOTS_PER_BEAT) % SLOTS_PER_BAR
            weight = n.amplitude * n.length
            if pos == 0:
                score += weight
            if pos + n.length > SLOTS_PER_BAR:
                score -= 0.5 * weight
        return score

    k = max(range(4), key=bar_score)
    first = next((n for n in mono if n.amplitude >= threshold), mono[0])
    first_bar = (first.slot - k * SLOTS_PER_BEAT) // SLOTS_PER_BAR
    shift = -(k * SLOTS_PER_BEAT + first_bar * SLOTS_PER_BAR)
    if shift:
        apply(shift)
        total += shift
    lowest = min(n.slot for n in mono)
    if lowest < 0:  # notes before the first real one (bleed / pickup): add whole bars in front
        shift = -(lowest // SLOTS_PER_BAR) * SLOTS_PER_BAR
        apply(shift)
        total += shift
    return total
