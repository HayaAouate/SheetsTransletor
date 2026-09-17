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
from basic_pitch.note_creation import model_frames_to_time

from .beats import BeatMap, track_beats

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
    onset: float = 1.0  # Basic Pitch onset activation at the note start, 0..1 (how clear the attack is)
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
    # Seconds of every written beat (index k = beat k of the score, measure 1 beat 1 = 0), from the
    # tracked beats when available: the score's timeline on the original recording, tempo drift
    # included. None -> constant tempo (beat_origin + k * 60 / bpm).
    beat_times: list = None

    @property
    def beat_seconds(self) -> float:
        return 60.0 / self.bpm

    def seconds_at(self, beats: float) -> float:
        """Time in the original recording of a position in the score, in beats."""
        if not self.beat_times:
            return self.beat_origin + beats * self.beat_seconds
        bt = np.asarray(self.beat_times)
        if beats <= 0:
            return float(bt[0] + beats * (bt[1] - bt[0]))
        if beats >= len(bt) - 1:
            return float(bt[-1] + (beats - (len(bt) - 1)) * (bt[-1] - bt[-2]))
        return float(np.interp(beats, np.arange(len(bt)), bt))

    def to_pretty_midi(self, program: int = 40) -> pretty_midi.PrettyMIDI:
        """Cleaned notes on the recording's timeline, as MIDI (audio preview / MIDI download)."""
        pm = pretty_midi.PrettyMIDI(initial_tempo=self.bpm)
        inst = pretty_midi.Instrument(program=program)
        for n in self.notes:
            start = max(0.0, self.seconds_at(n.offset_beats))
            end = max(start + 0.05, self.seconds_at(n.offset_beats + n.duration_beats))  # no t < 0
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
    method: str = "basic_pitch",
) -> Transcription:
    """
    Run the full transcription on `audio_path` (ideally an isolated stem, see separate.py).

    onset_threshold: confidence required to start a new note (lower = more notes, more false positives).
    frame_threshold: confidence required to keep sustaining a note.
    minimum_note_length: shortest note kept, in milliseconds (filters out noise blips).
    tempo_audio_path: audio used for tempo/beat detection; defaults to `audio_path`. Pass the original
                      mix when transcribing a separated stem: drums make beat tracking far more reliable.
    method: "basic_pitch" (polyphonic note detector; the three thresholds above apply) or "melody"
            (CREPE pitch tracking + our own segmentation, see melody.py: on a one-voice instrument
            it gets the fast runs and the octaves right where Basic Pitch does not).
    """
    low, high = INSTRUMENT_RANGES.get(instrument, (36, 96))
    if method == "melody":
        notes = _melody_notes(audio_path, low, high)
    else:
        notes = _basic_pitch_notes(audio_path, low, high, onset_threshold, frame_threshold, minimum_note_length)
    log.info("Après filtrage accompagnement / fantômes : %d notes", len(notes))
    # Same-pitch notes cut by the tracker: when is the cut a real re-attack? Basic Pitch's onset
    # activation is a reliable witness (>= 0.7 = re-attacked). CREPE notes carry the stem's onset
    # envelope instead, which vibrato drives up to 1.0 on a held note: there only a deep energy
    # dip (< 0.3 of the level, measured on both a clean take and a cover over a backing track)
    # with some attack behind it (>= 0.4) counts.
    merge_settings = ({"clear_onset": 1.01, "dip_ratio": 0.3, "weak_onset": 0.4, "strong_onset": 0.85} if method == "melody"
                      else {"clear_onset": 0.7, "dip_ratio": 0.4})

    # Beat grid. Preferred: Beat This! on the mix -> beat map (robust to tempo drift) + downbeats
    # (bar lines). Fallback: librosa tempo + phase, refined on the note onsets, bar lines guessed
    # from the notes. Beat This! needs drums or a steady pulse: on a solo instrument played freely
    # it can lock onto nothing in particular, so its grid is only used when the note onsets sit
    # on it at least as well as on the fallback grid.
    beat_times, downbeat_times = track_beats(tempo_audio_path or audio_path)
    bpm, beat_origin = detect_tempo(tempo_audio_path or audio_path)
    bpm = _readable_tempo(bpm, notes)
    bpm, beat_origin = _refine_grid(notes, bpm, beat_origin)
    if beat_times is not None:
        beat_map = BeatMap(beat_times)
        tracked_bpm = _readable_tempo(beat_map.bpm, notes)
        if tracked_bpm != beat_map.bpm:  # halved / doubled for readability: rebuild the map on that grid
            beat_map = BeatMap(_resample_beats(beat_times, tracked_bpm / beat_map.bpm))
        tracked_cost = _grid_cost(notes, beat_map.to_beats)
        fallback_cost = _grid_cost(notes, lambda t: (t - beat_origin) * bpm / 60.0)
        ratio = tracked_bpm / bpm
        same_pulse = any(abs(ratio - r) / r < 0.04 for r in (0.5, 1.0, 2.0))
        log.info("Grille : Beat This! %.1f bpm (coût %.2f) vs librosa %.1f bpm (coût %.2f)%s",
                 tracked_bpm, tracked_cost, bpm, fallback_cost, ", même pulsation" if same_pulse else "")
        # Same pulse (possibly halved / doubled): both heard the same beat, and only Beat This!
        # knows where the bars start. A different pulse: trust whichever the onsets sit on.
        if not same_pulse and tracked_cost > 1.15 * fallback_cost:
            beat_times = None
    if beat_times is not None:
        bpm = tracked_bpm
        mono = _clean_and_quantize(notes, beat_map.to_beats)
        mono = _merge_false_splits(mono, audio_path, **merge_settings)
        mono = _drop_leading_bleed(mono)
        first_downbeat = int(round(beat_map.to_beats(downbeat_times[0]))) * SLOTS_PER_BEAT
        shift = _align_to_downbeats(mono, first_downbeat)
        beat_origin = beat_map.to_seconds(-shift * GRID)  # slot 0 after the shift, in seconds
        # Score beat k sits at original beat k - shift*GRID: the score's timeline, beat by beat.
        last_beat = max((n.slot + n.length for n in mono), default=0) * GRID + 8
        beat_times = [beat_map.to_seconds(k - shift * GRID) for k in range(int(last_beat) + 1)]
        log.info("Tempo : %.1f bpm (Beat This!), mesure 1 à %.2fs", bpm, beat_origin)
    else:
        log.info("Tempo : %.1f bpm (librosa), origine=%.3fs", bpm, beat_origin)
        mono = _clean_and_quantize(notes, lambda t: (t - beat_origin) * bpm / 60.0)
        mono = _merge_false_splits(mono, audio_path, **merge_settings)
        mono = _drop_leading_bleed(mono)
        shift = _align_downbeat(mono)
        beat_origin -= shift * (60.0 / bpm * GRID)  # slots moved by +shift -> origin moves by -shift
        beat_times = None
    mono = _simplify_rhythm(mono, grid_sec=60.0 / bpm * GRID)
    log.info("Rythme : %d notes écrites (%d sur la grille croche)", len(mono), sum(1 for n in mono if n.slot % 2 == 0))
    return Transcription(notes=mono, bpm=bpm, beat_origin=beat_origin, instrument=instrument, beat_times=beat_times)


def _grid_cost(notes, to_beats) -> float:
    """How far the note onsets sit from the 16th grid given by `to_beats` (seconds -> beats):
    amplitude-weighted squared distance, 0 = every onset exactly on the grid. Used to compare two
    candidate grids for the same notes."""
    if not notes:
        return 0.0
    rel = np.array([to_beats(n.start) / GRID for n in notes])
    dist = np.abs(rel - np.round(rel))
    weights = np.array([n.amplitude for n in notes]) + 1e-3
    return float(np.sum(weights * dist**2) / np.sum(weights))


def _resample_beats(beat_times, factor: float):
    """Beat times at `factor` x the tracked rate (2 = every half beat, 0.5 = every other beat)."""
    if factor >= 1:
        n = int(round(factor))
        out = []
        for a, b in zip(beat_times, beat_times[1:]):
            out.extend(np.linspace(a, b, n, endpoint=False))
        out.append(beat_times[-1])
        return np.array(out)
    step = int(round(1 / factor))
    return np.asarray(beat_times)[::step]


def _melody_notes(audio_path, low, high):
    """CREPE pitch tracking + segmentation (melody.py). Amplitudes are stem RMS, 0..1."""
    from .melody import extract_melody

    log.info("CREPE sur %s", audio_path)
    notes = [NoteEvent(start=s, end=e, pitch=p, amplitude=a, onset=attack)
             for s, e, p, a, attack in extract_melody(audio_path, midi_low=low, midi_high=high)]
    return _drop_weak_notes(notes, ratio=0.3)


def _basic_pitch_notes(audio_path, low, high, onset_threshold, frame_threshold, minimum_note_length):
    """Basic Pitch note events in the instrument's range, cleaned of bleed and accompaniment."""
    log.info("Basic Pitch sur %s (onset=%.2f, min_note=%.0fms)", audio_path, onset_threshold, minimum_note_length)
    model_output, _, note_events = predict(
        audio_path,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length,
        minimum_frequency=librosa.midi_to_hz(low),
        maximum_frequency=librosa.midi_to_hz(high),
        melodia_trick=True,
    )
    onset_act = model_output["onset"]  # (frames, 88 pitches from A0)
    frame_times = model_frames_to_time(len(onset_act))

    def onset_strength(start: float, pitch: int) -> float:
        f = int(np.searchsorted(frame_times, start))
        return float(onset_act[max(0, f - 2):f + 3, pitch - 21].max())

    notes = [
        NoteEvent(start=float(s), end=float(e), pitch=int(p), amplitude=float(a),
                  onset=onset_strength(float(s), int(p)))
        for s, e, p, a, _bends in note_events
        if low <= int(p) <= high
    ]
    notes.sort(key=lambda n: (n.start, -n.amplitude))
    log.info("Basic Pitch : %d notes brutes dans la tessiture", len(notes))
    notes = _drop_weak_notes(notes)
    return _drop_accompaniment(notes)


def _drop_weak_notes(notes, ratio: float = 0.55, window: float = 3.0):
    """
    Drop notes far quieter than the notes around them (±`window` s): bleed from other instruments
    and octave ghosts come out of Basic Pitch at 0.25-0.45 while the played line sits at 0.6-0.9.
    The reference is local so that a soft verse is not wiped out by a loud chorus.
    """
    if len(notes) < 8:
        return notes
    starts = np.array([n.start for n in notes])
    amps = np.array([n.amplitude for n in notes])
    keep = []
    for n in notes:
        local = amps[(starts >= n.start - window) & (starts <= n.start + window)]
        if n.amplitude >= ratio * float(np.median(local)):
            keep.append(n)
    return keep


def _drop_accompaniment(notes, margin: float = 0.06, tail: float = 0.15, quieter: float = 0.15):
    """
    A note that starts while a melody note is sounding and is not the continuation of the line is
    accompaniment (piano chord under a held violin note) or a harmonic ghost: without this the held
    note gets cut and the chord tones are written in its place. It is dropped when it is clearly
    quieter than the sounding note, or when it ends well before the sounding note does without
    being louder. A note that runs to the end of the sounding note is a real transition (Basic
    Pitch lets the previous note ring over the next one for a bit).
    """
    kept = []
    for n in notes:  # time-sorted
        sounding = [k for k in kept if k.start + margin <= n.start < k.end - margin]
        if sounding:
            k = max(sounding, key=lambda k: k.amplitude)
            ends_early = n.end <= k.end - tail
            if n.amplitude < k.amplitude - quieter or (ends_early and n.amplitude <= k.amplitude + 0.1):
                continue
        kept.append(n)
    return kept


def _readable_tempo(bpm: float, notes) -> float:
    """
    Beat trackers sometimes lock on half or double the felt tempo. For notation the difference is
    huge: at half tempo every eighth note is written as a sixteenth. Only the extreme cases are
    corrected, from the spacing between note onsets (not their sounded length, which depends on
    how the note decays): a melody where a good share of the note-to-note spacings are a sixteenth
    or less is written at double tempo, one where they are two beats or more at half tempo.
    """
    if len(notes) < 8:
        return bpm
    onsets = np.array(sorted(n.start for n in notes))
    ioi = np.diff(onsets)
    ioi = ioi[ioi > 0.03]  # ignore near-simultaneous detections (octave ghosts)
    if len(ioi) < 6:
        return bpm
    beats = ioi * bpm / 60.0
    # More than 30 % of the notes would be sixteenths (or shorter) at this tempo: write it twice
    # as fast, they become eighths. Symmetrically, more than 30 % longer than two beats: halve.
    if float(np.percentile(beats, 30)) < 0.35 and bpm * 2 <= 200:
        return bpm * 2
    if float(np.percentile(beats, 70)) > 1.75 and bpm / 2 >= 55:
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


def _snap(slot: int, unit: int) -> int:
    """Nearest multiple of `unit`, halves rounded up (Python's round() rounds them to even, which
    would pair slots inconsistently)."""
    return int(np.floor(slot / unit + 0.5)) * unit


def _simplify_rhythm(mono, grid_sec: float, unit: int = 2, max_move: int = 1):
    """
    Make the written rhythm as simple as it can be WITHOUT losing a note:
      * onsets go on the eighth-note grid, each note in its own cell and in playing order, chosen
        together so that the total displacement is minimal (a run played slightly late still reads
        as consecutive eighths); a note that would have to move by more than `max_move` sixteenths
        to fit keeps its sixteenth position (a real fast run is written in sixteenths), and so does
        a note played shorter than a sixteenth (`grid_sec`): an ornament, the sixteenth of a
        dotted-eighth + sixteenth figure,
      * a gap shorter than a beat before the next note is not a rest, the note just lasts longer
        (the tracker stops a note when it decays, a player would hold it),
      * a note followed by a real rest gets a standard duration (the notation writes any longer
        legato value as tied standard values, never as a double-dotted note).
    """
    if not mono:
        return mono
    mono.sort(key=lambda n: n.slot)
    short = [n.end - n.start < grid_sec for n in mono]
    assigned = _assign_to_grid([n.slot for n in mono], unit, max_move, short)
    for n, slot in zip(mono, assigned):
        n.slot = slot
    mono.sort(key=lambda n: n.slot)

    for i, n in enumerate(mono):
        if i + 1 < len(mono):
            room = mono[i + 1].slot - n.slot  # slots until the next onset
            if room - n.length < SLOTS_PER_BEAT:  # short gap -> legato, no fiddly rest
                n.length = room
            else:  # real rest after the note: round the sounded length to a standard value
                n.length = max(v for v in NICE_LENGTHS if v <= max(1, min(n.length, room)))
        else:
            n.length = max(v for v in NICE_LENGTHS if v <= max(1, n.length))
    return mono


def _assign_to_grid(slots, unit: int, max_move: int, short=None):
    """
    Give each onset (sorted 16th slots) a distinct cell of the coarser grid (`unit` slots), in
    order, minimising the total displacement (dynamic programming). Where that would move a note
    by more than `max_move` slots, the notes of that stretch keep their fine slots instead; a
    `short` note may also keep its fine slot (at a small cost, so it does when moving would
    displace it or its neighbours). Returns the new slots (distinct, increasing).
    """
    if not slots:
        return []
    short = short or [False] * len(slots)
    span = max_move + unit
    # Candidates: the coarse cells around the onset, plus the onset's own fine slot. The fine slot
    # always being a candidate (the fine slots are distinct and increasing), a valid path always
    # exists — a dense run of sixteenths simply stays on the fine grid.
    cands = [sorted({(s + d) // unit * unit for d in range(-span, span + 1)} | {_snap(s, unit), s})
             for s in slots]

    def move_cost(i, c):
        if c == slots[i] and c % unit:  # staying off the coarse grid
            return 0.5 if short[i] else max_move + 0.5
        return abs(c - slots[i])

    best = [{c: (move_cost(0, c), None) for c in cands[0]}]
    for i in range(1, len(slots)):
        cur = {}
        for c in cands[i]:
            prev = [(cost, pc) for pc, (cost, _) in best[-1].items() if pc < c]
            if prev:
                cost, pc = min(prev)
                cur[c] = (cost + move_cost(i, c), pc)
        best.append(cur)
    # backtrack
    c = min(best[-1], key=lambda k: best[-1][k][0])
    coarse = [c]
    for i in range(len(slots) - 1, 0, -1):
        c = best[i][c][1]
        coarse.append(c)
    coarse.reverse()

    # A note pushed too far does not fit the coarse grid: this stretch is a genuine fast run,
    # written on the fine grid (its fine slots are distinct already, `_clean_and_quantize`).
    out = list(coarse)
    for i, (c, s) in enumerate(zip(coarse, slots)):
        if abs(c - s) > max_move:
            out[i] = s
    # Keep the sequence strictly increasing after mixing the two grids.
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + 1
    return out


NICE_LENGTHS = (1, 2, 3, 4, 6, 8, 12, 16)  # in 16th slots: 16th, 8th, dotted 8th, quarter, ... whole


def _merge_false_splits(mono, audio_path: str, dip_ratio: float = 0.4, clear_onset: float = 0.7, weak_onset: float = 0.0,
                        strong_onset: float = 2.0):
    """
    Basic Pitch cuts a sustained note in two whenever its onset detector fires above 0.5, which
    vibrato, a change of bow pressure or drum bleed do all the time: a held note comes out as a run
    of repeated sixteenths. Two contiguous same-pitch notes are re-joined unless the audio shows a
    real re-attack at the split, i.e. either:
      * Basic Pitch's own onset activation there is clearly high (>= 0.7: real repeated notes sit
        at 0.75-0.9, vibrato splits hover just above the 0.5 threshold), or
      * a deep energy dip just before the second note (a re-bowed / re-plucked note drops well
        below half; vibrato only modulates the level by 10-30 %).
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
        if prev.pitch == n.pitch and prev.slot + prev.length == n.slot and n.onset < clear_onset:
            dip = rms_between(n.start - 0.08, n.start + 0.02, np.min)
            level = rms_between(n.start, n.start + 0.12, np.max)
            # Nothing re-attacked (same note still ringing): no deep dip, or a dip with no attack
            # at all behind it (`weak_onset`: a drum hit in the stem's residue makes dips too).
            # An attack as sharp as the sharpest in the piece (>= `strong_onset`) is a bow stroke
            # even without a dip: legato repeated notes have no gap between them.
            if (dip >= dip_ratio * level or n.onset < weak_onset) and n.onset < strong_onset:
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

    def search(center, span, steps, start):
        best = start
        for b in center * np.linspace(1 - span, 1 + span, steps):
            grid_sec = 60.0 / b * GRID
            for phase in coarse_origin + np.linspace(-grid_sec / 2, grid_sec / 2, 25):
                c = cost(b, phase)
                if c < best[0]:
                    best = (c, float(b), float(phase))
        return best

    start = (cost(bpm, coarse_origin), bpm, coarse_origin)
    # Near the tracker's tempo first (±3 %). Beat trackers are sometimes 5-8 % off (121 for a song at
    # 130), so also look wider (±10 %), but a far tempo is only accepted when it aligns the onsets
    # clearly better: on a rubato performance the alignment cost is flat and the tracker, which
    # listens to the whole mix, is the better witness.
    near = search(bpm, 0.03, 121, start)
    wide = search(bpm, 0.10, 401, start)
    best = wide if wide[0] < 0.7 * near[0] else near
    best = search(best[1], 0.002, 41, best)  # pin the winner to ~0.1 bpm
    return best[1], best[2]


def _clean_and_quantize(notes, to_beats):
    """Snap notes to the beat grid (`to_beats`: seconds -> beats) and force a single, playable
    melodic line."""
    quantized = []
    for n in notes:
        n.slot = max(0, int(round(to_beats(n.start) / GRID)))
        end_slot = int(round(to_beats(n.end) / GRID))
        n.length = max(1, end_slot - n.slot)
        quantized.append(n)

    # Monophonic: one note per grid slot. Notes that start together (< 40 ms apart) are a double
    # stop or an octave ghost: keep the loudest (tie -> highest pitch). A note that starts clearly
    # later but rounds to the same slot is a real, fast note: it takes the next free slot instead
    # of vanishing (the run is then written in sixteenths, which is what it is).
    best = {}
    for n in quantized:
        cur = best.get(n.slot)
        if cur is None:
            best[n.slot] = n
        elif abs(n.start - cur.start) < 0.04:
            if (n.amplitude, n.pitch) > (cur.amplitude, cur.pitch):
                best[n.slot] = n
        elif (n.slot + 1) not in best:
            n.slot += 1
            best[n.slot] = n
        elif n.amplitude > cur.amplitude:
            best[n.slot] = n
    mono = [best[s] for s in sorted(best)]

    # No overlaps: a note stops when the next one starts.
    for cur, nxt in zip(mono, mono[1:]):
        cur.length = max(1, min(cur.length, nxt.slot - cur.slot))

    return mono


SLOTS_PER_BEAT = int(round(1 / GRID))
SLOTS_PER_BAR = 4 * SLOTS_PER_BEAT  # 4/4


def _align_to_downbeats(mono, first_downbeat_slot: int) -> int:
    """
    Bar lines from the tracked downbeats: shift the slots so that downbeats fall on multiples of
    SLOTS_PER_BAR, then start measure 1 on the bar of the first real note. Returns the shift.
    """
    if not mono:
        return 0
    threshold = 0.6 * max(n.amplitude for n in mono)
    first = next((n for n in mono if n.amplitude >= threshold), mono[0])
    first_bar = (first.slot - first_downbeat_slot) // SLOTS_PER_BAR
    shift = -(first_downbeat_slot + first_bar * SLOTS_PER_BAR)
    lowest = min(n.slot for n in mono) + shift
    if lowest < 0:  # notes before the first real one (bleed / pickup): whole bars in front
        shift -= (lowest // SLOTS_PER_BAR) * SLOTS_PER_BAR
    for n in mono:
        n.slot += shift
    return shift


def _align_downbeat(mono) -> int:
    """
    Fallback without tracked downbeats (see _align_to_downbeats).
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
