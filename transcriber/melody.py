"""Monophonic melody extraction: audio (ideally an isolated stem) -> note events, for one-voice
instruments (violin, flute, voice...).

Basic Pitch is a polyphonic transcriber: on a single melodic line it produces octave ghosts, misses
the quick notes of a run and cuts a sustained note wherever the accompaniment hits. A pitch tracker
does much better on one voice:

  1. CREPE (torchcrepe, "full" model) gives, every 10 ms, a probability over 360 pitch bins
     (20 cents each).
  2. Our own Viterbi decoding picks the pitch path through those bins. Staying on the same pitch
     is free, moving to a neighbouring bin (vibrato, glide) is cheap, jumping anywhere else (a real
     melodic leap) costs a fixed penalty: a two-frame octave flicker cannot pay for two jumps, a
     genuine leap that lasts can. (torchcrepe's own viterbi returns zeros with current librosa.)
  3. Frames are voiced when CREPE is confident AND the stem has energy there; the path is cut into
     notes where the (median) pitch changes by more than half a semitone, and additionally at
     onsets detected on the stem's energy, so a repeated note (Eb Eb Eb) is not one long note.
"""
import hashlib
import logging
import os
import tempfile

import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt

log = logging.getLogger("sheets.melody")

SR = 16000  # CREPE's sample rate
HOP = 160  # 10 ms
CENTS_PER_BIN = 20
BIN0_CENTS = 1997.3794084376191  # torchcrepe: bin i is at BIN0 + 20*i cents above 10 Hz


def _bin_to_midi(b):
    hz = 10 * 2 ** ((BIN0_CENTS + CENTS_PER_BIN * b) / 1200)
    return librosa.hz_to_midi(hz)


def extract_melody(
    audio_path: str,
    midi_low: int = 55,
    midi_high: int = 100,
    min_note_ms: float = 60.0,
    voicing: float = 0.35,
    jump_penalty: float = 8.0,
    register_weight: float = 0.4,
):
    """
    Return a list of (start_s, end_s, midi, amplitude, attack) tuples, time-sorted and
    non-overlapping; amplitude and attack are 0..1.

    midi_low/high: playable range of the instrument; bins outside are never chosen.
    min_note_ms:   shorter pitch segments are merged into their neighbour (vibrato / transitions).
    voicing:       CREPE confidence below which a frame is a rest.
    jump_penalty:  Viterbi cost (in log-prob units) of moving to a non-adjacent bin.
    register_weight: per-frame cost, per semitone beyond a fifth from the local register (2nd pass).
    """
    import torch
    import torchcrepe

    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    if len(y) < SR // 2:
        return []
    # Bass / kick / pad below the instrument's range confuse CREPE far more than losing the lowest
    # fundamentals does (the harmonics are still there): high-pass a bit under the lowest note.
    sos = butter(4, 0.85 * librosa.midi_to_hz(midi_low), "hp", fs=SR, output="sos")
    y = sosfiltfilt(sos, y).astype(np.float32)

    # 1. CREPE frame probabilities (T, 360) — the slow part (~3x real time on CPU), cached on disk
    #    so that re-transcribing the same audio with other settings is instant.
    cache = os.path.join(tempfile.gettempdir(), "sheets_crepe", hashlib.sha1(y.tobytes()).hexdigest()[:16] + ".npy")
    if os.path.exists(cache):
        probs = np.load(cache)
    else:
        probs = []
        with torch.no_grad():
            for frames in torchcrepe.preprocess(torch.from_numpy(y)[None], SR, HOP, batch_size=512, device="cpu", pad=True):
                probs.append(torchcrepe.infer(frames, model="full").numpy())
        probs = np.concatenate(probs, axis=0).astype(np.float16)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, probs)
    probs = probs.astype(np.float32)
    n_frames = probs.shape[0]
    times = np.arange(n_frames) * HOP / SR

    # 2. Energy and onsets of the stem
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP, center=True)[0][:n_frames]
    if len(rms) < n_frames:
        rms = np.pad(rms, (0, n_frames - len(rms)))
    floor = np.percentile(rms, 10)  # the stem's noise floor
    loud = rms > max(3 * floor, 0.02 * rms.max())
    onset_env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=SR, hop_length=HOP, units="frames", backtrack=False,
        pre_max=3, post_max=3, pre_avg=10, post_avg=10, delta=0.15, wait=6,
    )
    onset_set = set(int(f) for f in onset_frames)
    onset_env = onset_env[:n_frames] / (np.percentile(onset_env, 95) or 1.0)

    # 3. Viterbi over the bins of the instrument's range, twice. The accompaniment that leaks into
    #    the stem plays chord tones an octave or two under the melody, and CREPE jumps to them
    #    between two melody notes. The first pass gives the melody's register over time (from the
    #    notes long enough to be trusted); the second pass adds a prior that keeps the path within
    #    a fifth of that register unless the evidence is overwhelming.
    bins = np.arange(360)
    midi_of_bin = _bin_to_midi(bins)
    keep = (midi_of_bin >= midi_low - 0.5) & (midi_of_bin <= midi_high + 0.5)
    lo, hi = int(np.argmax(keep)), 360 - int(np.argmax(keep[::-1]))
    emission = probs[:, lo:hi]
    emission = emission / np.maximum(emission.sum(axis=1, keepdims=True), 1e-9)
    log_emission = np.log(emission + 1e-6)
    min_frames = max(1, int(min_note_ms / 10))

    def decode(log_em):
        path = _viterbi(log_em, jump_penalty)
        midi_frames = midi_of_bin[lo + path]
        confidence = probs[np.arange(n_frames), lo + path]  # raw CREPE probability of the chosen bin
        voiced = _close_gaps((confidence >= voicing) & loud, max_len=3)  # 30 ms dropout != rest
        notes = _segment(midi_frames, voiced, rms, onset_set, times, min_frames=min_frames)
        return notes, voiced

    notes, voiced = decode(log_emission)
    register = _register_curve(notes, times)
    if register is not None:
        prior = -register_weight * np.maximum(0.0, np.abs(midi_of_bin[lo:hi][None, :] - register[:, None]) - 7.0)
        notes, voiced = decode(log_emission + prior)

    # 4. Octave of each note from the melodic line, then re-join the fragments of one note.
    #    Attack strength (0..1) = peak of the onset envelope at the note start: a re-bowed note has
    #    a clear one, a fragment cut by a pitch wobble or a dropout has not.
    notes = _fix_octaves(notes, probs, midi_of_bin, midi_low, midi_high)
    notes = [(s, e, p, a, fs, fe, float(np.clip(onset_env[max(0, fs - 2):fs + 3].max(), 0, 1)))
             for s, e, p, a, fs, fe in notes]
    notes = _merge_fragments(notes, onset_set)
    notes = [(s, e, p, a, attack) for s, e, p, a, _fs, _fe, attack in notes]
    log.info("CREPE : %d notes (%.0f%% de trames voisées)", len(notes), 100 * voiced.mean())
    return notes


def _register_curve(notes, times, min_s=0.15, window_s=2.0):
    """Melodic register over time: for each frame, the median pitch (weighted by duration x
    amplitude) of the trusted notes (>= `min_s`) within +-`window_s`; NaN-free, None if no note."""
    trusted = [n for n in notes if n[1] - n[0] >= min_s]
    if not trusted:
        return None
    starts = np.array([n[0] for n in trusted])
    pitches = np.array([n[2] for n in trusted], dtype=float)
    weights = np.array([(n[1] - n[0]) * n[3] for n in trusted])
    order = np.argsort(pitches)
    pitches, weights, starts = pitches[order], weights[order], starts[order]
    # Evaluate every 0.5 s and interpolate: cheap and smooth enough.
    grid = np.arange(0, times[-1] + 0.5, 0.5)
    values = np.full(len(grid), np.nan)
    for i, t in enumerate(grid):
        m = np.abs(starts - t) <= window_s
        if m.any():
            cum = np.cumsum(weights[m])
            values[i] = pitches[m][min(int(np.searchsorted(cum, cum[-1] / 2)), int(m.sum()) - 1)]
    ok = ~np.isnan(values)
    if not ok.any():
        return None
    return np.interp(times, grid[ok], values[ok])


def _leap_cost(semitones: int, step_cost=0.12, leap_cost=0.45, easy=7) -> float:
    """Cost of a melodic interval: cheap up to a fifth, expensive beyond (a melody moves by steps
    and small leaps; a 17-semitone jump is almost always the tracker changing octave)."""
    return step_cost * min(semitones, easy) + leap_cost * max(semitones - easy, 0)


def _fix_octaves(notes, probs, midi_of_bin, midi_low, midi_high, min_ratio=0.1):
    """
    When the accompaniment shares a pitch class with the melody, CREPE hesitates between octaves
    (Bb4 or Bb5?) and sometimes picks the wrong one for a note in the middle of a phrase. For each
    note, every octave whose CREPE mass is at least `min_ratio` of the best one is a candidate; the
    octaves are then chosen jointly along the line (Viterbi) to keep the melody stepwise: the
    evidence for an octave (log of its mass relative to the best) minus the cost of the leap from
    the previous note (see _leap_cost).
    """
    if not notes:
        return notes
    sel = {}  # midi -> boolean mask of bins within half a semitone

    def mass(start, end, midi):
        if midi not in sel:
            sel[midi] = np.abs(midi_of_bin - midi) <= 0.5
        return float(probs[start:end][:, sel[midi]].sum(axis=1).mean())

    cands = []
    for s, e, p, _a, fs, fe in notes:
        opts = {}
        for q in (p - 12, p, p + 12):
            if midi_low <= q <= midi_high:
                opts[q] = mass(fs, fe, q)
        best = max(opts.values()) or 1e-9
        cands.append({q: np.log(max(m, 1e-9) / best) for q, m in opts.items() if m >= min_ratio * best})

    # Viterbi over notes: score = evidence - leap cost
    score = [dict(cands[0])]
    back = [{}]
    for i in range(1, len(cands)):
        cur, bk = {}, {}
        for q, ev in cands[i].items():
            prev_q = max(score[-1], key=lambda pq: score[-1][pq] - _leap_cost(abs(q - pq)))
            cur[q] = score[-1][prev_q] - _leap_cost(abs(q - prev_q)) + ev
            bk[q] = prev_q
        score.append(cur)
        back.append(bk)
    q = max(score[-1], key=score[-1].get)
    chosen = [q]
    for i in range(len(cands) - 1, 0, -1):
        q = back[i][q]
        chosen.append(q)
    chosen.reverse()
    return [(s, e, q, a, fs, fe) for (s, e, _p, a, fs, fe), q in zip(notes, chosen)]


def _merge_fragments(notes, onset_set, max_gap_s=0.15, weak_attack=0.6, blip_s=0.1):
    """
    Re-join what the segmentation broke:
      * two notes of the same pitch, contiguous (<= 60 ms apart) with no onset detected between
        them: a vibrato that strayed a semitone and came back (its spectral flux looks like a
        weak attack, so the attack strength alone would not tell);
      * two notes of the same pitch, close together (<= `max_gap_s`), the second one without a
        clear attack: one note with a dropout (a soft start before the bow bites);
      * a note shorter than `blip_s` without a clear attack: a finger or tracking accident in the
        middle of the previous note, which goes on. (A real short note, an ornament or a passing
        sixteenth, is bowed: its attack is clear and it is kept.)
    """
    if not notes:
        return notes
    merged = [list(notes[0])]
    for n in notes[1:]:
        prev = merged[-1]
        s, e, p, a, fs, fe, attack = n
        gap = s - prev[1]
        contiguous = gap <= 0.06 and not any(prev[5] < f <= fs for f in onset_set)
        same = p == prev[2] and (contiguous or (gap <= max_gap_s and attack < weak_attack))
        blip = e - s < blip_s and attack < 0.7
        if same or blip:
            if same or p == prev[2]:
                prev[1], prev[3], prev[5] = e, max(prev[3], a), fe
            else:  # different pitch: the previous note simply lasts through the blip
                prev[1], prev[5] = e, fe
        else:
            merged.append(list(n))
    return [tuple(m) for m in merged]


def _viterbi(log_emission: np.ndarray, jump_penalty: float) -> np.ndarray:
    """Best bin path: staying costs 0, moving one bin costs a little, any other move `jump_penalty`."""
    n_frames, n_bins = log_emission.shape
    near = 0.5  # cost of drifting to the adjacent bin (20 cents) — vibrato and glides are cheap
    score = log_emission[0].copy()
    back = np.zeros((n_frames, n_bins), dtype=np.int16)
    idx = np.arange(n_bins)
    for t in range(1, n_frames):
        # For each destination bin, the best source is one of: itself, a neighbour, or the global
        # best (paying the jump). Vectorized: no n_bins^2 matrix.
        stay = score
        left = np.concatenate(([-np.inf], score[:-1])) - near
        right = np.concatenate((score[1:], [-np.inf])) - near
        best_src = int(np.argmax(score))
        jump = np.full(n_bins, score[best_src] - jump_penalty)
        options = np.stack([stay, left, right, jump])
        choice = np.argmax(options, axis=0)
        src = np.where(choice == 0, idx, np.where(choice == 1, idx - 1, np.where(choice == 2, idx + 1, best_src)))
        back[t] = src
        score = options[choice, idx] + log_emission[t]
    path = np.zeros(n_frames, dtype=int)
    path[-1] = int(np.argmax(score))
    for t in range(n_frames - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _close_gaps(mask: np.ndarray, max_len: int) -> np.ndarray:
    out = mask.copy()
    i = 0
    while i < len(mask):
        if not mask[i]:
            j = i
            while j < len(mask) and not mask[j]:
                j += 1
            if 0 < i and j < len(mask) and j - i <= max_len:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def _segment(midi_frames, voiced, rms, onset_set, times, min_frames: int):
    """Cut the voiced frames into notes of stable pitch; split long same-pitch stretches at onsets."""
    segments = []  # [start_frame, end_frame) with a stable rounded pitch
    i = 0
    n = len(midi_frames)
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i + 1
        ref = midi_frames[i]
        while j < n and voiced[j]:
            # A note continues while the pitch stays within 0.6 semitone of the note's running
            # median (vibrato is ±0.5 at most); a sustained departure starts a new note.
            window = midi_frames[max(i, j - 10):j]
            ref = float(np.median(window))
            if abs(midi_frames[j] - ref) > 0.6:
                # departure must last (2 frames) to count, otherwise it is a glitch
                if j + 1 < n and voiced[j + 1] and abs(midi_frames[j + 1] - ref) > 0.6:
                    break
            j += 1
        segments.append([i, j])
        i = j

    # Merge segments too short to be notes into the neighbour with the closest pitch
    def pitch_of(seg):
        return float(np.median(midi_frames[seg[0]:seg[1]]))

    changed = True
    while changed and len(segments) > 1:
        changed = False
        for k, seg in enumerate(segments):
            if seg[1] - seg[0] >= min_frames:
                continue
            prev_ = segments[k - 1] if k > 0 and segments[k - 1][1] == seg[0] else None
            next_ = segments[k + 1] if k + 1 < len(segments) and segments[k + 1][0] == seg[1] else None
            if prev_ is None and next_ is None:
                segments.pop(k)  # isolated blip
            else:
                p = pitch_of(seg)
                cand = [c for c in (prev_, next_) if c is not None]
                target = min(cand, key=lambda c: abs(pitch_of(c) - p))
                if target is prev_:
                    prev_[1] = seg[1]
                else:
                    next_[0] = seg[0]
                segments.pop(k)
            changed = True
            break

    # Split at onsets (repeated notes), keeping both halves long enough
    final = []
    for start, end in segments:
        cuts = sorted(f for f in onset_set if start + min_frames <= f <= end - min_frames)
        last = start
        for c in cuts:
            if c - last >= min_frames:
                final.append((last, c))
                last = c
        final.append((last, end))

    notes = []  # (start_s, end_s, midi, amplitude, start_frame, end_frame)
    levels = [float(np.mean(rms[s:e])) for s, e in final]
    # Amplitude 0..1 relative to the loud notes (90th percentile, so one accent does not dwarf the
    # rest), compressed by a square root like perceived loudness: the downstream filters compare
    # notes to "0.6 x the loudest" and expect a played line to sit in 0.5-1.
    top = float(np.percentile(levels, 90)) if levels else 1.0
    for (start, end), level in zip(final, levels):
        pitch = int(round(float(np.median(midi_frames[start:end]))))
        amp = float(np.sqrt(min(1.0, level / (top or 1.0))))
        notes.append((float(times[start]), float(times[end - 1] + HOP / SR), pitch, amp, start, end))
    return notes
