"""Beat and downbeat tracking on the original mix, with Beat This! (CPJKU, 2024, PyTorch).

Bar lines are the part of a score a beat tracker cannot give: librosa finds a tempo and a beat
phase, but not which beat is the first of the bar, and it drifts on live playing. Beat This!
returns the time of every beat AND every downbeat, from the whole mix (drums and bass make it
reliable even when the melody plays off the beat), which gives:
  * bar lines where the music actually starts its bars, instead of a guess from the notes,
  * a beat map: the position of any instant in beats, by interpolation between the detected
    beats, so a performance that speeds up or slows down still lands on the grid.
"""
import logging

import numpy as np

log = logging.getLogger("sheets.beats")

_tracker = None


def track_beats(audio_path: str):
    """
    Return (beat_times, downbeat_times) in seconds as numpy arrays, or (None, None) when the
    tracker is unavailable or finds too little to trust (< 8 beats or no downbeat).
    """
    global _tracker
    try:
        if _tracker is None:
            from beat_this.inference import File2Beats

            # final0 = the released checkpoint (~77 MB, downloaded once into ~/.cache/torch/hub).
            # dbn=False: the raw network output, whose beats are cleaner than the DBN's on music
            # with a steady tempo (and needs no madmom).
            _tracker = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)
        beats, downbeats = _tracker(audio_path)
    except Exception as exc:  # optional dependency / model download failed: caller falls back
        log.warning("Beat This! indisponible (%s) : tempo librosa sans premiers temps", exc)
        return None, None
    beats, downbeats = np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float)
    if len(beats) < 8 or len(downbeats) == 0:
        log.warning("Beat This! : %d temps, %d premiers temps, pas assez pour une grille", len(beats), len(downbeats))
        return None, None
    per_bar = np.diff(np.searchsorted(beats, downbeats))
    log.info("Beat This! : %d temps, %d premiers temps, %s temps par mesure", len(beats), len(downbeats),
             int(np.median(per_bar)) if len(per_bar) else "?")
    return beats, downbeats


class BeatMap:
    """Seconds <-> beats, piecewise linear between detected beats (linear extrapolation outside)."""

    def __init__(self, beat_times: np.ndarray):
        self.beat_times = np.asarray(beat_times, dtype=float)
        self.index = np.arange(len(self.beat_times), dtype=float)
        self.period = float(np.median(np.diff(self.beat_times)))

    @property
    def bpm(self) -> float:
        return 60.0 / self.period

    def to_beats(self, t) -> float:
        t = float(t)
        if t < self.beat_times[0]:
            return (t - self.beat_times[0]) / self.period
        if t > self.beat_times[-1]:
            return self.index[-1] + (t - self.beat_times[-1]) / self.period
        return float(np.interp(t, self.beat_times, self.index))

    def to_seconds(self, beats) -> float:
        b = float(beats)
        if b < 0:
            return self.beat_times[0] + b * self.period
        if b > self.index[-1]:
            return self.beat_times[-1] + (b - self.index[-1]) * self.period
        return float(np.interp(b, self.index, self.beat_times))
