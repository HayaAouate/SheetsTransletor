"""Source separation with Demucs (Meta, open source): isolate one instrument from a full mix
before transcription. Without this step, Basic Pitch transcribes voice + bass + drum harmonics
+ everything else into one unreadable pile of notes.

Models (downloaded automatically on first use, cached in ~/.cache/torch/hub):
  - htdemucs     (4 stems): drums, bass, other, vocals            ~80 MB
  - htdemucs_6s  (6 stems): + guitar, piano                       ~80 MB
"""
import logging
import os
import tempfile

import librosa
import numpy as np
import soundfile as sf

MELODIC = ("other", "guitar", "vocals")  # stems a melodic instrument may land in, see below

# Label shown in the UI -> (demucs model, stem name, tuple of stems to add up, or MELODIC).
# None = skip separation, use the full mix.
STEM_CHOICES = {
    "Aucune (mix complet)": None,
    # 6-stem model: piano gets its own stem, so a violin + piano cover loses the piano. Demucs has
    # no idea what a violin is, though: from one song to the next, and within a song, it files it
    # under "other", "guitar" (whole bars of a cover over a backing track) or "vocals" (a cover of
    # a sung melody). Adding the three up does not work either (a distorted guitar drone drowns
    # the line), so MELODIC picks, every 2 s, the one of the three that carries the most voiced
    # energy (see _melodic_composite). Slower than the 4-stem model (~1.5x, + the scoring).
    "Violon / cordes / vents (piste mélodique auto)": ("htdemucs_6s", MELODIC),
    "Autres — violon, cordes, vents, synthé, piano": ("htdemucs", "other"),
    "Voix": ("htdemucs", "vocals"),
    "Guitare": ("htdemucs_6s", "guitar"),
    "Piano": ("htdemucs_6s", "piano"),
    "Basse": ("htdemucs", "bass"),
}

# Sensible default stem per target instrument.
DEFAULT_STEM = {
    "Violin": "Violon / cordes / vents (piste mélodique auto)",
    "Guitar": "Guitare",
}

log = logging.getLogger("sheets.separate")

_separators = {}


def _get_separator(model_name: str):
    """Load a Demucs model once per process (loading takes a few seconds)."""
    if model_name not in _separators:
        from demucs.api import Separator

        log.info("Chargement du modèle Demucs %s (téléchargé au premier usage)", model_name)
        # shifts=0: Demucs otherwise applies a random time shift on each run, so the same audio
        # gives a slightly different stem every time (and every downstream result moves with it).
        _separators[model_name] = Separator(model=model_name, device="cpu", progress=False, shifts=0)
    return _separators[model_name]


def isolate_stem(audio_path: str, choice: str, progress_callback=None) -> str:
    """
    Return the path of a WAV containing only the requested stem of `audio_path`.
    `choice` is a key of STEM_CHOICES. Returns `audio_path` unchanged when no separation is asked.
    """
    spec = STEM_CHOICES.get(choice)
    if spec is None:
        return audio_path
    model_name, stem = spec
    log.info("Séparation Demucs : modèle=%s, piste=%s", model_name, stem)

    import torch

    separator = _get_separator(model_name)
    if progress_callback:
        separator.update_parameter(callback=progress_callback)

    # Load with librosa (robust to mp3/m4a/wav via ffmpeg/soundfile) at the model's sample rate.
    y, _ = librosa.load(audio_path, sr=separator.samplerate, mono=False)
    if y.ndim == 1:  # mono -> fake stereo, Demucs expects 2 channels
        y = librosa.util.stack([y, y], axis=0)
    wav = torch.from_numpy(y.astype("float32"))

    _, stems = separator.separate_tensor(wav, sr=separator.samplerate)
    wanted = (stem,) if isinstance(stem, str) else tuple(stem)
    missing = [s for s in wanted if s not in stems]
    if missing:
        raise RuntimeError(f"Le modèle {model_name} n'a pas de piste {missing} (dispo : {list(stems)}).")

    out_dir = tempfile.mkdtemp()
    if stem is MELODIC:
        out_path = os.path.join(out_dir, "melodic.wav")
        stem_audio = _melodic_composite({s: stems[s].numpy() for s in wanted}, separator.samplerate)
    else:
        out_path = os.path.join(out_dir, "+".join(wanted) + ".wav")
        stem_audio = sum(stems[s] for s in wanted).numpy().T  # (channels, T) -> (T, channels)
    sf.write(out_path, stem_audio, separator.samplerate)
    return out_path


def _melodic_composite(stems: dict, sr: int, window_s: float = 2.0, hysteresis: float = 1.5, fade_s: float = 0.1,
                       silence: float = 0.3):
    """
    Assemble one track from several stems, taking in each window of `window_s` the stem that
    carries the most *voiced energy* (energy of the frames where a pitch tracker is confident):
    a melodic instrument is loud AND pitched there, a drum residue is loud but not pitched, an
    empty stem is neither. CREPE "tiny" at 20 ms does the scoring (~0.1x real time on CPU). The
    previous window's stem is kept unless another beats it by `hysteresis`, and windows are joined
    with a short crossfade. Input: {name: (channels, T)}; returns (T, channels).
    """
    import torch
    import torchcrepe
    from scipy.signal import butter, sosfiltfilt

    # Candidates: each stem alone, plus "other" with half of "vocals" added. Demucs can spread one
    # violin phrase over two stems note by note (the F's in "other", the G's in "vocals"): only
    # the sum has the whole line there. Vocals at half level: enough for the notes "other" lacks
    # (nothing competes with them at that instant), not enough for the octave-below doubling that
    # stem often carries to win over the line.
    stems = dict(stems)
    if "other" in stems and "vocals" in stems:
        stems["other+vocals/2"] = stems["other"] + 0.5 * stems["vocals"]
    names = list(stems)
    hop, crepe_sr = 320, 16000  # 20 ms
    scores = []
    sos = butter(4, 166, "hp", fs=crepe_sr, output="sos")  # below a violin's G3: bass/kick residue
    with torch.no_grad():
        for name in names:
            mono = stems[name].mean(axis=0)
            y = librosa.resample(mono, orig_sr=sr, target_sr=crepe_sr)
            y = sosfiltfilt(sos, y).astype(np.float32)
            _, conf = torchcrepe.predict(torch.from_numpy(y)[None], crepe_sr, hop, fmin=196, fmax=2637, model="tiny",
                                         batch_size=1024, device="cpu", return_periodicity=True,
                                         decoder=torchcrepe.decode.argmax)
            conf = conf[0].numpy()
            rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0][:len(conf)]
            scores.append((conf[:len(rms)] > 0.5) * rms)
    scores = np.stack(scores)  # (stems, frames)
    frames_per_win = max(1, int(window_s * crepe_sr / hop))
    n_win = int(np.ceil(scores.shape[1] / frames_per_win))
    win_scores = np.stack([scores[:, i * frames_per_win:(i + 1) * frames_per_win].mean(axis=1) for i in range(n_win)], axis=1)

    # Where the instrument is silent, every stem scores low and the best one is just whatever
    # accompaniment leaked into it: below `silence` x the piece's melodic level, output silence.
    peak = win_scores.max(axis=0)
    level = float(np.percentile(peak[peak > 0], 75)) if (peak > 0).any() else 0.0
    choice, current = [], int(np.argmax(win_scores[:, 0]))
    for i in range(n_win):
        best = int(np.argmax(win_scores[:, i]))
        if best != current and win_scores[best, i] > hysteresis * win_scores[current, i]:
            current = best
        choice.append(current if peak[i] >= silence * level else None)
    log.info("Piste mélodique : %s", " ".join("---" if c is None else names[c].replace("other", "oth").replace("vocals", "voc").replace("guitar", "gui") for c in choice))

    # Build the composite with crossfades at the switches
    n_ch, n_samples = stems[names[0]].shape
    win_samples = int(window_s * sr)
    fade = int(fade_s * sr)
    zeros = np.zeros((n_ch, n_samples), dtype=np.float32)
    out = np.zeros((n_ch, n_samples), dtype=np.float32)

    def src(c):
        return zeros if c is None else stems[names[c]]

    for i, c in enumerate(choice):
        a, b = i * win_samples, min((i + 1) * win_samples, n_samples)
        out[:, a:b] = src(c)[:, a:b]
        if i and choice[i - 1] != c and a >= fade:
            ramp = np.linspace(0, 1, fade, dtype=np.float32)
            out[:, a - fade:a] = src(choice[i - 1])[:, a - fade:a] * (1 - ramp) + src(c)[:, a - fade:a] * ramp
    return out.T
