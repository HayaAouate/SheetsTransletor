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
import soundfile as sf

# Label shown in the UI -> (demucs model, stem name or tuple of stems to add up).
# None = skip separation, use the full mix.
STEM_CHOICES = {
    "Aucune (mix complet)": None,
    # 6-stem model: piano and guitar get their own stems, so "other" is much closer to the violin
    # alone on a violin + piano cover (half as many spurious notes). Slower (~1.5x). The "guitar"
    # stem is added back: the model regularly files a violin phrase under guitar (whole bars of a
    # violin cover over a backing track went there), and a guitar cover is a separate choice.
    "Violon / cordes / vents (sans piano)": ("htdemucs_6s", ("other", "guitar")),
    "Autres — violon, cordes, vents, synthé, piano": ("htdemucs", "other"),
    "Voix": ("htdemucs", "vocals"),
    "Guitare": ("htdemucs_6s", "guitar"),
    "Piano": ("htdemucs_6s", "piano"),
    "Basse": ("htdemucs", "bass"),
}

# Sensible default stem per target instrument.
DEFAULT_STEM = {
    "Violin": "Violon / cordes / vents (sans piano)",
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
    out_path = os.path.join(out_dir, "+".join(wanted) + ".wav")
    stem_audio = sum(stems[s] for s in wanted).numpy().T  # (channels, T) -> (T, channels)
    sf.write(out_path, stem_audio, separator.samplerate)
    return out_path
