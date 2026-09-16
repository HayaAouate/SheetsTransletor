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

# Label shown in the UI -> (demucs model, stem name). None = skip separation, use the full mix.
STEM_CHOICES = {
    "Aucune (mix complet)": None,
    "Autres — violon, cordes, vents, synthé": ("htdemucs", "other"),
    "Voix": ("htdemucs", "vocals"),
    "Guitare": ("htdemucs_6s", "guitar"),
    "Piano": ("htdemucs_6s", "piano"),
    "Basse": ("htdemucs", "bass"),
}

# Sensible default stem per target instrument.
DEFAULT_STEM = {
    "Violin": "Autres — violon, cordes, vents, synthé",
    "Guitar": "Guitare",
}

log = logging.getLogger("sheets.separate")

_separators = {}


def _get_separator(model_name: str):
    """Load a Demucs model once per process (loading takes a few seconds)."""
    if model_name not in _separators:
        from demucs.api import Separator

        log.info("Chargement du modèle Demucs %s (téléchargé au premier usage)", model_name)
        _separators[model_name] = Separator(model=model_name, device="cpu", progress=False)
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
    if stem not in stems:
        raise RuntimeError(f"Le modèle {model_name} n'a pas de piste '{stem}' (dispo : {list(stems)}).")

    out_dir = tempfile.mkdtemp()
    out_path = os.path.join(out_dir, f"{stem}.wav")
    stem_audio = stems[stem].numpy().T  # (channels, T) -> (T, channels)
    sf.write(out_path, stem_audio, separator.samplerate)
    return out_path
