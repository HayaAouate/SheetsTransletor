# SheetsTranslator — MVP interne

Audio → partition (violon) ou tablature (guitare), à partir d'un fichier local ou d'un lien YouTube.

Pipeline : [Demucs](https://github.com/facebookresearch/demucs) (Meta, open source) isole l'instrument du mix →
[Basic Pitch](https://github.com/spotify/basic-pitch) (Spotify, open source) détecte les notes → détection de tempo
(librosa) + nettoyage/quantification → [music21](https://web.mit.edu/music21/) + [Lilypond](https://lilypond.org) pour l'engraving.

## Prérequis à installer une fois

1. **Python 3.10** (obligatoire : Basic Pitch exige TensorFlow < 2.15 sur 3.11+, qui n'existe pas pour 3.12).
   Téléchargement : https://www.python.org/downloads/ ou `winget install Python.Python.3.10`

2. **FFmpeg** (lecture des fichiers audio, téléchargement YouTube).
   Windows : `winget install Gyan.FFmpeg` (ou build sur https://www.gyan.dev/ffmpeg/builds/ + dossier `bin` dans le PATH).
   Vérifie avec : `ffmpeg -version`

3. **Lilypond** (génération des PDF de partition).
   Windows : `winget install LilyPond.LilyPond` (ou https://lilypond.org/download.html).
   Vérifie avec : `lilypond --version`

   (FFmpeg et Lilypond n'ont pas besoin d'être dans le PATH : l'app les cherche aussi dans les dossiers
   d'installation habituels — winget, Program Files — au démarrage.)

## Installation du projet

Dans un terminal, depuis ce dossier :

```bash
py -3.10 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

(~1,5 Go : Basic Pitch, Demucs et PyTorch CPU. Les modèles Demucs (~80 Mo chacun) sont téléchargés
automatiquement au premier usage, dans `~/.cache/torch/hub`.)

## Lancer l'application

```bash
.venv\Scripts\streamlit run app.py
```

Ça ouvre l'interface dans ton navigateur (http://localhost:8501).

## Utilisation

1. Choisis l'instrument (Violon → partition PDF / MusicXML ; Guitare → tablature PDF).
2. Choisis la source : fichier local (mp3/wav/m4a) ou lien YouTube.
3. Choisis la **piste à isoler** (Demucs) : « Autres » pour le violon, « Guitare », « Voix », « Piano », « Basse »…
   Sur une chanson complète c'est indispensable. « Aucune » seulement pour un enregistrement solo déjà propre.
4. Clique "Transcrire". **Écoute l'aperçu audio** de la transcription : s'il ne ressemble pas au morceau,
   la partition sera fausse aussi → change de piste isolée ou ajuste la sensibilité (réglages avancés).
5. Télécharge PDF / MusicXML / MIDI. Le MusicXML s'ouvre dans MuseScore (gratuit) pour corriger à la main.

## Limites connues de ce MVP

- **Monophonique** : une seule ligne mélodique (la note la plus forte à chaque instant). Pas d'accords.
- **Séparation** : le violon n'a pas de piste dédiée dans Demucs, il tombe dans « Autres » avec synthés, cordes
  d'accompagnement, etc. Sur un mix très chargé le résultat reste approximatif. Voix / guitare / piano / basse ont
  leur propre piste et ressortent bien mieux.
- **Rythme** : tempo constant détecté automatiquement (affiné sur les attaques des notes), 4/4, grille à la
  double-croche. Pas de triolets, pas de rubato ni de changements de tempo. La mesure 1 démarre sur le temps de la
  première note : un morceau qui commence par une levée sera décalé d'un temps (à corriger dans MuseScore).
- **Tablature guitare** : doigté par heuristique simple (position la plus proche de la note précédente).
- **Pas d'export Guitar Pro (.gp5)** — PDF (tab ASCII) et MIDI.

## Pistes d'amélioration pour la suite

- Détection des premiers temps (downbeats) et des changements de tempo.
- Support des triolets / signatures rythmiques autres que 4/4.
- Vrai algorithme de fingering guitare (minimisation du mouvement de la main sur plusieurs notes, pas juste la précédente).
- Export Guitar Pro via la librairie `PyGuitarPro`.
- Support d'autres instruments (piano, voix) — la brique `transcribe.py` est déjà générique, il "suffit" d'ajouter un chemin de rendu par instrument dans `notation.py`.

## Structure du projet

```
sheetsTransletor/
├── app.py                      # interface Streamlit
├── requirements.txt
├── transcriber/
│   ├── audio_input.py          # upload local + téléchargement YouTube (yt-dlp)
│   ├── separate.py             # isolation d'une piste du mix (Demucs)
│   ├── transcribe.py           # audio -> notes nettoyées et calées sur le tempo (Basic Pitch + librosa)
│   ├── notation.py             # notes -> partition PDF/MusicXML (violon, music21+Lilypond)
│   └── guitar_tabs.py          # notes -> tablature ASCII rythmique + PDF (guitare)
└── output/                     # fichiers générés (créé automatiquement)
```# SheetsTransletor
