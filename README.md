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
├── app.py                      # interface Streamlit (outil de dev ; sera remplacée par le front web)
├── requirements.txt
├── transcriber/                # le cœur métier : audio -> notes -> partition (ne dépend pas de l'API)
│   ├── audio_input.py          # upload local + téléchargement YouTube/TikTok (yt-dlp)
│   ├── separate.py             # isolation de la piste mélodique (Demucs, composite other/guitar/vocals)
│   ├── melody.py               # suivi de hauteur monophonique (CREPE + Viterbi, octaves, notes parasites)
│   ├── beats.py                # temps et premiers temps (Beat This!)
│   ├── transcribe.py           # notes -> rythme quantifié sur la grille des temps
│   ├── notation.py             # notes -> partition PDF/MusicXML (music21 + Lilypond)
│   ├── guitar_tabs.py          # notes -> tablature ASCII + PDF (guitare)
│   └── viewer.py               # lecteur de partition avec curseur synchronisé (OSMD)
├── api/                        # l'API HTTP (FastAPI) — voir api/app.py pour le détail
│   ├── app.py                  # l'application : middlewares, routes, threads de fond   (uvicorn api.app:app)
│   ├── settings.py             # TOUTES les variables d'environnement, en un seul endroit
│   ├── security.py             # la clé d'API (X-API-Key)
│   ├── schemas.py              # modèles des réponses (contrat avec le front, documenté dans /docs)
│   ├── routes/                 # health.py, transcriptions.py : une route = une fonction nommée par ce qu'elle fait
│   ├── storage/                # database.py (lignes SQLite), files.py (le dossier de chaque transcription)
│   └── workers/                # queue.py (un job à la fois), pipeline.py (la transcription), retention.py (nettoyage)
├── deploy/
│   ├── caddy/Caddyfile.local   # le Caddy de test sur le PC (profil `standalone`)
│   ├── caddy/Caddyfile.vps.snippet     # le bloc sheets.tracevault.tech à coller dans ~/app/Caddyfile
│   └── vps/docker-compose.service.yml  # le service à coller dans ~/app/docker-compose.yaml
├── Dockerfile, docker-compose.yml, docker-compose.vps.yml
├── data/                       # runtime (ignoré par git) : db.sqlite + un dossier par transcription
└── samples/                    # morceaux de test (jamais dans l'image Docker)
```

## API HTTP + déploiement Docker (VPS)

Le pipeline est exposé par une API (`api/`) : jobs asynchrones (un à la fois), historique en SQLite,
fichiers dans `data/<id>/`. Toutes les routes sauf `/health` exigent l'en-tête `X-API-Key` (clé dans `.env`,
bouton « Authorize » dans `/docs`). En local, hors Docker :

```bash
.venv\Scripts\uvicorn api.app:app --port 8000
```

puis http://localhost:8000/docs. Routes : `POST /transcriptions` (multipart : `file` ou `url`, `title`, `artist`,
`instrument` = Violin | Guitar), `GET /transcriptions` (historique), `GET /transcriptions/{id}` (statut,
progression, `timeline` pour le curseur, `files` disponibles), `GET /transcriptions/{id}/files/{kind}`
(pdf | musicxml | midi | preview | stem | original | timeline), `DELETE /transcriptions/{id}`.

Garde-fous (`.env`) : `MAX_UPLOAD_MB=50`, `MAX_AUDIO_MIN=10`, `MAX_PENDING=5`, `URL_HOSTS` (youtube, tiktok…).

### Stockage et nettoyage automatique

Une transcription de 3 min pèse ~10 Mo : l'audio (original mp3, piste isolée et aperçu en **FLAC**, 5× plus
petit que le WAV) fait tout le poids, la partition + MIDI + MusicXML + timeline < 100 Ko.
`api/workers/retention.py` tourne toutes les heures :

- job en erreur → supprimé après `ERROR_RETENTION_DAYS=2` ;
- job terminé non rouvert depuis `AUDIO_RETENTION_DAYS=30` → **audio supprimé, partition conservée**
  (l'historique et le PDF restent ; seule la lecture avec curseur demande de relancer) ;
- au-delà de `STORAGE_MAX_GB=5`, l'audio des moins récemment ouvertes est purgé en premier ;
- un job en attente ou en cours n'est jamais touché. Ouvrir une transcription (`GET /transcriptions/{id}`)
  remet son compteur à zéro (`last_opened_at`).

### Test sur le PC, à l'identique du VPS (Docker Desktop)

Le VPS a déjà un Caddy (`~/app/docker-compose.yaml`, réseau `publication_network`) qui sert les autres sites :
on ne lance pas de second Caddy là-bas, le conteneur `sheetstranslator-api` rejoint ce réseau et Caddy lui transmet
`sheets.tracevault.tech/api/*`. Sur le PC on reproduit exactement ça, avec un Caddy local en `:80` :

```bash
docker network create publication_network           # une fois
cp .env.example .env                                # décommenter COMPOSE_PATH_SEPARATOR, COMPOSE_FILE, PROXY_NETWORK
docker compose --profile standalone up -d --build   # API (aucun port publié) + Caddy local : http://localhost/api/docs
docker compose logs -f api
```

Vérifié : `sheetstranslator-api` est sur `publication_network` sans port publié, `http://localhost:8000` refuse,
`http://localhost/api/health` répond via Caddy ; `POST /api/transcriptions` avec `samples/IMG_4291.mp3`
→ 45 notes, 7 fichiers, 292 s pour 41 s d'audio (1er job : téléchargement des modèles inclus) ; avec
l'URL TikTok de référence → 103 notes, 290 s. Image : 3,55 Go.

### Mise en ligne sur le VPS (même modèle que les autres apps : image Docker Hub)

1. Sur le PC : `docker build -t miaouu/sheetstranslator-api:latest . && docker push miaouu/sheetstranslator-api:latest`
2. Sur le VPS, dans `~/app/docker-compose.yaml` : ajouter le service de `deploy/vps/docker-compose.service.yml`
   (+ le volume `sheetstranslator_models`).
3. Dans `~/app/Caddyfile` : ajouter le bloc de `deploy/caddy/Caddyfile.vps.snippet`, et un enregistrement DNS `A`
   `sheets.tracevault.tech` → IP du VPS.
4. `cd ~/app && docker compose pull sheetstranslator-api && docker compose up -d sheetstranslator-api && docker exec caddy caddy reload --config /etc/caddy/Caddyfile`

L'API répond sur `https://sheets.tracevault.tech/api/` (docs : `/api/docs`). Les transcriptions sont dans
`~/app/sheetstranslator/data` (à sauvegarder), les modèles dans le volume `sheetstranslator_models`. Aucun port à ouvrir :
seul Caddy (80/443) est exposé. Alternative sans Docker Hub : cloner le dépôt sur le VPS et
`docker compose up -d --build` avec le même `.env` que sur le PC (`docker-compose.vps.yml`).

Dimensionnement : une transcription prend 3–4 Go de RAM et tourne sur tous les cœurs ; sur un VPS 2 vCPU / 8 Go,
compter ~5–8× la durée du morceau. `mem_limit: 6g` dans `docker-compose.yml` protège le système.

### Temps de calcul : CPU vs GPU

Demucs (séparation) représente ~75 % du temps, CREPE ~15 %, le reste (Beat This!, Lilypond, synthèse) < 1 min.
Ce sont des réseaux de neurones : le gain avec plus de cœurs CPU est linéaire, un GPU va 10–20× plus vite.

| Machine | Titre de 3 min (estimation, à mesurer dans les logs `terminé en Xs`) |
|---|---|
| KVM 2 (2 vCPU) — actuel | 15–25 min |
| KVM 4 / KVM 8 | 8–12 min / 4–6 min |
| GPU d'entrée de gamme (T4, RTX 4090) | ~1 min |

Options, de la moins chère à la plus rapide :

1. **Rester en CPU sur KVM 2** : file d'attente, job en arrière-plan, historique. Pour les enregistrements solo
   (violon seul, sans accompagnement) choisir « sans séparation » évite Demucs → 2–3 min.
2. **GPU serverless (RunPod Serverless, Modal)** pour Demucs + CREPE uniquement, facturé à la seconde :
   < 1 centime par titre, 0 € quand personne n'utilise l'app, résultat en 1–2 min (démarrage à froid inclus).
   Le VPS garde l'API, l'historique, la file et le PDF ; une variable d'environnement (`GPU_ENDPOINT`) suffira,
   sans elle on retombe sur le CPU local.
3. **GPU Hostinger** (RTX 4090 0,42 $/h, L40S 0,96 $/h, A100 1,55 $/h, facturé à la minute) : **facturé tant que
   l'instance existe, même éteinte** — seule la destruction arrête les frais, et elle efface tout. ~300 $/mois
   pour une 4090 qui dort : à éviter pour un usage intermittent.

Décision : on déploie d'abord sur KVM 2 en CPU (ça marche, c'est juste lent), et on branche le GPU serverless
quand on veut passer à 1–2 min.
