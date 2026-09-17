# SheetsTranslator — Backlog

Objectif : une app équivalente à [songscription.ai](https://songscription.ai) — audio → partition
lisible, jouable et vérifiable à l'oreille, pour violon (partition) et guitare (tablature).

## Fonctionnalités relevées chez songscription (16/09/2026)

Relevé sur le site + l'app (captures) :

| Zone | Fonctionnalité |
|---|---|
| Entrée | Upload fichier · lien YouTube / Instagram / TikTok · **enregistrement micro** · drag & drop |
| Instruments | piano, guitare, basse, violon, flûte, trompette, sax, batterie, voix · « piano arrangement » de n'importe quel morceau |
| Sortie | partition · piano roll interactif · tablature guitare (doigtés) · export PDF, MIDI, MusicXML, GuitarPro |
| Partition | armure + chiffrage auto · **accords** (F, Dm, Am…) · tempo affiché · nuances / articulations · crédit « Transcribed by … » |
| Lecture | **playback de la partition avec curseur** · bascule **Transcribed / Original** (même curseur sur l'audio d'origine) · vitesse 0.5×–2× · métronome · transposition (Pitch) · zoom |
| Édition | mode Edit (corriger les notes) · Difficulty : Original / simplifié · Add lyrics · Re-transcribe |
| Organisation | liste des transcriptions · Recents · Collections · Search · Share (lien) · note « How did we do ? » |
| Compte | Trial 20 min · Upgrade · Feedback |

## Priorisation MoSCoW

### Must (le produit n'a pas de sens sans)
- [x] **M1** Audio (fichier / lien) → notes → partition PDF violon / tablature guitare (Demucs + Basic Pitch + music21 + Lilypond)
- [x] **M2** Armure correcte et orthographe des notes cohérente (bémols dans les tonalités en bémols…)
- [x] **M3** Titre / artiste saisis dans l'app, crédit « Violon d'or » sur la partition
- [x] **M4** **Affichage de la partition dans l'app + lecture avec curseur** (OpenSheetMusicDisplay + audio synthétisé), bascule *Transcription* / *Original* sur le même curseur, contrôle de vitesse
- [x] **M5** Rythme lisible : fusion des micro-silences, pas de mesures vides en tête, durées standard — **sans jamais supprimer une note** (fidélité d'abord : grille croche quand la place est libre, double-croche sinon)
- [x] **M6** Export MusicXML / MIDI / PDF depuis la vue (déjà là) + nom de fichier = titre du morceau

### Should (fort impact, faisable vite)
- [ ] **S1** Accords au-dessus de la portée (analyse chroma sur le mix original, 1 accord / mesure ou / temps)
- [ ] **S2** Vue tablature guitare rendue dans OSMD (MusicXML avec `<technical><string>/<fret>`) au lieu du PDF ASCII
- [ ] **S3** Historique des transcriptions (sqlite + dossier `data/`) : liste « Recents », ré-ouvrir sans recalculer, re-transcrire avec d'autres réglages
- [ ] **S4** Métronome et compte à rebours avant lecture ; boucle A-B sur une plage de mesures (mode Practice)
- [ ] **S5** Transposition (Pitch ±12) à l'affichage et à l'export
- [ ] **S6** Enregistrement micro dans l'app (`st.audio_input`)

### Could (différenciant, plus tard)
- [ ] **C1** Difficulty : version simplifiée (ôter les ornements, quantifier à la croche, limiter la tessiture)
- [ ] **C2** Mode Edit : corriger une note (hauteur / durée) dans la vue et ré-exporter
- [ ] **C3** Piano roll interactif (notes qui tombent), utile pour non-lecteurs
- [ ] **C4** Autres instruments (flûte, trompette, sax, voix) : tessitures + stems Demucs adaptés
- [ ] **C5** Paroles (Add lyrics) alignées sous les notes
- [ ] **C6** Partage par lien, collections, recherche
- [ ] **C7** Note de qualité « How did we do ? » + feedback stocké

### Won't (pas dans ce MVP)
- Piano à deux mains / polyphonie, batterie, GuitarPro, comptes utilisateurs / paiement, app mobile.

## Liste de tâches (ordre d'exécution)

Chaque tâche = une livraison testable dans l'app. Je coche au fur et à mesure, tu testes.

1. [x] **M4-a** Vue partition dans l'app : composant OSMD (MusicXML → SVG dans la page, zoom auto)
2. [x] **M4-b** Lecture : bouton Play/Pause, audio synthétisé (WAV déjà produit), curseur OSMD synchronisé sur le temps audio
3. [x] **M4-c** Bascule *Transcription* / *Original* (audio d'origine ou piste isolée) avec le même curseur ; vitesse 0.5×–1.5×
4. [x] **M4-d** Clic sur une mesure → la lecture repart de là
5. [x] **M5-a** Rythme : combler les silences < 1 double-croche en allongeant la note précédente ; supprimer les mesures vides en tête
6. [x] **M5-b** Durées écrites lisibles (pas de « noire pointée pointée ») : arrondi aux durées standard + liaisons
7. [x] **M6** Noms de fichiers = titre, export depuis la vue
8. [ ] **S1** Accords
9. [ ] **S2** Tablature dans OSMD
10. [ ] **S3** Historique / Recents
11. [ ] **S4** Métronome + boucle A-B
12. [ ] **S5** Transposition
13. [ ] **S6** Enregistrement micro

## Précision de la transcription — référence Songscription (16/09/2026)

Référence : `samples/tiktok_lac0v.mp3` (On the Floor, cover violon @lac0v) et sa transcription Songscription
`samples/tiktok_lac0v.songscription.json` (leur JSON `inscript/1`, lisible sans compte sur `/api/score/<base64>`).
Comparaison mesure par mesure : `.venv/Scripts/python tools/compare_ref.py samples/tiktok_lac0v.mp3 melody`.

Ce qu'on a appris en lisant leur front : pipeline en deux jobs (A2M audio→MIDI, puis M2S MIDI→partition),
tracker de temps **et de premiers temps** (`/api/beats`, éditable), modèles par instrument, sortie ~97 % sur la
grille croche, aucune nuance par note, un accord par mesure. Leur violon écrit la phrase grave **une octave
au-dessus** de ce qui est joué (Basic Pitch et CREPE sont d'accord entre eux) : la comparaison ignore l'octave.

Fait :
- [x] **A2M mono** `melody.py` : CREPE (torchcrepe) + Viterbi maison en deux passes (a priori de registre) +
      segmentation aux attaques, choix d'octave par continuité mélodique, fusion des fragments à attaque faible.
      Réglage « Détection des notes » dans l'app (défaut violon). Basic Pitch reste pour la guitare / polyphonie.
- [x] **Temps + premiers temps** `beats.py` : Beat This! sur le mix → grille par interpolation entre les temps
      (dérive de tempo absorbée) + barres de mesure sur les downbeats ; repli librosa quand les attaques
      tombent mieux sur sa grille (violon seul, rubato).
- [x] **Rythme** : affectation des attaques à la grille croche par programmation dynamique (cases consécutives,
      ordre conservé) ; une note jouée plus courte qu'une double garde sa position (croche pointée + double).
- [x] Demucs déterministe (`shifts=0`) ; stem violon = `other + guitar` du modèle 6 pistes (le modèle range des
      phrases de violon dans « guitar ») ; cache disque des probabilités CREPE.
- Résultat : mesures 5–12 de la référence (la phrase grave, deux fois) **identiques** (hauteurs + rythme) ;
  section aiguë : bonnes classes de hauteur, octave des `Bb` tirée vers le bas par la nappe synthé sur Sib3.

À faire :
- [ ] Octave des notes dont la nappe joue la même classe (Bb4/Bb5 à 21–36 s) : évidence harmonique (Basic Pitch
      voit les deux) à combiner avec CREPE, ou modèle par instrument.
- [x] Curseur du lecteur et aperçu synthétisé sur la carte des temps (`Transcription.beat_times`) : sur un
      cover de 61 s, écart curseur/audio médian 225 ms → 30 ms, plus de dérive (+372 ms en fin de morceau avant).
- [ ] CREPE « full » ≈ 3× la durée du morceau sur CPU : tester `tiny`, ou n'analyser que les zones voisées.
- [ ] Intro : Songscription transcrit la pulsation synthé (mesures 1–4) ; nous non (le stem l'exclut). À décider.
