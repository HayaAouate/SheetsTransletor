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
- [x] **M5** Rythme lisible : fusion des micro-silences, pas de mesures vides en tête, choix binaire/ternaire des durées (l'écriture actuelle est trop hachée)
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
