"""SheetsTranslator — MVP interne : audio -> partition (violon) ou tablature (guitare).

Lancer avec :  streamlit run app.py
"""
import io
import logging
import os
import tempfile
import time

import numpy as np
import soundfile as sf
import streamlit as st

from transcriber.audio_input import get_audio_path
from transcriber.guitar_tabs import build_tab, export_tab_pdf, render_ascii_tab
from transcriber.notation import CREDIT, configure_lilypond, export_pdf, transcription_to_score
from transcriber.separate import DEFAULT_STEM, STEM_CHOICES, isolate_stem
from transcriber.transcribe import transcribe_audio

# Logs dans le terminal qui a lancé `streamlit run` : une ligne par étape du pipeline.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("sheets")

st.set_page_config(page_title="SheetsTranslator", page_icon="🎻")
st.title("🎻 SheetsTranslator — Audio → Partition")
st.caption("MVP interne — transcription violon (partition) et guitare (tablature) à partir d'un fichier ou d'un lien YouTube.")

configure_lilypond()  # nécessite Lilypond installé et accessible dans le PATH (voir README)

instrument_choice = st.radio("Instrument", ["Violon", "Guitare"], horizontal=True)
instrument_key = "Violin" if instrument_choice == "Violon" else "Guitar"

col_t, col_a = st.columns(2)
song_title = col_t.text_input("Titre du morceau", placeholder="Die On This Hill").strip()
song_artist = col_a.text_input("Artiste (optionnel)", placeholder="Sienna Spiro").strip()
source_type = st.radio("Source audio", ["Fichier local", "Lien YouTube"], horizontal=True)

audio_bytes = None
audio_name = None
youtube_url = None

if source_type == "Fichier local":
    uploaded = st.file_uploader("Fichier audio (mp3, wav, m4a)", type=["mp3", "wav", "m4a"])
    if uploaded is not None:
        audio_bytes = uploaded.getvalue()  # pas .read() : le curseur reste en fin de fichier entre deux reruns
        audio_name = uploaded.name
else:
    youtube_url = st.text_input("Lien YouTube (ou Instagram / TikTok)")

stem_labels = list(STEM_CHOICES)
stem_choice = st.selectbox(
    "Piste à isoler avant transcription (Demucs)",
    stem_labels,
    index=stem_labels.index(DEFAULT_STEM[instrument_key]),
    help="Sur une chanson complète, isoler l'instrument est indispensable : sinon voix, basse et batterie "
    "sont transcrites en même temps. Choisis « Aucune » seulement pour un enregistrement solo déjà propre.",
)

with st.expander("Réglages avancés"):
    onset_threshold = st.slider(
        "Sensibilité de détection des notes",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Plus bas = plus de notes détectées (plus de faux positifs). Plus haut = seulement les notes les plus sûres.",
    )
    min_note_ms = st.slider(
        "Durée minimale d'une note (ms)", min_value=30, max_value=250, value=80, step=10,
        help="Les notes plus courtes sont considérées comme du bruit et supprimées.",
    )

if st.button("Transcrire", type="primary"):
    if not audio_bytes and not youtube_url:
        st.error("Ajoute un fichier audio ou un lien YouTube avant de lancer la transcription.")
        st.stop()

    try:
        t0 = time.time()
        log.info("Transcription lancée — instrument=%s, source=%s, stem=%s, onset=%.2f, min_note=%sms",
                 instrument_key, youtube_url or audio_name, stem_choice, onset_threshold, min_note_ms)
        with st.spinner("Récupération de l'audio..."):
            if youtube_url:
                audio_path = get_audio_path(youtube_url, is_url=True)
            else:
                audio_path = get_audio_path(None, is_url=False, upload_bytes=audio_bytes, upload_name=audio_name)

        log.info("Audio prêt : %s (%.1fs)", audio_path, time.time() - t0)

        stem_path = audio_path
        if STEM_CHOICES[stem_choice] is not None:
            progress = st.progress(0.0, text="Séparation de sources (Demucs)... 1 à 3 min sur CPU pour un morceau entier.")

            def _on_progress(info):
                total = info.get("audio_length") or 1
                done = info.get("segment_offset", 0)
                progress.progress(min(done / total, 1.0), text="Séparation de sources (Demucs)...")

            stem_path = isolate_stem(audio_path, stem_choice, progress_callback=_on_progress)
            progress.progress(1.0, text="Séparation terminée.")
            log.info("Séparation terminée : %s (%.1fs)", stem_path, time.time() - t0)

        with st.spinner("Transcription en cours (Basic Pitch)... ça peut prendre 10-30s selon la durée."):
            tr = transcribe_audio(
                stem_path,
                instrument=instrument_key,
                onset_threshold=onset_threshold,
                minimum_note_length=float(min_note_ms),
                tempo_audio_path=audio_path,
            )

        log.info("Transcription terminée : %d notes, %.0f bpm (%.1fs)", len(tr.notes), tr.bpm, time.time() - t0)

        if not tr.notes:
            st.warning("Aucune note détectée. Essaie une autre piste à isoler, un enregistrement plus propre, ou baisse la sensibilité.")
            st.stop()

        # Aperçu audio de la transcription : si ça ne ressemble pas au morceau ici, la partition sera fausse aussi.
        st.subheader("Vérification à l'oreille")
        preview_midi = tr.to_pretty_midi(program=40 if instrument_key == "Violin" else 24)
        preview_wave = preview_midi.synthesize(fs=22050)
        preview_wave = preview_wave / (np.abs(preview_wave).max() or 1.0) * 0.8
        buf = io.BytesIO()
        sf.write(buf, preview_wave.astype(np.float32), 22050, format="WAV")
        st.audio(buf.getvalue(), format="audio/wav")
        if stem_path != audio_path:
            with st.expander("Écouter la piste isolée (ce que le modèle a vraiment transcrit)"):
                with open(stem_path, "rb") as f:
                    st.audio(f.read(), format="audio/wav")

        out_dir = tempfile.mkdtemp()
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Tempo détecté", f"{tr.bpm:.0f} bpm")
        col_c.metric("Notes", len(tr.notes))

        if instrument_choice == "Violon":
            with st.spinner("Génération de la partition..."):
                score, detected_key = transcription_to_score(
                    tr, title=song_title or "Transcription violon", artist=song_artist or None
                )
                col_b.metric("Tonalité", str(detected_key) if detected_key else "—")
                pdf_path = os.path.join(out_dir, "partition_violon.pdf")
                pdf_path, xml_path = export_pdf(score, pdf_path)

            log.info("Partition générée : %s (%.1fs)", pdf_path, time.time() - t0)
            st.success("Partition générée.")
            col1, col2, col3 = st.columns(3)
            with col1:
                with open(pdf_path, "rb") as f:
                    st.download_button("📄 Partition (PDF)", f, file_name="partition_violon.pdf")
            with col2:
                with open(xml_path, "rb") as f:
                    st.download_button("🎼 MusicXML (MuseScore)", f, file_name="partition_violon.musicxml")
            with col3:
                midi_path = os.path.join(out_dir, "transcription.mid")
                preview_midi.write(midi_path)
                with open(midi_path, "rb") as f:
                    st.download_button("🎹 MIDI", f, file_name="transcription.mid")

        else:  # Guitare
            with st.spinner("Génération de la tablature..."):
                assignments = build_tab(tr)
                ascii_tab = render_ascii_tab(assignments)
                pdf_path = os.path.join(out_dir, "tab_guitare.pdf")
                export_tab_pdf(
                    ascii_tab, pdf_path,
                    title=song_title or "Tablature guitare", subtitle=song_artist, credit=CREDIT,
                    info=f"{tr.bpm:.0f} bpm — 4/4 — une colonne = une double-croche",
                )
                col_b.metric("Tonalité", "—")

            log.info("Tablature générée : %s (%.1fs)", pdf_path, time.time() - t0)
            st.success("Tablature générée.")
            st.code(ascii_tab, language=None)
            col1, col2 = st.columns(2)
            with col1:
                with open(pdf_path, "rb") as f:
                    st.download_button("📄 Tablature (PDF)", f, file_name="tab_guitare.pdf")
            with col2:
                midi_path = os.path.join(out_dir, "transcription.mid")
                preview_midi.write(midi_path)
                with open(midi_path, "rb") as f:
                    st.download_button("🎹 MIDI", f, file_name="transcription.mid")

    except Exception as exc:  # MVP: on affiche l'erreur brute pour debug rapide
        log.exception("Échec de la transcription")
        st.error(f"Erreur pendant la transcription : {exc}")
        raise

st.divider()
with st.expander("Limites connues de ce MVP"):
    st.markdown(
        """
        - **Monophonique** : une seule ligne mélodique est écrite (la note la plus forte à chaque instant). Pas d'accords.
        - **Séparation de sources** (Demucs) : très efficace pour isoler voix / guitare / piano / basse. Le violon
          tombe dans la piste « Autres » avec tout ce qui n'est pas voix/basse/batterie — sur un morceau très chargé
          (synthés, cordes d'accompagnement), le résultat reste approximatif.
        - **Rythme** : tempo constant détecté automatiquement, mesure à 4/4, quantification à la double-croche.
          Les triolets, rubato et changements de tempo ne sont pas gérés. Le premier temps de la mesure 1 peut être décalé.
        - **Tablature guitare** : doigté choisi par heuristique simple (position la plus proche de la note précédente).
        - Pour corriger à la main : ouvre le MusicXML dans MuseScore (gratuit).
        """
    )
