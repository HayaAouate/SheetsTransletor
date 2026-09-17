"""SheetsTranslator — MVP interne : audio -> partition (violon) ou tablature (guitare).

Lancer avec :  streamlit run app.py
"""
import logging
import re
import os
import tempfile
import time

import numpy as np
import soundfile as sf
import streamlit as st
import streamlit.components.v1 as components

from transcriber.audio_input import get_audio_path
from transcriber.guitar_tabs import build_tab, export_tab_pdf, render_ascii_tab
from transcriber.notation import CREDIT, configure_lilypond, export_pdf, key_label, transcription_to_score
from transcriber.separate import DEFAULT_STEM, STEM_CHOICES, isolate_stem
from transcriber.transcribe import transcribe_audio
from transcriber.viewer import audio_data_uri, build_viewer_html

# Logs dans le terminal qui a lancé `streamlit run` : une ligne par étape du pipeline.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("sheets")

st.set_page_config(page_title="SheetsTranslator", page_icon="🎻")
st.title("🎻 SheetsTranslator — Audio → Partition")
st.caption("MVP interne — transcription violon (partition) et guitare (tablature) à partir d'un fichier ou d'un lien YouTube.")

configure_lilypond()  # trouve Lilypond + FFmpeg tout seul (PATH ou dossiers d'installation connus)

# Every input widget is keyed on this counter: bumping it (see "+ Nouvelle transcription" at the bottom)
# gives fresh, empty widgets instead of the values of the previous run.
gen = st.session_state.setdefault("form_gen", 0)

instrument_choice = st.radio("Instrument", ["Violon", "Guitare"], horizontal=True, key=f"instrument_{gen}")
instrument_key = "Violin" if instrument_choice == "Violon" else "Guitar"

col_t, col_a = st.columns(2)
song_title = col_t.text_input("Titre du morceau", placeholder="Die On This Hill", key=f"title_{gen}").strip()
song_artist = col_a.text_input("Artiste (optionnel)", placeholder="Sienna Spiro", key=f"artist_{gen}").strip()
source_type = st.radio("Source audio", ["Fichier local", "Lien YouTube"], horizontal=True, key=f"source_{gen}")

audio_bytes = None
audio_name = None
youtube_url = None

if source_type == "Fichier local":
    uploaded = st.file_uploader("Fichier audio (mp3, wav, m4a)", type=["mp3", "wav", "m4a"], key=f"file_{gen}")
    if uploaded is not None:
        audio_bytes = uploaded.getvalue()  # pas .read() : le curseur reste en fin de fichier entre deux reruns
        audio_name = uploaded.name
else:
    youtube_url = st.text_input("Lien YouTube (ou Instagram / TikTok)", key=f"url_{gen}")

stem_labels = list(STEM_CHOICES)
stem_choice = st.selectbox(
    "Piste à isoler avant transcription (Demucs)",
    stem_labels,
    index=stem_labels.index(DEFAULT_STEM[instrument_key]),
    key=f"stem_{gen}",
    help="Sur une chanson complète, isoler l'instrument est indispensable : sinon voix, basse et batterie "
    "sont transcrites en même temps. Choisis « Aucune » seulement pour un enregistrement solo déjà propre.",
)

with st.expander("Réglages avancés"):
    method_label = st.radio(
        "Détection des notes",
        ["Suivi de mélodie (CREPE) — instrument à une voix", "Basic Pitch — polyphonique"],
        index=0 if instrument_key == "Violin" else 1,
        key=f"method_{gen}",
        help="Le suivi de mélodie suit la hauteur de la ligne jouée (violon, flûte, voix) : traits rapides "
        "et octaves justes, notes tenues d'un seul tenant. Basic Pitch détecte plusieurs notes à la fois "
        "(guitare, accords) ; les deux réglages ci-dessous ne concernent que lui.",
    )
    method = "melody" if method_label.startswith("Suivi") else "basic_pitch"
    onset_threshold = st.slider(
        "Sensibilité de détection des notes",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Plus bas = plus de notes détectées (plus de faux positifs). Plus haut = seulement les notes les plus sûres.",
    )
    min_note_ms = st.slider(
        "Durée minimale d'une note (ms)", min_value=30, max_value=250, value=80, step=10,
        help="Les notes plus courtes sont considérées comme du bruit et supprimées.",
    )

def _safe_filename(name: str, fallback: str) -> str:
    cleaned = re.sub(r"[^\w\- ]+", "", name or "").strip().replace(" ", "_")
    return cleaned or fallback


def run_transcription():
    """Full pipeline; everything the results section needs is stored in st.session_state.result."""
    t0 = time.time()
    log.info("Transcription lancée — instrument=%s, source=%s, stem=%s, onset=%.2f, min_note=%sms",
             instrument_key, youtube_url or audio_name, stem_choice, method, onset_threshold, min_note_ms)
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

    label = ("Suivi de mélodie (CREPE) + temps (Beat This!)... environ 3x la durée du morceau."
             if method == "melody" else "Transcription en cours (Basic Pitch)... ça peut prendre 10-30s selon la durée.")
    with st.spinner(label):
        tr = transcribe_audio(
            stem_path,
            instrument=instrument_key,
            onset_threshold=onset_threshold,
            minimum_note_length=float(min_note_ms),
            tempo_audio_path=audio_path,
            method=method,
        )
    log.info("Transcription terminée : %d notes, %.0f bpm (%.1fs)", len(tr.notes), tr.bpm, time.time() - t0)
    if not tr.notes:
        st.warning("Aucune note détectée. Essaie une autre piste à isoler, un enregistrement plus propre, ou baisse la sensibilité.")
        return None

    out_dir = tempfile.mkdtemp()
    title = song_title or ("Transcription violon" if instrument_key == "Violin" else "Tablature guitare")
    # File names = "Titre - Artiste" as typed (sanitised), else a generic name.
    base = _safe_filename(" - ".join(x for x in (song_title, song_artist) if x),
                          "partition_violon" if instrument_key == "Violin" else "tab_guitare")

    # Audio synthétisé de la transcription (pré-écoute + MIDI) : même ligne de temps que l'audio d'origine.
    preview_midi = tr.to_pretty_midi(program=40 if instrument_key == "Violin" else 24)
    preview_wave = preview_midi.synthesize(fs=22050)
    preview_wave = preview_wave / (np.abs(preview_wave).max() or 1.0) * 0.8
    preview_path = os.path.join(out_dir, "transcription.wav")
    sf.write(preview_path, preview_wave.astype(np.float32), 22050, format="WAV", subtype="PCM_16")
    midi_path = os.path.join(out_dir, base + ".mid")
    preview_midi.write(midi_path)

    result = {"tr": tr, "instrument": instrument_key, "title": title, "artist": song_artist, "base": base,
              "midi_path": midi_path, "files": [], "source": youtube_url or audio_name}

    with st.spinner("Génération de la partition..."):
        score, detected_key = transcription_to_score(tr, title=title, artist=song_artist or None)
        result["key"] = key_label(detected_key)
        pdf_path, xml_path = export_pdf(score, os.path.join(out_dir, base + ".pdf"))
        with open(xml_path, encoding="utf-8") as f:
            result["musicxml"] = f.read()
        log.info("Partition générée : %s (%.1fs)", pdf_path, time.time() - t0)

    if instrument_key == "Violin":
        result["files"] = [("📄 Partition (PDF)", pdf_path, base + ".pdf"),
                           ("🎼 MusicXML (MuseScore)", xml_path, base + ".musicxml"),
                           ("🎹 MIDI", midi_path, base + ".mid")]
    else:
        assignments = build_tab(tr)
        ascii_tab = render_ascii_tab(assignments)
        tab_pdf = os.path.join(out_dir, base + "_tab.pdf")
        export_tab_pdf(ascii_tab, tab_pdf, title=title, subtitle=song_artist, credit=CREDIT,
                       info=f"{tr.bpm:.0f} bpm — 4/4 — une colonne = une double-croche")
        result["ascii_tab"] = ascii_tab
        result["files"] = [("📄 Tablature (PDF)", tab_pdf, base + "_tab.pdf"),
                           ("📄 Partition (PDF)", pdf_path, base + ".pdf"),
                           ("🎼 MusicXML", xml_path, base + ".musicxml"),
                           ("🎹 MIDI", midi_path, base + ".mid")]
        log.info("Tablature générée : %s (%.1fs)", tab_pdf, time.time() - t0)

    with st.spinner("Préparation de la lecture..."):
        sources = {"Transcription": audio_data_uri(preview_path)}
        if stem_path != audio_path:
            sources["Piste isolée"] = audio_data_uri(stem_path)
        sources["Original"] = audio_data_uri(audio_path)
        result["sources"] = sources  # the viewer itself is rebuilt at display time (see below)
    log.info("Terminé (%.1fs)", time.time() - t0)
    return result


if st.button("Transcrire", type="primary"):
    if not audio_bytes and not youtube_url:
        st.error("Ajoute un fichier audio ou un lien YouTube avant de lancer la transcription.")
        st.stop()
    # Forget the previous score first: if this run fails, the old one must not stay on screen and
    # pass for the transcription of the new source.
    st.session_state.result = None
    try:
        st.session_state.result = run_transcription()
    except Exception as exc:  # MVP: on affiche l'erreur brute pour debug rapide
        log.exception("Échec de la transcription")
        st.error(f"Erreur pendant la transcription : {exc}")
        raise

result = st.session_state.get("result")
if result:
    tr = result["tr"]
    st.caption(f"Source transcrite : {result.get('source') or '—'}")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Tempo détecté", f"{tr.bpm:.0f} bpm")
    col_b.metric("Tonalité", result["key"])
    col_c.metric("Notes", len(tr.notes))

    # Vue interactive : partition + lecture avec curseur. Bascule Transcription / Original pour
    # vérifier à l'oreille et à l'œil que ce qui est écrit correspond au morceau. Rebuilt on every
    # rerun (cheap: the audio is already encoded) so that a change to the viewer applies on reload.
    viewer_html = build_viewer_html(
        result["musicxml"], result["sources"], bpm=tr.bpm, beat_origin=tr.beat_origin, beat_times=tr.beat_times,
        title=result["title"], artist=result["artist"], credit=CREDIT,
    )
    components.html(viewer_html, height=760, scrolling=True)

    cols = st.columns(len(result["files"]))
    for col, (label, path, name) in zip(cols, result["files"]):
        with open(path, "rb") as f:
            col.download_button(label, f, file_name=name)

    if result.get("ascii_tab"):
        with st.expander("Tablature (texte)"):
            st.code(result["ascii_tab"], language=None)

    if st.button("＋ Nouvelle transcription", type="primary", use_container_width=True):
        # Drop the score and give the form fresh widgets, then land at the top of the page.
        st.session_state.result = None
        st.session_state.form_gen = gen + 1
        st.session_state.scroll_top = True
        st.rerun()

if st.session_state.pop("scroll_top", False):
    components.html("<script>window.parent.scrollTo({top: 0, behavior: 'instant'});</script>", height=0)

st.divider()
with st.expander("Limites connues de ce MVP"):
    st.markdown(
        """
        - **Monophonique** : une seule ligne mélodique est écrite (la note la plus forte à chaque instant). Pas d'accords.
        - **Séparation de sources** (Demucs) : très efficace pour isoler voix / guitare / piano / basse. Le violon
          tombe dans la piste « Autres » avec tout ce qui n'est pas voix/basse/batterie — sur un morceau très chargé
          (synthés, cordes d'accompagnement), le résultat reste approximatif.
        - **Rythme** : tempo constant détecté automatiquement, mesure à 4/4, écriture à la croche (double-croche
          seulement si le morceau l'exige) ; les petits silences entre deux notes sont lus comme du legato.
          Les triolets, rubato et changements de tempo ne sont pas gérés. Le premier temps de la mesure 1 peut être décalé.
        - **Tablature guitare** : doigté choisi par heuristique simple (position la plus proche de la note précédente).
        - Pour corriger à la main : ouvre le MusicXML dans MuseScore (gratuit).
        """
    )
