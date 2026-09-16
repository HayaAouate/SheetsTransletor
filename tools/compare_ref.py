"""Compare our transcription of an audio file with a Songscription reference score (their
`inscript/1` JSON, saved next to the sample as <name>.songscription.json).

    .venv/Scripts/python tools/compare_ref.py samples/tiktok_lac0v.mp3

Prints both scores measure by measure in a compact form (`Eb5:0.5` = pitch:quarterLength,
`r1` = rest, `~` = tied to the next note) and a few numbers: note count, share of onsets on the
eighth grid, and a bar-by-bar match rate once the two scores are aligned on their first common bar.
"""
import collections
import json
import logging
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music21 import note  # noqa: E402

from transcriber.notation import key_label, transcription_to_score  # noqa: E402
from transcriber.separate import DEFAULT_STEM, isolate_stem  # noqa: E402
from transcriber.transcribe import transcribe_audio  # noqa: E402


def load_reference(path):
    """Songscription inscript/1 JSON -> list of measures, each a list of (name, quarterLength)."""
    score = json.load(open(path, encoding="utf-8"))["score"]
    voice = score["parts"][0]["staves"][0]["voices"][0]
    measures = []
    for m in voice["liveByMeasure"]:
        events = []
        for span in m["spans"]:
            for e in span["events"]:
                ql = Fraction(e["duration"]["num"], e["duration"]["den"]) * 4
                if e["kind"] == "rest":
                    events.append(("r", ql))
                else:
                    p = e["heads"][0]["pitch"]
                    name = p["step"] + {-1: "b", 1: "#", 0: ""}[p.get("alter", 0)] + str(p["octave"])
                    events.append((name, ql))
        measures.append(events)
    bpm = score["tempos"][0]["quarterBpm"]
    fifths = score["keys"][0]["fifths"]
    return measures, bpm, fifths


def ours(audio, stem_choice=None, **kw):
    stem = isolate_stem(audio, stem_choice or DEFAULT_STEM["Violin"])
    tr = transcribe_audio(stem, instrument="Violin", tempo_audio_path=audio, **kw)
    score, k = transcription_to_score(tr)
    part = score.parts[0]
    part.makeTies(inPlace=True)
    measures = []
    for m in part.getElementsByClass("Measure"):
        events = []
        for e in m.notesAndRests:
            ql = Fraction(e.quarterLength).limit_denominator(16)
            if e.isRest:
                events.append(("r", ql))
            else:
                tied = "~" if e.tie and e.tie.type != "stop" else ""
                events.append((e.pitch.nameWithOctave.replace("-", "b") + tied, ql))
        measures.append(events)
    return measures, tr, k


def fmt(events):
    return " ".join(f"{n}:{ql}" if n != "r" else f"r{ql}" for n, ql in events)


def pc(name):
    """Pitch class only: Songscription writes this violin an octave above what is played, and
    the octave is a display choice anyway; the comparison is on pitch class + rhythm."""
    return name.rstrip("~").rstrip("0123456789")


def onsets(measures):
    """(bar, beat, pitch class) of every note onset, ignoring tie continuations."""
    out = []
    for i, events in enumerate(measures):
        pos = Fraction(0)
        prev_tied = False
        for n, ql in events:
            if n != "r" and not prev_tied:
                out.append((i, pos, pc(n)))
            prev_tied = n.endswith("~")
            pos += ql
    return out


def stats(measures, label):
    ons = onsets(measures)
    on_eighth = sum(1 for _, pos, _ in ons if pos % Fraction(1, 2) == 0)
    durs = collections.Counter(str(ql) for ev in measures for _, ql in ev)
    print(f"{label}: {len(measures)} mesures, {len(ons)} notes, {100 * on_eighth / max(1, len(ons)):.0f}% sur la grille croche, "
          f"durées={dict(durs)}")
    return ons


def bar_match(ref, got, shift):
    """Share of reference bars whose note sequence (pitches + onsets) is reproduced exactly in ours,
    with ours shifted by `shift` bars."""
    ref_bars = collections.defaultdict(list)
    got_bars = collections.defaultdict(list)
    for i, pos, n in onsets(ref):
        ref_bars[i].append((pos, n))
    for i, pos, n in onsets(got):
        got_bars[i - shift].append((pos, n))
    exact = sum(1 for i in ref_bars if got_bars.get(i) == ref_bars[i])
    # Softer: pitch sequence only (rhythm ignored)
    seq = sum(1 for i in ref_bars if [n for _, n in got_bars.get(i, [])] == [n for _, n in ref_bars[i]])
    return exact, seq, len(ref_bars)


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    audio = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else "basic_pitch"
    ref_path = os.path.splitext(audio)[0] + ".songscription.json"
    ref, ref_bpm, ref_fifths = load_reference(ref_path)
    got, tr, k = ours(audio, method=method)

    print(f"\nREF : {ref_bpm:.1f} bpm, {ref_fifths} à l'armure")
    print(f"OURS ({method}): {tr.bpm:.1f} bpm, {key_label(k)}")
    ref_ons = stats(ref, "REF ")
    got_ons = stats(got, "OURS")

    best = max(range(-4, 5), key=lambda s: bar_match(ref, got, s)[1])
    exact, seq, n = bar_match(ref, got, best)
    print(f"\nAlignement: nos mesures décalées de {best} | mesures identiques {exact}/{n}, mêmes hauteurs {seq}/{n}\n")
    width = max(len(fmt(ev)) for ev in ref)
    for i in range(max(len(ref), len(got) - best)):
        r = fmt(ref[i]) if i < len(ref) else ""
        g = fmt(got[i + best]) if 0 <= i + best < len(got) else ""
        mark = "  " if r and g and onsets([ref[i]]) == onsets([got[i + best]]) else "≠ "
        print(f"M{i + 1:2d} {mark}REF  {r}")
        print(f"     {'  '}OURS {g}")


if __name__ == "__main__":
    main()
