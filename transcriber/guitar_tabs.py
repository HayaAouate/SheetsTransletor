"""Convert a cleaned Transcription into a guitar tablature, using a simple fret-position heuristic
(standard EADGBE tuning, picks the string/fret combo closest to the previous hand position).
This is an MVP heuristic, not a true fingering-optimization algorithm.

The tab is rhythmic: one 3-char column per grid slot (16th note by default), bar lines every measure,
so the spacing between numbers reflects the actual timing of the notes.
"""
from .transcribe import GRID, Transcription

# Open-string MIDI pitch numbers, standard tuning, low to high.
STANDARD_TUNING = {"E2": 40, "A2": 45, "D3": 50, "G3": 55, "B3": 59, "E4": 64}
STRING_ORDER = ["E2", "A2", "D3", "G3", "B3", "E4"]  # low to high
MAX_FRET = 15
BEATS_PER_BAR = 4
SLOTS_PER_BAR = int(BEATS_PER_BAR / GRID)


def assign_string_fret(pitch: int, prev_fret: int = None):
    """Pick the (string, fret) for a MIDI pitch, minimizing the jump from the previous fret (playability)."""
    candidates = [
        (s, pitch - STANDARD_TUNING[s])
        for s in STRING_ORDER
        if 0 <= pitch - STANDARD_TUNING[s] <= MAX_FRET
    ]
    if not candidates:
        return None  # out of standard guitar range
    if prev_fret is None:
        return min(candidates, key=lambda c: c[1])  # start low on the neck
    return min(candidates, key=lambda c: abs(c[1] - prev_fret))


def build_tab(tr: Transcription):
    """Return a list of (slot, string_name, fret) for every playable note, in time order."""
    assignments = []
    prev_fret = None
    for n in tr.notes:
        choice = assign_string_fret(n.pitch, prev_fret)
        if choice is None:
            continue  # note out of guitar range, skipped in this MVP
        string_name, fret = choice
        assignments.append((n.slot, string_name, fret))
        prev_fret = fret
    return assignments


def render_ascii_tab(assignments, bars_per_line: int = 3) -> str:
    """Render assignments as a classic 6-line ASCII tab (high E on top), one column per grid slot."""
    if not assignments:
        return "(aucune note dans la tessiture de la guitare)"

    by_slot = {slot: (string_name, fret) for slot, string_name, fret in assignments}
    last_bar = max(by_slot) // SLOTS_PER_BAR
    blocks = []
    for first_bar in range(0, last_bar + 1, bars_per_line):
        rows = {s: [] for s in STRING_ORDER}
        for bar in range(first_bar, min(first_bar + bars_per_line, last_bar + 1)):
            for s in STRING_ORDER:
                rows[s].append("|")
            for slot in range(bar * SLOTS_PER_BAR, (bar + 1) * SLOTS_PER_BAR):
                hit = by_slot.get(slot)
                for s in STRING_ORDER:
                    cell = f"{hit[1]:<2}-" if hit and hit[0] == s else "---"
                    rows[s].append(cell)
        lines = []
        for s in reversed(STRING_ORDER):  # high string on top
            lines.append(f"{s[0]}" + "".join(rows[s]) + "|")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def export_tab_pdf(ascii_tab: str, out_path: str, title: str = "Tablature guitare", subtitle: str = "",
                   info: str = "", credit: str = ""):
    """Render the ASCII tab into a simple monospace PDF.
    title/subtitle (song, artist) top-left, `credit` top-right, `info` (tempo etc.) in small print."""
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(out_path, pagesize=landscape(A4))
    width, height = landscape(A4)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(15 * mm, height - 15 * mm, title)
    if credit:
        c.setFont("Helvetica", 10)
        c.drawRightString(width - 15 * mm, height - 15 * mm, credit)
    y = height - 21 * mm
    if subtitle:
        c.setFont("Helvetica", 11)
        c.drawString(15 * mm, y, subtitle)
        y -= 5 * mm
    if info:
        c.setFont("Helvetica", 8)
        c.drawString(15 * mm, y, info)
        y -= 5 * mm
    c.setFont("Courier", 7.5)
    y -= 4 * mm
    line_height = 3.6 * mm
    for line in ascii_tab.split("\n"):
        if y < 15 * mm:
            c.showPage()
            c.setFont("Courier", 7.5)
            y = height - 15 * mm
        c.drawString(12 * mm, y, line)
        y -= line_height
    c.save()
    return out_path
