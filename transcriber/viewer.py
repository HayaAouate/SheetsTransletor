"""Interactive score viewer for the Streamlit app: MusicXML rendered in the browser by
OpenSheetMusicDisplay (OSMD), with audio playback and a cursor that follows the music.

Everything (play/pause, source switch, speed, click-to-seek) lives inside one HTML component so
that using it never triggers a Streamlit rerun. The cursor is driven by the audio element's
currentTime: each cursor position maps to a time in seconds through the transcription's tempo
and beat origin, which are the same for the synthesized preview, the original mix and the
isolated stem (they all share the original recording's timeline).
"""
import base64
import json
import os
import shutil
import subprocess
import tempfile

OSMD_CDN = "https://cdn.jsdelivr.net/npm/opensheetmusicdisplay@1.9.9/build/opensheetmusicdisplay.min.js"


def audio_data_uri(path: str, mp3_bitrate: str = "96k") -> str:
    """
    Return a data: URI for an audio file, compressed to mono MP3 through ffmpeg when available
    (a 3-minute WAV is ~30 MB, which is too heavy to inline in the page; the MP3 is ~2 MB).
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        out = os.path.join(tempfile.mkdtemp(), "audio.mp3")
        result = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", path, "-ac", "1", "-b:a", mp3_bitrate, out],
            capture_output=True,
        )
        if result.returncode == 0 and os.path.exists(out):
            with open(out, "rb") as f:
                return "data:audio/mpeg;base64," + base64.b64encode(f.read()).decode("ascii")
    mime = {".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".ogg": "audio/ogg"}.get(
        os.path.splitext(path)[1].lower(), "audio/wav"
    )
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode("ascii")


def build_viewer_html(
    musicxml: str,
    sources: dict,
    bpm: float,
    beat_origin: float,
    title: str = "",
    artist: str = "",
    credit: str = "",
) -> str:
    """
    Build the self-contained HTML for the viewer.

    musicxml: the score as a MusicXML string.
    sources: {"label": data_uri} audio sources, in display order (first one selected).
    bpm / beat_origin: from the Transcription; time(s) = beat_origin + whole_notes * 4 * 60 / bpm.
    """
    config = {
        "musicxml": musicxml,
        "sources": sources,
        "secondsPerWhole": 4 * 60.0 / bpm,
        "beatOrigin": beat_origin,
        "title": title,
        "artist": artist,
        "credit": credit,
    }
    return _TEMPLATE.replace("__OSMD_CDN__", OSMD_CDN).replace("__CONFIG__", json.dumps(config))


_TEMPLATE = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<script src="__OSMD_CDN__"></script>
<style>
  body { margin: 0; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; background: #fff; color: #222; }
  #toolbar { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; padding: 8px 12px;
             border-bottom: 1px solid #e5e5e5; position: sticky; top: 0; background: #fff; z-index: 2; }
  #toolbar button { border: 0; border-radius: 6px; padding: 6px 14px; font-size: 14px; cursor: pointer; }
  #play { background: #0d8f7f; color: #fff; font-weight: 600; min-width: 84px; }
  #stop { background: #eee; }
  .seg { display: inline-flex; border: 1px solid #ccc; border-radius: 6px; overflow: hidden; }
  .seg button { border-radius: 0; background: #fff; }
  .seg button.on { background: #0d8f7f; color: #fff; }
  #time { font-variant-numeric: tabular-nums; color: #555; min-width: 90px; }
  label { font-size: 13px; color: #555; display: inline-flex; align-items: center; gap: 6px; }
  #header { text-align: center; padding: 14px 12px 0; position: relative; }
  #header h2 { margin: 0; font-size: 22px; }
  #header .artist { color: #444; margin-top: 2px; }
  #header .credit { position: absolute; right: 16px; top: 16px; color: #444; font-size: 13px; }
  #score { padding: 0 8px 24px; cursor: pointer; }
  #status { padding: 12px; color: #888; font-size: 13px; }
</style>
</head>
<body>
<div id="toolbar">
  <button id="play">▶ Lecture</button>
  <button id="stop">■</button>
  <span id="time">0:00 / 0:00</span>
  <span class="seg" id="sources"></span>
  <label>Vitesse <input id="speed" type="range" min="0.5" max="1.5" step="0.05" value="1" style="width:110px"> <span id="speedv">1.00×</span></label>
</div>
<div id="header"></div>
<div id="score"></div>
<div id="status">Chargement de la partition…</div>
<audio id="audio" preload="auto"></audio>
<script>
const CFG = __CONFIG__;
const audio = document.getElementById("audio");
const playBtn = document.getElementById("play");
const timeEl = document.getElementById("time");
const status = document.getElementById("status");

// ---- header (title / artist / credit) ----
const header = document.getElementById("header");
header.innerHTML = (CFG.title ? "<h2>" + esc(CFG.title) + "</h2>" : "")
  + (CFG.artist ? "<div class='artist'>" + esc(CFG.artist) + "</div>" : "")
  + (CFG.credit ? "<div class='credit'>" + esc(CFG.credit) + "</div>" : "");
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }

// ---- audio sources ----
const labels = Object.keys(CFG.sources);
const srcBox = document.getElementById("sources");
let currentSrc = labels[0];
labels.forEach(l => {
  const b = document.createElement("button");
  b.textContent = l;
  b.className = l === currentSrc ? "on" : "";
  b.onclick = () => selectSource(l);
  srcBox.appendChild(b);
});
function selectSource(l) {
  const wasPlaying = !audio.paused;
  const t = audio.currentTime;
  currentSrc = l;
  [...srcBox.children].forEach(b => b.className = b.textContent === l ? "on" : "");
  audio.src = CFG.sources[l];
  audio.currentTime = t;
  if (wasPlaying) audio.play();
}
audio.src = CFG.sources[currentSrc];

const speed = document.getElementById("speed");
speed.oninput = () => { audio.playbackRate = parseFloat(speed.value); document.getElementById("speedv").textContent = parseFloat(speed.value).toFixed(2) + "×"; };
audio.preservesPitch = true;

playBtn.onclick = () => { if (audio.paused) audio.play(); else audio.pause(); };
document.getElementById("stop").onclick = () => { audio.pause(); audio.currentTime = 0; syncCursor(true); };
let timer = null;  // setInterval rather than requestAnimationFrame: keeps running in a hidden tab/iframe
audio.onplay = () => { playBtn.textContent = "❚❚ Pause"; clearInterval(timer); timer = setInterval(tick, 40); };
audio.onpause = () => { playBtn.textContent = "▶ Lecture"; clearInterval(timer); tick(); };
audio.onended = () => { playBtn.textContent = "▶ Lecture"; clearInterval(timer); tick(); };

function fmt(t) { t = Math.max(0, t || 0); return Math.floor(t / 60) + ":" + String(Math.floor(t % 60)).padStart(2, "0"); }

// ---- score ----
const osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay("score", {
  autoResize: true, drawTitle: false, drawSubtitle: false, drawComposer: false, drawLyricist: false,
  drawPartNames: false, followCursor: true, cursorsOptions: [{type: 0, color: "#0d8f7f", alpha: 0.5, follow: true}],
});
let cursorTimes = [];   // seconds at which each cursor position starts
let cursorIndex = 0;
let measureTimes = [];  // seconds at which each measure starts

osmd.load(CFG.musicxml).then(() => {
  osmd.render();
  osmd.cursor.show();
  // Walk the whole score once to map cursor positions to seconds.
  osmd.cursor.reset();
  const it = osmd.cursor.iterator;
  while (!it.EndReached) {
    cursorTimes.push(CFG.beatOrigin + it.currentTimeStamp.RealValue * CFG.secondsPerWhole);
    osmd.cursor.next();
  }
  osmd.cursor.reset();
  cursorIndex = 0;
  measureTimes = osmd.Sheet.SourceMeasures.map(m => CFG.beatOrigin + m.AbsoluteTimestamp.RealValue * CFG.secondsPerWhole);
  status.textContent = "";
  timeEl.textContent = fmt(0) + " / " + fmt(audio.duration);
}).catch(e => { status.textContent = "Impossible d'afficher la partition : " + e; });
audio.onloadedmetadata = () => { timeEl.textContent = fmt(audio.currentTime) + " / " + fmt(audio.duration); };

// Move the cursor to the position whose start time is the last one <= t.
function syncCursor(force) {
  if (!cursorTimes.length) return;
  const t = audio.currentTime;
  let target = cursorTimes.findIndex(ct => ct > t) - 1;
  if (target < 0) target = cursorTimes.length ? (cursorTimes[0] > t ? 0 : cursorTimes.length - 1) : 0;
  if (target === cursorIndex && !force) return;
  if (target < cursorIndex || force) { osmd.cursor.reset(); cursorIndex = 0; }
  while (cursorIndex < target && !osmd.cursor.iterator.EndReached) { osmd.cursor.next(); cursorIndex++; }
}
function tick() {
  syncCursor(false);
  timeEl.textContent = fmt(audio.currentTime) + " / " + fmt(audio.duration);
}

// ---- click on a measure -> seek there ----
document.getElementById("score").addEventListener("click", ev => {
  if (!osmd.GraphicSheet) return;
  const svg = document.querySelector("#score svg");
  if (!svg) return;
  const r = svg.getBoundingClientRect();
  const unit = 10 * osmd.zoom;  // OSMD units -> px
  const x = (ev.clientX - r.left) / unit, y = (ev.clientY - r.top) / unit;
  let best = -1, bestD = Infinity;
  osmd.GraphicSheet.MeasureList.forEach((staffMeasures, i) => {
    const m = staffMeasures[0];
    if (!m) return;
    const b = m.PositionAndShape;
    const left = b.AbsolutePosition.x, top = b.AbsolutePosition.y - 2, w = b.Size.width, h = b.Size.height + 4;
    if (y >= top && y <= top + h) {
      const d = x < left ? left - x : (x > left + w ? x - left - w : 0);
      if (d < bestD) { bestD = d; best = i; }
    }
  });
  if (best >= 0 && measureTimes[best] !== undefined) {
    audio.currentTime = Math.max(0, measureTimes[best]);
    syncCursor(true);
    timeEl.textContent = fmt(audio.currentTime) + " / " + fmt(audio.duration);
    if (audio.paused) audio.play();
  }
});
</script>
</body>
</html>
"""
