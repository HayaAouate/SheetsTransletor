"""Locate the external binaries the pipeline needs (Lilypond, FFmpeg) without relying on the user's
PATH: a freshly opened terminal on Windows often does not have winget's additions yet, and the app
should just work. Found folders are prepended to this process' PATH so that yt-dlp, librosa and
music21 pick them up as well.
"""
import glob
import os
import shutil

# Glob patterns for the usual install locations, most likely first. `~` and env vars are expanded.
_CANDIDATES = {
    "lilypond": [
        r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\LilyPond.LilyPond_*\lilypond-*\bin\lilypond.exe",
        r"%LOCALAPPDATA%\Programs\lilypond*\bin\lilypond.exe",
        r"%ProgramFiles%\lilypond*\bin\lilypond.exe",
        r"%ProgramFiles(x86)%\lilypond*\bin\lilypond.exe",
        r"C:\lilypond*\bin\lilypond.exe",
        "/opt/homebrew/bin/lilypond",
        "/usr/local/bin/lilypond",
        "/Applications/LilyPond.app/Contents/Resources/bin/lilypond",
    ],
    "ffmpeg": [
        r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_*\ffmpeg-*\bin\ffmpeg.exe",
        r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_*\ffmpeg-*\bin\ffmpeg.exe",
        r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe",
        r"%ProgramFiles%\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ],
}


def find_tool(name: str) -> str | None:
    """Full path of `name` (lilypond / ffmpeg): PATH first, then the known install folders.
    The folder of a match found outside PATH is added to PATH for the rest of the process."""
    found = shutil.which(name)
    if found:
        return found
    for pattern in _CANDIDATES.get(name, []):
        matches = sorted(glob.glob(os.path.expandvars(os.path.expanduser(pattern))), reverse=True)
        if matches:
            found = matches[0]  # newest version when several are installed
            os.environ["PATH"] = os.path.dirname(found) + os.pathsep + os.environ.get("PATH", "")
            return found
    return None


def ensure_tools_on_path() -> dict:
    """Locate every known tool once at startup. Returns {name: path or None}."""
    return {name: find_tool(name) for name in _CANDIDATES}
