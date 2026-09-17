"""Every setting of the API, read once from the environment (see .env.example for the meaning and defaults).

Nothing else in `api/` calls os.environ: this is the one place to look for what can be configured.
"""
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


# --- Paths ---
DATA_DIR = _env("DATA_DIR", os.path.join(_ROOT, "data"))       # db.sqlite + one folder per transcription

# --- HTTP ---
ROOT_PATH = _env("ROOT_PATH", "")                              # prefix stripped by the reverse proxy (Caddy: /api)
CORS_ORIGINS = [o.strip() for o in _env("CORS_ORIGINS", "*").split(",") if o.strip()]

# --- Access control: the shared secret every request must carry in `X-API-Key` (except /health).
# Fail closed: without a key the process refuses to start rather than serving an open API on the internet.
API_KEY = _env("API_KEY", "")
if len(API_KEY) < 16:
    raise RuntimeError("API_KEY manquante ou trop courte (≥ 16 caractères) : génère-la avec "
                       "python -c \"import secrets; print(secrets.token_urlsafe(32))\" et mets-la dans .env")

# --- Abuse limits ---
MAX_UPLOAD_MB = float(_env("MAX_UPLOAD_MB", "50"))             # uploaded file size
MAX_AUDIO_MIN = float(_env("MAX_AUDIO_MIN", "10"))             # audio length (a job costs minutes of CPU per minute of audio)
MAX_PENDING = int(_env("MAX_PENDING", "5"))                    # jobs waiting in the queue (one runs at a time)
URL_HOSTS = tuple(h.strip().lower() for h in _env(
    "URL_HOSTS", "youtube.com,youtu.be,tiktok.com,vm.tiktok.com,instagram.com,soundcloud.com").split(",") if h.strip())

# --- Storage retention (api/workers/retention.py) ---
AUDIO_RETENTION_DAYS = float(_env("AUDIO_RETENTION_DAYS", "30"))  # audio (original/stem/preview) dropped after N days unopened; score kept
ERROR_RETENTION_DAYS = float(_env("ERROR_RETENTION_DAYS", "2"))   # failed jobs deleted entirely after N days
STORAGE_MAX_GB = float(_env("STORAGE_MAX_GB", "5"))               # above this, audio of the least recently opened is dropped first
RETENTION_INTERVAL_MIN = float(_env("RETENTION_INTERVAL_MIN", "60"))
