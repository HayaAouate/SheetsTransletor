"""The FastAPI application: middlewares, routes, background threads.

    uvicorn api.app:app --port 8000        (API_KEY must be set, see .env.example)
    docs: http://localhost:8000/docs       (click "Authorize" and paste the key)

Layout of the package:
    settings.py   every environment variable, in one place
    security.py   the API key check
    schemas.py    response models (the contract with the web front)
    routes/       one file per resource: health, transcriptions
    storage/      database.py (SQLite rows) and files.py (the folder of each transcription)
    workers/      queue.py (one job at a time), pipeline.py (the transcription itself), retention.py (clean-up)
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import settings
from .routes import health, transcriptions
from .workers import queue, retention

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    queue.start()      # re-queues what a previous process left unfinished
    retention.start()  # first sweep now, then every RETENTION_INTERVAL_MIN
    yield


app = FastAPI(
    title="SheetsTranslator — transcription API",
    version="0.2",
    lifespan=lifespan,
    root_path=settings.ROOT_PATH,  # the prefix the reverse proxy strips (Caddy: /api), so /docs keeps working
)
app.add_middleware(  # the web front runs on another origin (dev server, or another container)
    CORSMiddleware, allow_origins=settings.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"],
)
app.include_router(health.router)
app.include_router(transcriptions.router)
