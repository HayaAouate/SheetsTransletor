# Backend: the transcription API (FastAPI) + the pipeline (Demucs, CREPE, Beat This!, music21, Lilypond).
# CPU only. ~3 GB image; models (~250 MB) are downloaded on first use into the /models volume.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data \
    TORCH_HOME=/models/torch \
    XDG_CACHE_HOME=/models/cache \
    SHEETS_CACHE_DIR=/models/crepe \
    OMP_NUM_THREADS=2

# Lilypond (PDF engraving), ffmpeg (audio decoding, yt-dlp), git (beat_this is installed from GitHub)
RUN apt-get update && apt-get install -y --no-install-recommends lilypond ffmpeg git libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU wheels of torch/torchaudio (the default index ships CUDA builds, 3x bigger), then the rest.
# Streamlit is the desktop UI only: not needed by the API, left out of the image.
RUN pip install --upgrade pip \
    && pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && grep -vi streamlit requirements.txt > /tmp/requirements-api.txt \
    && pip install -r /tmp/requirements-api.txt

COPY transcriber ./transcriber
COPY api ./api

VOLUME ["/data", "/models"]
EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
