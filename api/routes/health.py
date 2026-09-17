"""GET /health — open (no key): for uptime checks and Caddy."""
from fastapi import APIRouter

from ..schemas import HealthOut
from ..workers import queue

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
def health():
    return {"ok": True, "pending_jobs": queue.pending_count()}
