"""Shapes of what the API returns (Pydantic models): documented in /docs and the contract the web front relies on."""
from typing import Literal

from pydantic import BaseModel, Field

Status = Literal["queued", "running", "done", "error"]
FileKind = Literal["original", "stem", "preview", "musicxml", "pdf", "midi", "tab_pdf", "timeline"]


class Timeline(BaseModel):
    """What the score player needs to move its cursor in sync with the recording."""
    bpm: float
    beat_origin: float = Field(description="Seconds in the recording where beat 0 (start of bar 1) falls")
    beat_times: list[float] | None = Field(description="Seconds of each beat; None when the tempo is constant")


class TranscriptionOut(BaseModel):
    id: str
    created_at: float
    updated_at: float
    last_opened_at: float | None = Field(description="Last GET of this transcription; drives the audio retention")
    title: str
    artist: str
    instrument: Literal["Violin", "Guitar"]
    method: Literal["melody", "basic_pitch"]
    stem_choice: str
    source_kind: Literal["upload", "url"]
    source: str = Field(description="Uploaded file name, or the URL")
    status: Status
    progress: float = Field(ge=0, le=1)
    message: str = Field(description="Current step while running, the error when failed")
    bpm: float | None
    key_label: str | None
    note_count: int | None
    duration_s: float | None
    files: list[FileKind] = Field(description="Files ready to download; audio kinds disappear after the retention delay")
    timeline: Timeline | None
    queue_position: int = Field(description="Jobs ahead of this one; 0 unless queued")


class HealthOut(BaseModel):
    ok: bool
    pending_jobs: int
