from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.services.ingestion import ingest_transcript

router = APIRouter()


class IngestRequest(BaseModel):
    transcript_path: str


class IngestResponse(BaseModel):
    audio_file: str
    chunks_per_window: dict[int, int]


@router.post("/ingest")
async def post_ingest(body: IngestRequest, request: Request) -> IngestResponse:
    transcript_path = Path(body.transcript_path)
    if not transcript_path.is_file():
        raise HTTPException(status_code=404, detail=f"Transcript not found: {body.transcript_path}")
    try:
        audio_file, counts = await ingest_transcript(request.app.state.pool, transcript_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return IngestResponse(audio_file=audio_file, chunks_per_window=counts)
