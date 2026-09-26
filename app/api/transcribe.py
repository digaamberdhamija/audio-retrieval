from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.transcription import transcribe_audio

router = APIRouter()


class TranscribeRequest(BaseModel):
    audio_path: str


class TranscribeResponse(BaseModel):
    transcript_path: str
    speakers: dict[str, str]
    segments: int
    words: int


@router.post("/transcribe")
async def post_transcribe(body: TranscribeRequest) -> TranscribeResponse:
    audio_path = Path(body.audio_path)
    if not audio_path.is_file():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {body.audio_path}")
    try:
        transcript, transcript_path = await transcribe_audio(audio_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return TranscribeResponse(
        transcript_path=transcript_path.as_posix(),
        speakers=transcript.speakers,
        segments=len(transcript.segments),
        words=sum(len(segment.words) for segment in transcript.segments),
    )
