import asyncio
import contextlib
import json
from collections import Counter
from dataclasses import asdict
from functools import cache
from pathlib import Path

from google import genai
from google.genai import errors, types

from app.core.config import GEMINI_API_KEY, settings
from app.core.models import Segment, Transcript, Word

MIME_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mp3",
    ".aif": "audio/aiff",
    ".aiff": "audio/aiff",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/m4a",
    ".opus": "audio/opus",
    ".webm": "audio/webm",
    ".mpeg": "audio/mpeg",
}


@cache
def _client() -> genai.Client:
    timeout_ms = settings.transcription.request_timeout_seconds * 1000
    return genai.Client(api_key=GEMINI_API_KEY, http_options=types.HttpOptions(timeout=timeout_ms))


async def transcribe_audio(audio_path: Path) -> tuple[Transcript, Path]:
    config = settings.transcription

    uploaded = await upload_audio(audio_path)
    try:
        response = await _client().aio.models.generate_content(
            model=config.model,
            contents=[types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type)],
            config=types.GenerateContentConfig(
                audio_transcription_config=types.AudioTranscriptionConfig(
                    mode=types.AudioTranscriptionConfigMode.VERBATIM,
                    word_timestamp=True,
                    diarization=True,
                    language_codes=config.language_codes or None,
                ),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
    finally:
        with contextlib.suppress(errors.APIError):
            await _client().aio.files.delete(name=uploaded.name)

    segments = parse_segments(response)
    speakers = assign_speaker_names(segments, config.expected_speakers)
    transcript = Transcript(
        audio_file=audio_path.as_posix(),
        model=config.model,
        speakers=speakers,
        segments=segments,
    )
    return transcript, save_transcript(transcript)


async def upload_audio(audio_path: Path) -> types.File:
    mime_type = MIME_TYPES.get(audio_path.suffix.lower())
    if mime_type is None:
        raise ValueError(f"Unsupported audio format: {audio_path.suffix}")

    uploaded = await _client().aio.files.upload(file=audio_path, config={"mime_type": mime_type})
    while uploaded.state == types.FileState.PROCESSING:
        await asyncio.sleep(2)
        uploaded = await _client().aio.files.get(name=uploaded.name)

    if uploaded.state == types.FileState.FAILED:
        raise RuntimeError(f"Gemini could not process {audio_path}")
    return uploaded


def parse_segments(response: types.GenerateContentResponse) -> list[Segment]:
    candidate = response.candidates[0] if response.candidates else None
    parts = candidate.content.parts if candidate and candidate.content else None
    segments = [
        to_segment(part.audio_transcription) for part in parts or [] if part.audio_transcription
    ]
    if not segments:
        raise RuntimeError("Gemini returned no transcription")
    return segments


def to_segment(transcription: types.Transcription) -> Segment:
    words = [
        Word(word.word or "", to_seconds(word.start_offset), to_seconds(word.end_offset))
        for word in transcription.words or []
    ]
    return Segment(
        speaker=transcription.speaker_label or "",
        start=min((word.start for word in words if word.start is not None), default=None),
        end=max((word.end for word in words if word.end is not None), default=None),
        text=(transcription.text or " ".join(word.text for word in words)).strip(),
        words=words,
    )


def assign_speaker_names(segments: list[Segment], expected_speakers: int) -> dict[str, str]:
    """Rename the most talkative model labels to "Speaker 1".."Speaker N" in order of first
    appearance. Segments with any other label take the speaker of the segment before them."""
    words_per_label = Counter()
    for segment in segments:
        if segment.speaker:
            words_per_label[segment.speaker] += len(segment.text.split())
    main_labels = [label for label, _ in words_per_label.most_common(expected_speakers)]

    names: dict[str, str] = {}
    for segment in segments:
        if segment.speaker in main_labels and segment.speaker not in names:
            names[segment.speaker] = f"Speaker {len(names) + 1}"

    current_name = next(iter(names.values()), "Unknown")
    for segment in segments:
        current_name = names.get(segment.speaker, current_name)
        segment.speaker = current_name

    return {name: label for label, name in names.items()}


def save_transcript(transcript: Transcript) -> Path:
    output_dir = Path(settings.transcription.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(transcript.audio_file).stem}.json"
    output_path.write_text(json.dumps(asdict(transcript), indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def to_seconds(offset: str | None) -> float | None:
    return float(offset.removesuffix("s")) if offset else None
