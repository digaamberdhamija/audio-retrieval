import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import asyncpg

from app.core.config import settings
from app.services.chunking import load_words, sliding_windows
from app.services.embeddings import embed_texts


async def ingest_transcript(pool: asyncpg.Pool, transcript_path: Path) -> tuple[str, dict[int, int]]:
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    audio_file = Path(transcript["audio_file"]).as_posix()
    words = load_words(transcript)
    if not words:
        raise ValueError(f"No timestamped words found in {transcript_path}")

    windows = [
        window
        for config in settings.windows
        for window in sliding_windows(words, config.length_seconds, config.stride_seconds)
    ]
    embeddings = await embed_texts([window.text for window in windows])

    async with pool.acquire() as conn, conn.transaction():
        await conn.execute("DELETE FROM transcripts WHERE audio_file = $1", audio_file)
        transcript_id = await conn.fetchval(
            "INSERT INTO transcripts (audio_file, transcript_path, words) VALUES ($1, $2, $3) RETURNING id",
            audio_file,
            transcript_path.as_posix(),
            [asdict(word) for word in words],
        )
        await conn.executemany(
            """
            INSERT INTO chunks
                (transcript_id, window_seconds, start_seconds, end_seconds, speakers, text, embedding)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            [
                (
                    transcript_id,
                    window.window_seconds,
                    window.start_seconds,
                    window.end_seconds,
                    window.speakers,
                    window.text,
                    embedding,
                )
                for window, embedding in zip(windows, embeddings, strict=True)
            ],
        )

    return audio_file, dict(Counter(window.window_seconds for window in windows))
