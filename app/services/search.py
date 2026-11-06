import asyncio

import asyncpg

from app.core.config import WindowSettings, settings
from app.core.models import Chunk, SpeakerWord
from app.services.chunking import render_text, speakers_in, words_between
from app.services.embeddings import embed_texts
from app.services.reranker import rerank_scores

HYBRID_SEARCH_SQL = """
WITH query AS (
    SELECT replace(plainto_tsquery($2::regconfig, $1)::text, '&', '|')::tsquery AS tsquery
),
keyword AS (
    SELECT id, row_number() OVER (ORDER BY keyword_score DESC) AS rank
    FROM (
        SELECT c.id, ts_rank(c.text_search, query.tsquery) AS keyword_score
        FROM chunks c
        CROSS JOIN query
        WHERE c.window_seconds = $4 AND c.text_search @@ query.tsquery
        ORDER BY keyword_score DESC
        LIMIT $5
    ) matches
),
semantic AS (
    SELECT id, row_number() OVER (ORDER BY distance) AS rank
    FROM (
        SELECT c.id, c.embedding <=> $3 AS distance
        FROM chunks c
        WHERE c.window_seconds = $4
        ORDER BY c.embedding <=> $3
        LIMIT $5
    ) nearest
),
fused AS (
    SELECT id,
           coalesce($6::float8 / ($8 + keyword.rank), 0) + coalesce($7::float8 / ($8 + semantic.rank), 0) AS score
    FROM keyword
    FULL OUTER JOIN semantic USING (id)
)
SELECT c.id, c.transcript_id, t.audio_file, c.window_seconds,
       c.start_seconds, c.end_seconds, c.speakers, c.text, fused.score
FROM fused
JOIN chunks c ON c.id = fused.id
JOIN transcripts t ON t.id = c.transcript_id
ORDER BY fused.score DESC
"""


def overlaps(a: Chunk, b: Chunk, min_overlap_seconds: float) -> bool:
    overlap = min(a.end_seconds, b.end_seconds) - max(a.start_seconds, b.start_seconds)
    return a.transcript_id == b.transcript_id and overlap >= min_overlap_seconds


def to_chunk(row: asyncpg.Record) -> Chunk:
    return Chunk(
        chunk_ids=[row["id"]],
        transcript_id=row["transcript_id"],
        audio_file=row["audio_file"],
        window_seconds=row["window_seconds"],
        start_seconds=row["start_seconds"],
        end_seconds=row["end_seconds"],
        speakers=list(row["speakers"]),
        text=row["text"],
        score=row["score"],
    )


def merge_overlapping(chunks: list[Chunk], min_overlap: float, max_length: float) -> list[Chunk]:
    merged: list[Chunk] = []
    for chunk in sorted(chunks, key=lambda c: (c.transcript_id, c.start_seconds)):
        last = merged[-1] if merged else None
        if (
            last is not None
            and overlaps(last, chunk, min_overlap)
            and max(last.end_seconds, chunk.end_seconds) - last.start_seconds <= max_length
        ):
            last.chunk_ids.extend(chunk.chunk_ids)
            last.end_seconds = max(last.end_seconds, chunk.end_seconds)
            last.score = max(last.score, chunk.score)
        else:
            merged.append(chunk)
    return merged


async def search_window(
    pool: asyncpg.Pool, query: str, query_embedding: list[float], window: WindowSettings
) -> list[Chunk]:
    config = settings.search
    rows = await pool.fetch(
        HYBRID_SEARCH_SQL,
        query,
        config.text_search_language,
        query_embedding,
        window.length_seconds,
        config.candidates_per_search,
        config.text_weight,
        config.semantic_weight,
        config.rrf_k,
    )
    chunks = [to_chunk(row) for row in rows]
    if not config.merge_overlaps:
        return chunks
    return merge_overlapping(chunks, window.merge_overlap_seconds, window.max_merged_seconds)


async def rebuild_merged_text(pool: asyncpg.Pool, chunks: list[Chunk]) -> None:
    merged = [chunk for chunk in chunks if len(chunk.chunk_ids) > 1]
    if not merged:
        return
    rows = await pool.fetch(
        "SELECT id, words FROM transcripts WHERE id = ANY($1)",
        list({chunk.transcript_id for chunk in merged}),
    )
    words_by_transcript = {row["id"]: [SpeakerWord(**word) for word in row["words"]] for row in rows}
    for chunk in merged:
        words = words_between(words_by_transcript[chunk.transcript_id], chunk.start_seconds, chunk.end_seconds)
        chunk.text = render_text(words)
        chunk.speakers = speakers_in(words)


def drop_overlapping(ranked: list[Chunk], min_overlap: float, top_k: int) -> list[Chunk]:
    kept: list[Chunk] = []
    for chunk in ranked:
        if len(kept) == top_k:
            break
        if not any(overlaps(chunk, other, min_overlap) for other in kept):
            kept.append(chunk)
    return kept


async def search(pool: asyncpg.Pool, query: str, top_k: int) -> list[Chunk]:
    [query_embedding] = await embed_texts([query], is_query=True)
    per_window = await asyncio.gather(
        *(search_window(pool, query, query_embedding, window) for window in settings.windows)
    )
    candidates = [chunk for chunks in per_window for chunk in chunks]
    await rebuild_merged_text(pool, candidates)

    scores = await rerank_scores(query, candidates)
    for chunk, score in zip(candidates, scores, strict=True):
        chunk.rerank_score = score
    candidates.sort(key=lambda chunk: chunk.rerank_score, reverse=True)
    return drop_overlapping(candidates, settings.search.dedupe_overlap_seconds, top_k)
