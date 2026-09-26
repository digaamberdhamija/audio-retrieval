import json

import asyncpg
from pgvector.asyncpg import register_vector

from app.core.config import settings

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS transcripts (
    id SERIAL PRIMARY KEY,
    audio_file TEXT NOT NULL UNIQUE,
    transcript_path TEXT NOT NULL,
    words JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    transcript_id INT NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
    window_seconds INT NOT NULL,
    start_seconds DOUBLE PRECISION NOT NULL,
    end_seconds DOUBLE PRECISION NOT NULL,
    speakers TEXT[] NOT NULL,
    text TEXT NOT NULL,
    text_search TSVECTOR GENERATED ALWAYS AS (to_tsvector('{language}', text)) STORED,
    embedding VECTOR({dimensions}) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_window_idx ON chunks (window_seconds);
CREATE INDEX IF NOT EXISTS chunks_text_search_idx ON chunks USING GIN (text_search);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING HNSW (embedding vector_cosine_ops);
"""


def _connection_args() -> dict:
    db = settings.database
    return {
        "host": db.host,
        "port": db.port,
        "database": db.name,
        "user": db.user,
        "password": db.password,
    }


async def _init_connection(conn: asyncpg.Connection) -> None:
    await register_vector(conn)
    await conn.set_type_codec("jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")


async def create_schema() -> None:
    conn = await asyncpg.connect(**_connection_args())
    try:
        await conn.execute(
            SCHEMA.format(
                language=settings.search.text_search_language,
                dimensions=settings.embedding.dimensions,
            )
        )
    finally:
        await conn.close()


async def create_pool() -> asyncpg.Pool:
    await create_schema()
    return await asyncpg.create_pool(
        **_connection_args(),
        init=_init_connection,
        server_settings={"hnsw.iterative_scan": "strict_order"},
    )
