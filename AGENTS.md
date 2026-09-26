# AGENTS.md

Hybrid search over audio conversations between two speakers. A FastAPI service transcribes audio (speakers + word timestamps), cuts transcripts into overlapping time windows of several lengths, and searches them with Postgres full-text search plus pgvector. Design notes are in `docs/engineering.md`.

## Commands

```bash
uv run uvicorn app.main:app --port 8000   # run the API
uv run python -m evals.recall_at_k        # recall@k eval, needs the API running
```

## Layout

- `app/main.py`: FastAPI app, routers and error handling
- `app/core/`: config loading, database pool and schema, dataclasses
- `app/api/`: routes (`/transcribe`, `/ingest`, `/search`) with request/response models
- `app/services/`: transcription, chunking, embeddings, ingestion, search, reranker
- `evals/`: recall@k script and hand-labeled dataset
- `config.yaml`: all settings; `.env` holds only `GEMINI_API_KEY`

## Conventions

- Keep code simple and easy for a human to read. Match the style of the surrounding code.
- Use async Python throughout. Do not use threads or thread pools.
- Keep comments minimal. Do not explain design decisions in comments.
- Put every setting in `config.yaml`, never hard-coded. Secrets go in `.env`.
- Ask clarifying questions when a requirement is ambiguous instead of assuming.

## Things to know

- Postgres with pgvector runs on `localhost:5433` (database `app`, user and password `postgres`). Tables are created on startup. Changing the embedding dimensions or the full-text language means dropping the `chunks` and `transcripts` tables and re-ingesting.
- `embed_texts()` in `app/services/embeddings.py` is the single place to swap the embedding model. `gemini-embedding-2` ignores `task_type`; use the query/document prefixes in `config.yaml` instead.
- `rerank_scores()` in `app/services/reranker.py` scores candidates with a local ColBERT model (PyLate, `GTE-ModernColBERT-v1`). The model loads at startup and inference runs in-process, blocking the event loop for about a second per search on the GPU. torch comes from the PyTorch CUDA 12.6 index on Windows and Linux (see `pyproject.toml`); PyLate pins the torch version.
- Transcript JSON files (in `outputs/`) are the contract between transcription and ingestion: segments with a speaker, text and words, each word with start and end seconds.
- The Gemini key is on the free tier. Rate-limited calls are not retried: the API returns 429, and the eval skips those queries.
