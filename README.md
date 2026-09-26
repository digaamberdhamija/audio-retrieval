# g2-hackathon

FastAPI service that transcribes 2-speaker audio with Gemini, then indexes the transcripts for hybrid (Postgres full-text + pgvector) retrieval over sliding windows of several lengths.

All settings live in `config.yaml`; `GEMINI_API_KEY` is read from `.env`. The design, retrieval strategy and database schema are explained in [docs/engineering.md](docs/engineering.md). How AI was used on this project is documented in [docs/ai-usage.md](docs/ai-usage.md). A short slide deck of the approach is in [docs/audio-retrieval-deck.pptx](docs/audio-retrieval-deck.pptx), and [docs/prototype.html](docs/prototype.html) shows the approach with real captured responses from each API (open it in a browser).

## Run it locally

### Prerequisites

- Python 3.13 and [uv](https://docs.astral.sh/uv/)
- Docker, for Postgres with the pgvector extension
- A Gemini API key
- Optional: an NVIDIA GPU for the reranker. It runs on the CPU otherwise.

### 1. Install dependencies

```bash
uv sync
```

On Windows and Linux this installs the CUDA 12.6 build of torch.

### 2. Add your Gemini API key

```bash
cp .env.example .env
```

Then replace the placeholder in `.env` with your key.

### 3. Start Postgres with pgvector

```bash
docker run -d --name pgvector -p 5433:5432 -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=app pgvector/pgvector:pg18
```

This matches the `database` section of `config.yaml` (localhost:5433, database `app`, user and password `postgres`). If you already have a Postgres container with pgvector running, skip this step and point the `database` section at it instead. The service enables the extension and creates its tables on startup.

### 4. Start the API

```bash
uv run uvicorn app.main:app --port 8000
```

On first start the reranker model is downloaded from Hugging Face (about 600 MB) and loaded; later starts reuse the cached copy. Tables are created on startup. Interactive API docs are at http://localhost:8000/docs.

### 5. Transcribe, ingest and search

Put your two-speaker recordings in `audios/` (wav, mp3, m4a, flac, ogg, aac, aiff, opus or webm, up to 30 minutes each), then call the three endpoints below in order.

On the Gemini free tier, send transcription requests one at a time, and expect ingestion to fail with a 429 for recordings longer than about 14 minutes: each minute of audio needs about 7 embeddings, and the free tier allows 100 per minute.

## Layout

```
app/
├── main.py          FastAPI app
├── core/            config, database, data models
├── api/             routes: /transcribe, /ingest, /search
└── services/        transcription, chunking, embeddings, ingestion, search, reranker
evals/               recall@k script and labeled dataset
docs/                engineering notes, AI usage, slide deck
```

## Transcribe an audio file

Writes `<audio name>.json` to `transcription.output_dir`. Diarized transcription can take a minute or more.

```bash
curl -X POST localhost:8000/transcribe -H "Content-Type: application/json" -d '{"audio_path": "audios/ai_podcast.wav"}'
```

## Ingest a transcript

```bash
curl -X POST localhost:8000/ingest -H "Content-Type: application/json" -d '{"transcript_path": "outputs/ai_podcast.json"}'
```

Re-ingesting the same audio file replaces its chunks.

## Search

```bash
curl -X POST localhost:8000/search -H "Content-Type: application/json" -d '{"query": "why did NVIDIA start with video games", "top_k": 5}'
```

Pipeline per query:

1. For each window length, run full-text and semantic search and fuse them with weighted RRF.
2. Within each length, merge results whose time ranges overlap by at least `merge_overlap_seconds`, capped at `max_merged_seconds`.
3. Rerank all merged results together with a local ColBERT model (`app/services/reranker.py`, `lightonai/GTE-ModernColBERT-v1` via PyLate).
4. Drop results overlapping a higher-ranked one by at least `dedupe_overlap_seconds` and return the top k.

## Evaluate recall@k

With the server running:

```bash
uv run python -m evals.recall_at_k
```

Queries and their answer spans live in `evals/dataset.yaml`. A retrieved chunk is a hit when it is from the same audio file and covers at least `eval.min_span_coverage` of the answer span. Queries that hit the Gemini rate limit are skipped and counted.

## Changing settings

Most settings in `config.yaml` take effect on the next restart. If you change `embedding.dimensions` or `search.text_search_language`, drop the `chunks` and `transcripts` tables and re-ingest.
