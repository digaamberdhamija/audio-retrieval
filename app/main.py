from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google.genai import errors

from app.api import ingest, search, transcribe
from app.core.db import create_pool
from app.services.reranker import load_reranker


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_reranker()
    app.state.pool = await create_pool()
    yield
    await app.state.pool.close()


app = FastAPI(title="Audio Transcript Retrieval", lifespan=lifespan)


@app.exception_handler(errors.APIError)
async def gemini_error_handler(request: Request, exc: errors.APIError) -> JSONResponse:
    status_code = 429 if exc.code == 429 else 502
    return JSONResponse(status_code=status_code, content={"detail": f"Gemini API error: {exc.message}"})


app.include_router(transcribe.router)
app.include_router(ingest.router)
app.include_router(search.router)
