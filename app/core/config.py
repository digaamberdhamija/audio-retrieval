import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


class DatabaseSettings(BaseModel):
    host: str
    port: int
    name: str
    user: str
    password: str


class TranscriptionSettings(BaseModel):
    model: str
    language_codes: list[str]
    expected_speakers: int
    request_timeout_seconds: int
    output_dir: str


class EmbeddingSettings(BaseModel):
    model: str
    dimensions: int
    batch_size: int
    query_prefix: str = ""
    document_prefix: str = ""


class WindowSettings(BaseModel):
    length_seconds: int
    stride_seconds: float
    merge_overlap_seconds: float
    max_merged_seconds: float


class SearchSettings(BaseModel):
    text_search_language: str
    candidates_per_search: int
    text_weight: float
    semantic_weight: float
    rrf_k: int
    dedupe_overlap_seconds: float
    top_k: int


class RerankerSettings(BaseModel):
    model: str
    query_length: int
    document_length: int
    batch_size: int
    gpu_dtype: str


class EvalSettings(BaseModel):
    api_url: str
    dataset_path: str
    k_values: list[int]
    min_span_coverage: float


class Settings(BaseModel):
    database: DatabaseSettings
    transcription: TranscriptionSettings
    embedding: EmbeddingSettings
    windows: list[WindowSettings]
    search: SearchSettings
    reranker: RerankerSettings
    eval: EvalSettings


settings = Settings(**yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8")))
