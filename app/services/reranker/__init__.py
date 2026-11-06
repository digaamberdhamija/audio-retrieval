from importlib import import_module

from app.core.config import settings
from app.core.models import Chunk

_strategy = import_module(f"app.services.reranker.{settings.reranker.strategy}")


def load_reranker() -> None:
    _strategy.load()


async def rerank_scores(query: str, chunks: list[Chunk]) -> list[float]:
    if not chunks:
        return []
    return await _strategy.score(query, [chunk.text for chunk in chunks])
