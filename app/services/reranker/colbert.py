from functools import cache

import torch
from pylate import models, rank

from app.core.config import settings


@cache
def _model() -> models.ColBERT:
    config = settings.reranker.colbert
    on_gpu = torch.cuda.is_available()
    return models.ColBERT(
        config.model,
        device="cuda" if on_gpu else "cpu",
        query_length=config.query_length,
        document_length=config.document_length,
        model_kwargs={"dtype": config.gpu_dtype} if on_gpu else None,
    )


def load() -> None:
    _model()


async def score(query: str, texts: list[str]) -> list[float]:
    model = _model()
    query_embeddings = model.encode([query], is_query=True)
    document_embeddings = model.encode(texts, is_query=False, batch_size=settings.reranker.colbert.batch_size)
    [ranked] = rank.rerank(
        documents_ids=[list(range(len(texts)))],
        queries_embeddings=query_embeddings,
        documents_embeddings=[document_embeddings],
    )
    scores = {result["id"]: result["score"] for result in ranked}
    return [scores[index] for index in range(len(texts))]
