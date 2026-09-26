from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.models import Chunk
from app.services.search import search

router = APIRouter()


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=settings.search.top_k, ge=1)


@router.post("/search")
async def post_search(body: SearchRequest, request: Request) -> list[Chunk]:
    return await search(request.app.state.pool, body.query, body.top_k)
