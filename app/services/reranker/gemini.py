from functools import cache

from google import genai
from google.genai import types
from pydantic import BaseModel

from app.core.config import GEMINI_API_KEY, settings


class PassageScore(BaseModel):
    id: int
    score: int


@cache
def _client() -> genai.Client:
    return genai.Client(api_key=GEMINI_API_KEY)


def load() -> None:
    _client()


def _prompt(query: str, texts: list[str]) -> str:
    passages = "\n\n".join(f'<passage id="{index}">\n{text}\n</passage>' for index, text in enumerate(texts))
    return f"Question: {query}\n\n{passages}"


async def score(query: str, texts: list[str]) -> list[float]:
    config = settings.reranker.gemini
    response = await _client().aio.models.generate_content(
        model=config.model,
        contents=_prompt(query, texts),
        config=types.GenerateContentConfig(
            system_instruction=config.instruction,
            response_mime_type="application/json",
            response_schema=list[PassageScore],
            thinking_config=types.ThinkingConfig(thinking_level=config.thinking_level),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    scores = {passage.id: passage.score for passage in response.parsed}
    return [float(scores.get(index, 0)) for index in range(len(texts))]
