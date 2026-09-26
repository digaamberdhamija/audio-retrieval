from functools import cache
from itertools import batched

from google import genai
from google.genai import types

from app.core.config import GEMINI_API_KEY, settings


@cache
def _client() -> genai.Client:
    return genai.Client(api_key=GEMINI_API_KEY)


async def embed_texts(texts: list[str], is_query: bool = False) -> list[list[float]]:
    config = settings.embedding
    prefix = config.query_prefix if is_query else config.document_prefix
    vectors = []
    for batch in batched(texts, config.batch_size):
        response = await _client().aio.models.embed_content(
            model=config.model,
            contents=[types.Content(parts=[types.Part(text=prefix + text)]) for text in batch],
            config=types.EmbedContentConfig(output_dimensionality=config.dimensions),
        )
        vectors.extend(embedding.values for embedding in response.embeddings)
    return vectors
