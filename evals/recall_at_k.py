import asyncio

import httpx
import yaml

from app.core.config import PROJECT_ROOT, settings


def is_hit(result: dict, span: dict) -> bool:
    overlap = min(result["end_seconds"], span["end"]) - max(result["start_seconds"], span["start"])
    coverage = max(overlap, 0.0) / (span["end"] - span["start"])
    return result["audio_file"] == span["audio_file"] and coverage >= settings.eval.min_span_coverage


def recall_at_k(results: list[dict], relevant_spans: list[dict], k: int) -> float:
    top_k = results[:k]
    found = sum(any(is_hit(result, span) for result in top_k) for span in relevant_spans)
    return found / len(relevant_spans)


async def main() -> None:
    config = settings.eval
    dataset = yaml.safe_load((PROJECT_ROOT / config.dataset_path).read_text(encoding="utf-8"))
    recalls: dict[int, list[float]] = {k: [] for k in config.k_values}
    rate_limited = 0

    async with httpx.AsyncClient(base_url=config.api_url, timeout=60) as client:
        for item in dataset:
            response = await client.post("/search", json={"query": item["query"], "top_k": max(config.k_values)})
            if response.status_code == 429:
                rate_limited += 1
                continue
            response.raise_for_status()
            for k in config.k_values:
                recalls[k].append(recall_at_k(response.json(), item["relevant"], k))

    evaluated = len(dataset) - rate_limited
    print(f"Evaluated {evaluated} of {len(dataset)} queries ({rate_limited} skipped: rate limited)")
    if evaluated:
        for k, values in recalls.items():
            print(f"recall@{k}: {sum(values) / evaluated:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
