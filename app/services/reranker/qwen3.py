from functools import cache
from itertools import batched

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from app.core.config import settings

PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
    'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


@cache
def _tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(settings.reranker.qwen3.model, padding_side="left")


@cache
def _model() -> PreTrainedModel:
    config = settings.reranker.qwen3
    gpu_memory = {0: config.max_gpu_memory} if torch.cuda.is_available() else {}
    return AutoModelForCausalLM.from_pretrained(
        config.model,
        dtype=config.dtype,
        device_map="auto",
        max_memory=gpu_memory | {"cpu": config.max_cpu_memory},
    )


def load() -> None:
    _tokenizer()
    _model()


def _prompt(query: str, text: str) -> str:
    instruction = settings.reranker.qwen3.instruction
    return f"{PREFIX}<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {text}{SUFFIX}"


async def score(query: str, texts: list[str]) -> list[float]:
    tokenizer, model = _tokenizer(), _model()
    no, yes = tokenizer.convert_tokens_to_ids(["no", "yes"])
    shortest_first = sorted(range(len(texts)), key=lambda index: len(texts[index]))
    scores = {}
    with torch.inference_mode():
        for batch in batched(shortest_first, settings.reranker.qwen3.batch_size):
            prompts = [_prompt(query, texts[index]) for index in batch]
            inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
            logits = model(**inputs, logits_to_keep=1).logits[:, -1, [no, yes]]
            scores.update(zip(batch, logits.float().softmax(dim=-1)[:, 1].tolist()))
    return [scores[index] for index in range(len(texts))]
