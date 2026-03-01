from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenizedExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


def is_t5_model(model_name: str) -> bool:
    name = model_name.lower()
    return name.startswith("t5") or "/t5" in name


def build_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Missing dependency `transformers`. Install it in backend/requirements.txt") from e
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def tokenize_pair(
    tokenizer: Any,
    *,
    model_name: str,
    article: str,
    summary: str,
    max_input_length: int,
    max_summary_length: int,
) -> TokenizedExample:
    if is_t5_model(model_name):
        article = "summarize: " + article

    model_inputs = tokenizer(
        article,
        padding="max_length",
        truncation=True,
        max_length=max_input_length,
        return_attention_mask=True,
    )

    # Transformers 5.x prefers `text_target=` over `as_target_tokenizer()`.
    labels = tokenizer(
        text_target=summary,
        padding="max_length",
        truncation=True,
        max_length=max_summary_length,
        return_attention_mask=False,
    )

    label_ids = labels["input_ids"]
    pad = tokenizer.pad_token_id
    label_ids = [(-100 if (pad is not None and t == pad) else t) for t in label_ids]

    return TokenizedExample(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        labels=label_ids,
    )


def approx_token_len(tokenizer: Any, text: str, *, cap: int) -> int:
    """Fast-ish token length check.

    We truncate at cap+1 so we can detect over-limit without tokenizing huge texts.
    """
    out = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=cap + 1,
    )
    return len(out["input_ids"])
