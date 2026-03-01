from __future__ import annotations

from typing import Any


def build_rouge_metric():
    """Lazy-load ROUGE metric to keep backend import-time light."""
    import evaluate

    return evaluate.load("rouge")


def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, Any]:
    rouge = build_rouge_metric()
    return rouge.compute(predictions=predictions, references=references, use_stemmer=True)
