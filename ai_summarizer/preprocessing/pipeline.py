from __future__ import annotations

import random
from typing import Any

from ai_summarizer.config import Settings
from ai_summarizer.preprocessing.augmentation import random_sentence_dropout
from ai_summarizer.preprocessing.cleaning import clean_text
from ai_summarizer.preprocessing.segmentation import join_paragraphs, split_paragraphs
from ai_summarizer.preprocessing.tokenization import approx_token_len, tokenize_pair


def preprocess_and_tokenize_dataset(ds: Any, tokenizer: Any, settings: Settings, *, split_name: str, is_train: bool):
    """Clean, filter, (optionally) augment, segment paragraphs, and tokenize."""
    rng = random.Random(settings.project.seed + (0 if split_name == "train" else 1))
    dropout_p = settings.preprocessing.sentence_dropout_prob if is_train else 0.0

    min_a = settings.filtering.min_article_tokens
    max_a = settings.filtering.max_article_tokens
    min_s = settings.filtering.min_summary_tokens
    max_s = settings.filtering.max_summary_tokens

    max_in = settings.tokenization.max_input_length
    max_out = settings.tokenization.max_summary_length
    model_name = settings.tokenization.model_name
    lowercase = settings.preprocessing.lowercase

    def _map(ex: dict[str, Any]) -> dict[str, Any]:
        article = clean_text(ex["article"], lowercase=lowercase)
        summary = clean_text(ex["summary"], lowercase=lowercase)

        # Paragraph segmentation (kept mainly to satisfy the requirement; we join back for tokenization).
        paras = split_paragraphs(article)
        article = join_paragraphs(paras) if paras else article

        if dropout_p > 0.0:
            article = random_sentence_dropout(article, dropout_p, rng=rng)

        # Filtering based on approximate token length.
        a_len = approx_token_len(tokenizer, article, cap=max_a)
        s_len = approx_token_len(tokenizer, summary, cap=max_s)
        if a_len < min_a or a_len > max_a:
            return {"_drop": True, "input_ids": [], "attention_mask": [], "labels": []}
        if s_len < min_s or s_len > max_s:
            return {"_drop": True, "input_ids": [], "attention_mask": [], "labels": []}

        tok = tokenize_pair(
            tokenizer,
            model_name=model_name,
            article=article,
            summary=summary,
            max_input_length=max_in,
            max_summary_length=max_out,
        )
        return {
            "input_ids": tok.input_ids,
            "attention_mask": tok.attention_mask,
            "labels": tok.labels,
            "_drop": False,
        }

    mapped = ds.map(_map, remove_columns=[c for c in ds.column_names if c not in ("article", "summary")])
    mapped = mapped.filter(lambda ex: not ex.get("_drop", False))
    mapped = mapped.remove_columns([c for c in mapped.column_names if c == "_drop"])
    mapped.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return mapped
