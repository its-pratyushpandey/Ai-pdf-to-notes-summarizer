from __future__ import annotations

from typing import Any, Callable, Optional


def build_compute_metrics(
    tokenizer: Any,
    *,
    bertscore_model_type: Optional[str] = None,
    bertscore_lang: str = "en",
) -> Callable[[Any], dict[str, float]]:
    try:
        import evaluate
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Missing dependency `evaluate`. Install: pip install evaluate rouge-score bert-score") from e

    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    def _compute(eval_preds: Any) -> dict[str, float]:
        preds, labels = eval_preds

        import numpy as np

        preds = np.array(preds)
        labels = np.array(labels)

        # Some transformer versions can produce negative IDs in preds; sanitize before decode.
        pad = tokenizer.pad_token_id
        if pad is None:
            pad = 0
        preds = np.where(preds < 0, pad, preds)

        # Replace -100 in labels with pad for decoding.
        labels = np.where(labels == -100, pad, labels)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        rouge_res = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        # BERTScore returns per-example arrays; report mean.
        # NOTE: `evaluate`/`bert-score` defaults to `roberta-large` if model_type isn't provided.
        bert_kwargs: dict[str, Any] = {"lang": bertscore_lang}
        if bertscore_model_type:
            bert_kwargs["model_type"] = bertscore_model_type

        bert_res = bertscore.compute(predictions=decoded_preds, references=decoded_labels, **bert_kwargs)
        bert_f1 = float(sum(bert_res["f1"]) / max(1, len(bert_res["f1"])))

        return {
            "rouge1": float(rouge_res.get("rouge1", 0.0)),
            "rouge2": float(rouge_res.get("rouge2", 0.0)),
            "rougeL": float(rouge_res.get("rougeL", 0.0)),
            "bertscore_f1": bert_f1,
        }

    return _compute
