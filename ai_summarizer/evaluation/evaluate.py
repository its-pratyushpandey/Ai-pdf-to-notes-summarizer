from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ai_summarizer.config import load_settings
from ai_summarizer.data.cnn_dailymail import load_cnn_dailymail
from ai_summarizer.preprocessing.pipeline import preprocess_and_tokenize_dataset
from ai_summarizer.preprocessing.tokenization import build_tokenizer
from ai_summarizer.utils.analytics import AnalyticsStore


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a summarization model (ROUGE + BERTScore)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--model-dir", default=None, help="Path to trained model directory (defaults to training.output_dir)")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    settings = load_settings(args.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model_dir = Path(args.model_dir or settings.training.output_dir)
    if not (model_dir / "config.json").exists():
        raise RuntimeError(f"Model dir does not look like a HF model: {model_dir}")

    from transformers import AutoModelForSeq2SeqLM
    from ai_summarizer.evaluation.metrics import build_compute_metrics

    tokenizer = build_tokenizer(str(model_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))

    ds = load_cnn_dailymail(settings.dataset.hf_name, settings.dataset.hf_config)
    test_raw = ds.get("test")
    if test_raw is None:
        raise RuntimeError("Dataset missing test split.")
    if args.limit > 0:
        test_raw = test_raw.select(range(min(args.limit, len(test_raw))))

    test_ds = preprocess_and_tokenize_dataset(test_raw, tokenizer, settings, split_name="test", is_train=False)

    from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

    tmp_args = Seq2SeqTrainingArguments(
        output_dir=str(model_dir / "_eval_tmp"),
        per_device_eval_batch_size=max(1, settings.training.per_device_eval_batch_size),
        predict_with_generate=True,
        generation_max_length=settings.tokenization.max_summary_length,
        generation_num_beams=max(1, settings.inference.default_decoding.num_beams),
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=tmp_args,
        eval_dataset=test_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        compute_metrics=build_compute_metrics(
            tokenizer,
            bertscore_model_type=settings.evaluation.bertscore_model_type,
            bertscore_lang=settings.evaluation.bertscore_lang,
        ),
    )

    metrics = trainer.evaluate()
    metrics = {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()}
    logging.info("Test metrics: %s", metrics)

    store = AnalyticsStore(
        log_dir=Path(settings.logging.log_dir),
        requests_jsonl=settings.logging.requests_jsonl,
        usage_json=settings.logging.usage_json,
        last_eval_json=settings.logging.last_eval_json,
    )
    store.write_last_eval(metrics)
    (Path(settings.logging.log_dir) / "last_eval_print.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
