from __future__ import annotations

import argparse
import inspect
import logging
from pathlib import Path

from ai_summarizer.config import load_settings
from ai_summarizer.data.cnn_dailymail import load_cnn_dailymail
from ai_summarizer.evaluation.metrics import build_compute_metrics
from ai_summarizer.preprocessing.pipeline import preprocess_and_tokenize_dataset
from ai_summarizer.preprocessing.tokenization import build_tokenizer
from ai_summarizer.utils.seed import set_global_seed


def _build_model(model_name: str):
    from transformers import AutoModelForSeq2SeqLM

    return AutoModelForSeq2SeqLM.from_pretrained(model_name)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune a summarization model on CNN/DailyMail")
    ap.add_argument("--config", default=None, help="Path to config.yaml (defaults to repo root config.yaml)")
    ap.add_argument("--output-dir", default=None, help="Override training.output_dir")
    ap.add_argument("--model-name", default=None, help="Override tokenization.model_name (e.g., t5-small, t5-base, facebook/bart-large-cnn)")
    ap.add_argument("--limit-train", type=int, default=0, help="Limit train examples for quick runs")
    ap.add_argument("--limit-eval", type=int, default=0, help="Limit eval examples for quick runs")
    args = ap.parse_args()

    settings = load_settings(args.config)
    if args.output_dir:
        settings.training.output_dir = args.output_dir
    if args.model_name:
        settings.tokenization.model_name = args.model_name

    set_global_seed(settings.project.seed)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Loading dataset %s/%s", settings.dataset.hf_name, settings.dataset.hf_config)

    ds = load_cnn_dailymail(settings.dataset.hf_name, settings.dataset.hf_config)
    tokenizer = build_tokenizer(settings.tokenization.model_name)

    train_raw = ds["train"]
    val_raw = ds.get("validation") or ds.get("val")
    if val_raw is None:
        raise RuntimeError("Dataset missing validation split.")

    if args.limit_train and args.limit_train > 0:
        train_raw = train_raw.select(range(min(args.limit_train, len(train_raw))))
    if args.limit_eval and args.limit_eval > 0:
        val_raw = val_raw.select(range(min(args.limit_eval, len(val_raw))))

    logging.info("Preprocessing + tokenizing train/validation")
    train_ds = preprocess_and_tokenize_dataset(train_raw, tokenizer, settings, split_name="train", is_train=True)
    val_ds = preprocess_and_tokenize_dataset(val_raw, tokenizer, settings, split_name="validation", is_train=False)

    model = _build_model(settings.tokenization.model_name)

    from transformers import (
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    out_dir = Path(settings.training.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Transformers 5.x renamed some TrainingArguments fields.
    # Build kwargs and adapt to the installed version by inspecting the signature.
    arg_sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    arg_names = set(arg_sig.parameters.keys())

    train_kwargs: dict[str, object] = {
        "output_dir": str(out_dir),
        "num_train_epochs": settings.training.num_train_epochs,
        "per_device_train_batch_size": settings.training.per_device_train_batch_size,
        "per_device_eval_batch_size": settings.training.per_device_eval_batch_size,
        "learning_rate": settings.training.learning_rate,
        "weight_decay": settings.training.weight_decay,
        "warmup_ratio": settings.training.warmup_ratio,
        "gradient_accumulation_steps": settings.training.gradient_accumulation_steps,
        "max_grad_norm": settings.training.max_grad_norm,
        "fp16": settings.training.fp16,
        "bf16": settings.training.bf16,
        "evaluation_strategy": settings.training.evaluation_strategy,
        "save_strategy": settings.training.save_strategy,
        "load_best_model_at_end": settings.training.load_best_model_at_end,
        "metric_for_best_model": settings.training.metric_for_best_model,
        "greater_is_better": settings.training.greater_is_better,
        "predict_with_generate": True,
        "generation_max_length": settings.tokenization.max_summary_length,
        "generation_num_beams": max(1, settings.inference.default_decoding.num_beams),
        "logging_steps": 50,
        "save_total_limit": 2,
        "report_to": [],
        "seed": settings.project.seed,
        "data_seed": settings.project.seed,
        "lr_scheduler_type": "linear",
    }

    # evaluation_strategy -> eval_strategy (transformers>=5)
    if "evaluation_strategy" not in arg_names and "eval_strategy" in arg_names:
        train_kwargs["eval_strategy"] = train_kwargs.pop("evaluation_strategy")

    # Filter out any keys not accepted by this transformers version.
    train_kwargs = {k: v for k, v in train_kwargs.items() if k in arg_names}

    training_args = Seq2SeqTrainingArguments(**train_kwargs)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    compute_metrics = build_compute_metrics(
        tokenizer,
        bertscore_model_type=settings.evaluation.bertscore_model_type,
        bertscore_lang=settings.evaluation.bertscore_lang,
    )

    trainer_kwargs: dict[str, object] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "tokenizer": tokenizer,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=settings.training.early_stopping_patience)],
    }

    trainer_sig = inspect.signature(Seq2SeqTrainer.__init__)
    trainer_names = set(trainer_sig.parameters.keys())
    trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if k in trainer_names}

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    metrics = trainer.evaluate()
    logging.info("Final validation metrics: %s", metrics)


if __name__ == "__main__":
    main()
