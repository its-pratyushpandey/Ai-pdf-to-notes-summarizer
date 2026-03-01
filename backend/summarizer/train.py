from __future__ import annotations

import argparse
import json
import logging
import os
import inspect
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import DEFAULT_SUMMARIZER_CONFIG
from .data import load_article_highlights_csv
from .metrics import compute_rouge


logger = logging.getLogger("summarizer.train")


@dataclass
class TrainArgs:
    base_model_name: str
    output_dir: Path
    csv_path: Optional[Path]
    processed_dir: Optional[Path]
    max_train_samples: Optional[int]
    max_eval_samples: Optional[int]
    seed: int
    val_ratio: float
    epochs: float
    learning_rate: float
    batch_size: int
    grad_accum: int
    max_input_tokens: int
    max_target_tokens: int
    weight_decay: float
    warmup_ratio: float
    early_stopping_patience: int


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_rows(train_args: TrainArgs) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if train_args.processed_dir:
        train_path = train_args.processed_dir / "train.jsonl"
        val_path = train_args.processed_dir / "val.jsonl"
        if train_path.exists() and val_path.exists():
            train_rows = _read_jsonl(train_path)
            val_rows = _read_jsonl(val_path)
            return train_rows, val_rows

    # Fallback: split directly from CSV
    cfg = DEFAULT_SUMMARIZER_CONFIG
    csv_path = train_args.csv_path or (cfg.dataset_csv_default if cfg.dataset_csv_default.exists() else cfg.dataset_csv_fallback)
    df = load_article_highlights_csv(csv_path)
    if train_args.max_train_samples or train_args.max_eval_samples:
        # Keep deterministic slice before split if limiting
        limit = (train_args.max_train_samples or 0) + (train_args.max_eval_samples or 0)
        if limit > 0:
            df = df.head(limit)

    # local split
    import random

    idx = list(range(len(df)))
    rng = random.Random(train_args.seed)
    rng.shuffle(idx)
    val_n = max(1, int(len(df) * train_args.val_ratio))
    val_idx = idx[:val_n]
    train_idx = idx[val_n:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    train_rows = train_df[[c for c in ("article", "summary") if c in train_df.columns]].to_dict(orient="records")
    val_rows = val_df[[c for c in ("article", "summary") if c in val_df.columns]].to_dict(orient="records")
    return train_rows, val_rows


def main() -> int:
    cfg = DEFAULT_SUMMARIZER_CONFIG

    parser = argparse.ArgumentParser(description="Fine-tune a transformer summarization model")
    parser.add_argument("--base-model", type=str, default=cfg.base_model_name)
    parser.add_argument("--output-dir", type=str, default=str(cfg.model_dir))
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--processed-dir", type=str, default=None, help="Directory containing train.jsonl/val.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-target-tokens", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=200)
    args = parser.parse_args()

    train_args = TrainArgs(
        base_model_name=args.base_model,
        output_dir=Path(args.output_dir),
        csv_path=Path(args.csv) if args.csv else None,
        processed_dir=Path(args.processed_dir) if args.processed_dir else None,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        seed=args.seed,
        val_ratio=args.val_ratio,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_input_tokens=args.max_input_tokens,
        max_target_tokens=args.max_target_tokens,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        early_stopping_patience=args.early_stopping_patience,
    )

    logging.basicConfig(level=logging.INFO)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Heavy imports kept inside main so backend imports stay fast.
    import torch
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    train_rows, val_rows = _load_rows(train_args)
    if train_args.max_train_samples is not None:
        train_rows = train_rows[: train_args.max_train_samples]
    if train_args.max_eval_samples is not None:
        val_rows = val_rows[: train_args.max_eval_samples]

    tokenizer = AutoTokenizer.from_pretrained(train_args.base_model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(train_args.base_model_name)

    def _maybe_prefix(article: str) -> str:
        # T5-style models typically require a task prefix for best performance.
        if "t5" in train_args.base_model_name.lower() and not article.lstrip().lower().startswith("summarize:"):
            return f"summarize: {article}"
        return article

    def preprocess_fn(ex: dict[str, Any]) -> dict[str, Any]:
        model_inputs = tokenizer(
            _maybe_prefix(ex["article"]),
            max_length=train_args.max_input_tokens,
            truncation=True,
        )

        tok_sig = inspect.signature(tokenizer.__call__).parameters
        if "text_target" in tok_sig:
            labels = tokenizer(
                text_target=ex["summary"],
                max_length=train_args.max_target_tokens,
                truncation=True,
            )
        else:
            labels = tokenizer(
                ex["summary"],
                max_length=train_args.max_target_tokens,
                truncation=True,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Minimal torch dataset to avoid requiring `datasets`.
    class RowsDataset(torch.utils.data.Dataset):
        def __init__(self, rows: list[dict[str, Any]]):
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            return preprocess_fn(self.rows[idx])

    train_ds = RowsDataset(train_rows)
    val_ds = RowsDataset(val_rows)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Scale eval/save cadence down for small datasets so we actually evaluate at least once.
    steps_per_epoch = max(1, math.ceil(len(train_rows) / max(1, train_args.batch_size) / max(1, train_args.grad_accum)))
    eval_steps = int(min(200, max(10, steps_per_epoch)))
    save_steps = eval_steps
    logging_steps = int(min(50, max(5, eval_steps // 2)))

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Some transformers versions can pad predictions with -100; batch_decode cannot handle negatives.
        if hasattr(predictions, "detach"):
            predictions = predictions.detach().cpu().numpy()
        predictions = np.asarray(predictions)
        if predictions.size:
            predictions = np.where(predictions < 0, tokenizer.pad_token_id, predictions)

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]
        rouge = compute_rouge(decoded_preds, decoded_labels)

        # Extra useful stats
        pred_lens = [np.count_nonzero(p != tokenizer.pad_token_id) for p in predictions]
        rouge["gen_len"] = float(np.mean(pred_lens))
        # Trainer expects scalar floats
        return {k: float(v) if not isinstance(v, dict) else v for k, v in rouge.items()}

    training_args = Seq2SeqTrainingArguments(
        **(lambda: (
            # Transformers 5.x changed / removed some TrainingArguments fields.
            # Build kwargs then filter by the installed signature for compatibility.
            (lambda allowed: {
                k: v
                for k, v in {
                    "output_dir": str(train_args.output_dir),
                    # strategy naming differs across versions
                    ("evaluation_strategy" if "evaluation_strategy" in allowed else "eval_strategy"): "steps",
                    "eval_steps": eval_steps,
                    "save_steps": save_steps,
                    "logging_steps": logging_steps,
                    "save_total_limit": 2,
                    "learning_rate": train_args.learning_rate,
                    "per_device_train_batch_size": train_args.batch_size,
                    "per_device_eval_batch_size": train_args.batch_size,
                    "gradient_accumulation_steps": train_args.grad_accum,
                    "num_train_epochs": train_args.epochs,
                    "weight_decay": train_args.weight_decay,
                    "predict_with_generate": True,
                    "generation_max_length": train_args.max_target_tokens,
                    "warmup_ratio": train_args.warmup_ratio,
                    "fp16": torch.cuda.is_available(),
                    "seed": train_args.seed,
                    "load_best_model_at_end": True,
                    "metric_for_best_model": "rougeL",
                    "greater_is_better": True,
                    # disable external loggers by default
                    "report_to": [],
                }.items()
                if k in allowed
            })({
                name for name in inspect.signature(Seq2SeqTrainingArguments.__init__).parameters.keys()
            })
        ))()
    )

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": collator,
        "compute_metrics": compute_metrics,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=train_args.early_stopping_patience)],
    }

    # transformers 5.x renamed tokenizer -> processing_class.
    trainer_sig = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "tokenizer" in trainer_sig:
        trainer_kwargs["tokenizer"] = tokenizer
    if "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer

    # Filter unknown kwargs for forward-compat.
    trainer = Seq2SeqTrainer(**{k: v for k, v in trainer_kwargs.items() if k in trainer_sig})

    logger.info("Starting training")
    trainer.train()

    logger.info("Saving model -> %s", train_args.output_dir)
    train_args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(train_args.output_dir))
    tokenizer.save_pretrained(str(train_args.output_dir))

    logger.info("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
