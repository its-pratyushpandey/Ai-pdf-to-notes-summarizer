from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import DEFAULT_SUMMARIZER_CONFIG, SummarizerConfig


logger = logging.getLogger("summarizer.service")


@dataclass
class SummarizeResult:
    summary: str
    model_id: str
    elapsed_ms: float
    chunk_count: int


def _pick_model_path(cfg: SummarizerConfig) -> tuple[str, Optional[Path]]:
    # Prefer locally fine-tuned model if present.
    if (cfg.model_dir / "config.json").exists():
        return str(cfg.model_dir), cfg.model_dir
    return cfg.base_model_name, None


def _chunk_text_by_tokens(text: str, tokenizer, max_input_tokens: int) -> list[str]:
    """Split long text into chunks roughly limited by token count."""
    text = text.strip()
    if not text:
        return []

    words = text.split()
    chunks: list[str] = []
    current: list[str] = []

    # Greedy word accumulation with token-based check
    for w in words:
        current.append(w)
        if len(current) < 20:
            continue
        candidate = " ".join(current)
        token_len = len(tokenizer(candidate, add_special_tokens=False)["input_ids"])
        if token_len >= max_input_tokens:
            # move last word to next chunk
            current.pop()
            if current:
                chunks.append(" ".join(current))
            current = [w]

    if current:
        chunks.append(" ".join(current))
    return chunks


class SummarizerService:
    """Loads a summarization model (fine-tuned if available) and serves fast inference."""

    def __init__(
        self,
        cfg: SummarizerConfig = DEFAULT_SUMMARIZER_CONFIG,
        device: Optional[int] = None,
    ) -> None:
        self.cfg = cfg
        self._device = device
        self._tokenizer = None
        self._model = None
        self._torch_device = None
        self._model_id = None

    def load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        model_id, local_path = _pick_model_path(self.cfg)
        logger.info("Loading summarizer model: %s", model_id)

        # Heavy imports deferred.
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        # device: None => auto (cuda if available), otherwise CPU.
        if self._device is None:
            self._torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            # Keep backward-compat: allow pipeline-style int device
            if self._device >= 0 and torch.cuda.is_available():
                self._torch_device = torch.device(f"cuda:{self._device}")
            else:
                self._torch_device = torch.device("cpu")

        self._tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self._model.to(self._torch_device)
        self._model.eval()

        self._model_id = model_id

    @property
    def model_id(self) -> str:
        if not self._model_id:
            self.load()
        return str(self._model_id)

    def summarize(
        self,
        text: str,
        *,
        min_length: int = 30,
        max_length: int = 120,
        max_input_tokens: int = 768,
    ) -> SummarizeResult:
        self.load()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch_device is not None

        import torch

        start = time.perf_counter()
        chunks = _chunk_text_by_tokens(text, self._tokenizer, max_input_tokens=max_input_tokens)
        if not chunks:
            return SummarizeResult(summary="", model_id=self.model_id, elapsed_ms=0.0, chunk_count=0)

        def _maybe_prefix(task_text: str) -> str:
            # T5-style models expect a task prefix.
            mid = (self._model_id or "").lower()
            if "t5" in mid and not task_text.lstrip().lower().startswith("summarize:"):
                return f"summarize: {task_text.strip()}"
            return task_text

        def _summarize_one(chunk: str) -> str:
            chunk = _maybe_prefix(chunk)
            inputs = self._tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens,
            )
            inputs = {k: v.to(self._torch_device) for k, v in inputs.items()}
            with torch.inference_mode():
                output_ids = self._model.generate(
                    **inputs,
                    num_beams=4,
                    do_sample=False,
                    min_length=min_length,
                    max_length=max_length,
                )
            return self._tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        # First pass: summarize chunks
        summaries: list[str] = [_summarize_one(chunk) for chunk in chunks]

        final = " ".join(summaries).strip()

        # If we chunked, do a second pass to produce a coherent final summary.
        if len(summaries) > 1:
            final = _summarize_one(final)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return SummarizeResult(
            summary=final,
            model_id=self.model_id,
            elapsed_ms=elapsed_ms,
            chunk_count=len(chunks),
        )


__all__ = ["SummarizerService", "SummarizeResult"]
