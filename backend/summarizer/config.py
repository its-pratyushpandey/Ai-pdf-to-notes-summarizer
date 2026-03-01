from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SummarizerConfig:
    # A small, already-summarization-capable checkpoint to keep training/inference practical.
    base_model_name: str = "t5-small"

    # Where fine-tuned artifacts are saved/loaded.
    # NOTE: Keep this inside backend/ so deployment can bundle it.
    model_dir: Path = Path(__file__).resolve().parents[1] / "models" / "summarizer"

    # Dataset defaults
    dataset_csv_fallback: Path = Path(__file__).resolve().parents[1] / ".venv" / "article_highlights.csv"
    dataset_csv_default: Path = Path(__file__).resolve().parents[1] / "data" / "article_highlights.csv"
    processed_dir: Path = Path(__file__).resolve().parents[1] / "data" / "processed"


DEFAULT_SUMMARIZER_CONFIG = SummarizerConfig()
