from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


_WHITESPACE_RE = re.compile(r"\s+")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def clean_text(text: str) -> str:
    """Lightweight cleaning suitable for summarization training.

    - Removes control characters
    - Normalizes whitespace
    - Strips leading/trailing spaces

    We intentionally do NOT remove normal punctuation because it helps model quality.
    """
    if text is None:
        return ""
    text = str(text)
    text = _CONTROL_CHARS_RE.sub(" ", text)
    text = text.replace("\u00a0", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _drop_nulls(df: pd.DataFrame, required_cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in required_cols:
        if col not in out.columns:
            raise ValueError(f"Dataset is missing required column '{col}'. Found: {list(out.columns)}")
        out[col] = out[col].astype(str)
        out[col] = out[col].map(clean_text)
    # Drop empty rows after cleaning
    mask = True
    for col in required_cols:
        mask = mask & (out[col].str.len() > 0)
    return out.loc[mask].reset_index(drop=True)


@dataclass(frozen=True)
class ArticleSummaryExample:
    article: str
    summary: str
    url: Optional[str] = None


def load_article_highlights_csv(csv_path: Path) -> pd.DataFrame:
    """Load the CSV dataset with columns: url, article, highlights."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # Normalize expected schema
    if "highlights" in df.columns and "summary" not in df.columns:
        df = df.rename(columns={"highlights": "summary"})
    df = _drop_nulls(df, required_cols=("article", "summary"))

    # De-duplicate exact duplicates
    subset_cols = ["article", "summary"]
    if "url" in df.columns:
        subset_cols = ["url", "article", "summary"]
    df = df.drop_duplicates(subset=subset_cols, keep="first").reset_index(drop=True)
    return df


def iter_examples(df: pd.DataFrame, limit: Optional[int] = None) -> Iterable[ArticleSummaryExample]:
    count = 0
    for _, row in df.iterrows():
        if limit is not None and count >= limit:
            break
        url = row["url"] if "url" in row else None
        yield ArticleSummaryExample(article=row["article"], summary=row["summary"], url=url)
        count += 1
