from __future__ import annotations

import re


_PARA_SPLIT_RE = re.compile(r"\n\s*\n+")


def split_paragraphs(text: str) -> list[str]:
    """Split into logical paragraphs while preserving punctuation."""
    if not text:
        return []
    parts = [p.strip() for p in _PARA_SPLIT_RE.split(text) if p.strip()]
    return parts


def join_paragraphs(paragraphs: list[str]) -> str:
    return "\n\n".join([p for p in paragraphs if p])
