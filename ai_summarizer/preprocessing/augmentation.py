from __future__ import annotations

import random
import re


_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def random_sentence_dropout(text: str, p: float, *, rng: random.Random) -> str:
    """Drop sentences at random for robustness.

    Keeps at least one sentence.
    """
    if p <= 0.0:
        return text
    sentences = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
    if len(sentences) <= 1:
        return text
    kept: list[str] = [s for s in sentences if rng.random() > p]
    if not kept:
        kept = [rng.choice(sentences)]
    return " ".join(kept)
