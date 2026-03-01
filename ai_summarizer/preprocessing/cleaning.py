from __future__ import annotations

import re


_WS_RE = re.compile(r"\s+")


def strip_html(text: str) -> str:
    """Remove HTML tags, scripts/styles, and return visible text."""
    if "<" not in text and ">" not in text:
        return text
    try:
        from bs4 import BeautifulSoup
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Missing dependency `beautifulsoup4`. Install: pip install beautifulsoup4") from e

    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "meta", "noscript"]):
        tag.decompose()
    return soup.get_text(" ")


def normalize_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def clean_text(text: str, *, lowercase: bool = False) -> str:
    text = strip_html(text)
    text = normalize_whitespace(text)
    if lowercase:
        text = text.lower()
    return text
