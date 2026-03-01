from __future__ import annotations

from summarizer.data import clean_text


def test_clean_text_normalizes_whitespace_and_controls():
    raw = "Hello\x00\x07\n\t  world\u00a0  !  "
    assert clean_text(raw) == "Hello world !"


def test_clean_text_handles_none():
    assert clean_text(None) == ""
