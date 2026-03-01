from __future__ import annotations

from summarizer.service import _chunk_text_by_tokens


class DummyTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False):
        # Treat each word as a token to keep the test deterministic.
        return {"input_ids": list(range(len(text.split())))}


def test_chunking_splits_long_text():
    tok = DummyTokenizer()
    text = " ".join(["word"] * 200)
    chunks = _chunk_text_by_tokens(text, tok, max_input_tokens=50)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 60 for c in chunks)  # fuzzy bound due to greedy splitting


def test_chunking_empty_text():
    tok = DummyTokenizer()
    assert _chunk_text_by_tokens("   ", tok, max_input_tokens=50) == []
