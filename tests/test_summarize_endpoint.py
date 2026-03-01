from __future__ import annotations

from fastapi.testclient import TestClient


def test_summarize_endpoint_validates_and_returns_summary(monkeypatch):
    import server

    # Stub model load and inference to keep tests fast/offline.
    monkeypatch.setattr(server.summarizer_service, "load", lambda: None)
    monkeypatch.setattr(server, "SUMMARIZER_AVAILABLE", True)

    from summarizer.service import SummarizeResult

    def _fake_summarize(text: str, **kwargs):
        return SummarizeResult(summary="fake summary", model_id="fake/model", elapsed_ms=1.23, chunk_count=1)

    monkeypatch.setattr(server.summarizer_service, "summarize", _fake_summarize)

    client = TestClient(server.app)
    resp = client.post(
        "/api/summarize",
        json={"text": "This is a long article about something important.", "min_length": 5, "max_length": 20},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["summary"] == "fake summary"
    assert data["model_id"] == "fake/model"
    assert data["chunk_count"] == 1


def test_summarize_endpoint_rejects_empty_text():
    import server

    client = TestClient(server.app)
    resp = client.post("/api/summarize", json={"text": "  "})
    assert resp.status_code == 400


def test_summarize_endpoint_rejects_invalid_lengths():
    import server

    client = TestClient(server.app)
    resp = client.post("/api/summarize", json={"text": "x", "min_length": 50, "max_length": 10})
    assert resp.status_code == 400
