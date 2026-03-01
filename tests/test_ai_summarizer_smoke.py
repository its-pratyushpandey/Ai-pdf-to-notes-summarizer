from __future__ import annotations

from pathlib import Path

from ai_summarizer.preprocessing.cleaning import clean_text
from ai_summarizer.utils.analytics import AnalyticsStore, RequestLogRecord


def test_clean_text_strips_html_and_normalizes_ws() -> None:
    raw = "<html><body><script>1</script><p>Hello</p>   <p>World</p></body></html>"
    out = clean_text(raw)
    assert "script" not in out.lower()
    assert "hello" in out.lower()
    assert "world" in out.lower()
    assert "  " not in out


def test_analytics_store_writes_files(tmp_path: Path) -> None:
    store = AnalyticsStore(log_dir=tmp_path, requests_jsonl="requests.jsonl", usage_json="usage.json", last_eval_json="last_eval.json")
    store.append_request(
        RequestLogRecord(
            ts=123.0,
            latency_ms=10.0,
            input_chars=100,
            summary_chars=20,
            confidence_score=0.5,
            decoding={"num_beams": 4},
        )
    )
    assert (tmp_path / "requests.jsonl").exists()
    assert (tmp_path / "usage.json").exists()
