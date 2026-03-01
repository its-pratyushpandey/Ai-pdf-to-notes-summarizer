from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class RequestLogRecord:
    ts: float
    latency_ms: float
    input_chars: int
    summary_chars: int
    confidence_score: float
    decoding: dict[str, Any]


class AnalyticsStore:
    def __init__(self, log_dir: Path, requests_jsonl: str, usage_json: str, last_eval_json: str):
        self.log_dir = log_dir
        self.requests_path = log_dir / requests_jsonl
        self.usage_path = log_dir / usage_json
        self.last_eval_path = log_dir / last_eval_json
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def append_request(self, rec: RequestLogRecord) -> None:
        self.requests_path.parent.mkdir(parents=True, exist_ok=True)
        with self.requests_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")

        self._update_usage(latency_ms=rec.latency_ms, summary_chars=rec.summary_chars)

    def _update_usage(self, latency_ms: float, summary_chars: int) -> None:
        now = time.time()
        base: dict[str, Any] = {
            "updated_at": now,
            "request_count": 0,
            "avg_latency_ms": 0.0,
            "avg_summary_chars": 0.0,
        }
        if self.usage_path.exists():
            try:
                base.update(json.loads(self.usage_path.read_text(encoding="utf-8")) or {})
            except Exception:
                pass

        n = int(base.get("request_count", 0))
        avg_lat = float(base.get("avg_latency_ms", 0.0))
        avg_sum = float(base.get("avg_summary_chars", 0.0))

        n2 = n + 1
        base["request_count"] = n2
        base["avg_latency_ms"] = (avg_lat * n + latency_ms) / n2
        base["avg_summary_chars"] = (avg_sum * n + summary_chars) / n2
        base["updated_at"] = now

        self.usage_path.write_text(json.dumps(base, indent=2), encoding="utf-8")

    def write_last_eval(self, metrics: dict[str, Any]) -> None:
        self.last_eval_path.write_text(json.dumps({"updated_at": time.time(), **metrics}, indent=2), encoding="utf-8")

    def read_last_eval(self) -> Optional[dict[str, Any]]:
        if not self.last_eval_path.exists():
            return None
        try:
            return json.loads(self.last_eval_path.read_text(encoding="utf-8"))
        except Exception:
            return None
