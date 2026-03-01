from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ai_summarizer.config import load_settings
from ai_summarizer.models.runtime import SummarizerRuntime
from ai_summarizer.utils.analytics import AnalyticsStore, RequestLogRecord
from ai_summarizer.utils.logging import setup_logging


settings = load_settings(os.environ.get("AI_SUMMARIZER_CONFIG"))
log_dir = Path(settings.logging.log_dir)
setup_logging(log_dir)
logger = logging.getLogger("ai_summarizer.api")


def _resolve_model_dir() -> Optional[Path]:
    env = os.environ.get("AI_SUMMARIZER_MODEL_DIR")
    if env:
        p = Path(env)
        return p if p.exists() else None

    # Default: if training.output_dir looks like a model folder, use it.
    p = Path(settings.training.output_dir)
    if (p / "config.json").exists():
        return p
    return None


runtime = SummarizerRuntime(
    model_name=settings.tokenization.model_name,
    model_dir=_resolve_model_dir(),
    device=settings.inference.device,
)

analytics = AnalyticsStore(
    log_dir=log_dir,
    requests_jsonl=settings.logging.requests_jsonl,
    usage_json=settings.logging.usage_json,
    last_eval_json=settings.logging.last_eval_json,
)

app = FastAPI()


class SummarizeRequest(BaseModel):
    text: str = Field(min_length=1)
    strategy: Optional[Literal["greedy", "beam", "topk", "topp"]] = None
    max_length: int = Field(default=120, ge=16, le=256)
    min_length: int = Field(default=30, ge=0, le=256)
    temperature: float = Field(default=1.0, ge=0.1, le=2.0)
    num_beams: int = Field(default=4, ge=1, le=8)
    top_k: int = Field(default=50, ge=0, le=200)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.2, ge=1.0, le=2.0)


class SummarizeResponse(BaseModel):
    summary: str
    confidence_score: float
    latency_ms: float


@app.on_event("startup")
def _startup() -> None:
    # Warm-up at startup (non-blocking so the service starts fast).
    if not settings.inference.warmup:
        return

    def _do() -> None:
        try:
            runtime.load()
            runtime.warmup()
            logger.info("Model warmup complete")
        except Exception as e:
            logger.exception("Model warmup failed: %s", e)

    t = threading.Thread(target=_do, daemon=True)
    t.start()


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    html_path = Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest) -> SummarizeResponse:
    try:
        strategy = req.strategy or settings.inference.default_decoding.strategy
        res = runtime.summarize(
            req.text,
            max_length=req.max_length,
            min_length=req.min_length,
            strategy=strategy,
            temperature=req.temperature,
            num_beams=req.num_beams,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            max_input_length=settings.tokenization.max_input_length,
        )
    except Exception as e:
        logger.exception("Summarization failed")
        raise HTTPException(status_code=500, detail=str(e))

    analytics.append_request(
        RequestLogRecord(
            ts=__import__("time").time(),
            latency_ms=res.latency_ms,
            input_chars=len(req.text),
            summary_chars=len(res.summary),
            confidence_score=res.confidence_score,
            decoding={
                "strategy": req.strategy,
                "max_length": req.max_length,
                "num_beams": req.num_beams,
                "top_k": req.top_k,
                "top_p": req.top_p,
            },
        )
    )

    logger.info(
        "summarize latency_ms=%.1f summary_chars=%d confidence=%.3f",
        res.latency_ms,
        len(res.summary),
        res.confidence_score,
    )

    return SummarizeResponse(summary=res.summary, confidence_score=res.confidence_score, latency_ms=res.latency_ms)
