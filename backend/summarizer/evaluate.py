from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

from .config import DEFAULT_SUMMARIZER_CONFIG
from .metrics import compute_rouge
from .service import SummarizerService


logger = logging.getLogger("summarizer.evaluate")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    cfg = DEFAULT_SUMMARIZER_CONFIG

    parser = argparse.ArgumentParser(description="Evaluate the summarizer on validation samples (ROUGE)")
    parser.add_argument("--val", type=str, default=str(cfg.processed_dir / "val.jsonl"))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--min-length", type=int, default=30)
    parser.add_argument("--max-length", type=int, default=120)
    parser.add_argument("--max-input-tokens", type=int, default=768)
    parser.add_argument("--out", type=str, default=None, help="Optional JSON file to write metrics")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    val_path = Path(args.val)
    if not val_path.exists():
        raise FileNotFoundError(
            f"Validation file not found: {val_path}. Run preprocess first to create train/val jsonl splits."
        )

    rows = _read_jsonl(val_path)
    rows = rows[: args.limit] if args.limit else rows
    if not rows:
        raise RuntimeError("No validation rows found")

    service = SummarizerService(cfg)
    service.load()
    logger.info("Evaluating model=%s on %s samples", service.model_id, len(rows))

    preds: list[str] = []
    refs: list[str] = []
    elapsed: list[float] = []

    for r in rows:
        article = r.get("article", "")
        reference = r.get("summary", "")
        if not article or not reference:
            continue
        result = service.summarize(
            article,
            min_length=args.min_length,
            max_length=args.max_length,
            max_input_tokens=args.max_input_tokens,
        )
        preds.append(result.summary)
        refs.append(reference)
        elapsed.append(result.elapsed_ms)

    metrics = compute_rouge(preds, refs)
    metrics["n"] = len(preds)
    metrics["avg_elapsed_ms"] = float(sum(elapsed) / max(1, len(elapsed)))
    metrics["model_id"] = service.model_id

    print(json.dumps(metrics, indent=2))

    out: Optional[str] = args.out
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        logger.info("Wrote metrics -> %s", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
