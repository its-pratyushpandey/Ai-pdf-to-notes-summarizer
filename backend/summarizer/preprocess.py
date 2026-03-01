from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .config import DEFAULT_SUMMARIZER_CONFIG
from .data import load_article_highlights_csv


def _train_val_split_indices(n: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    val_n = max(1, int(n * val_ratio))
    val_idx = idx[:val_n]
    train_idx = idx[val_n:]
    return train_idx, val_idx


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    cfg = DEFAULT_SUMMARIZER_CONFIG

    parser = argparse.ArgumentParser(description="Preprocess article->summary dataset and create train/val jsonl splits")
    parser.add_argument("--csv", type=str, default=None, help="Path to article_highlights.csv")
    parser.add_argument("--out", type=str, default=str(cfg.processed_dir), help="Output directory for processed splits")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Optional max rows for quick experiments")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else (cfg.dataset_csv_default if cfg.dataset_csv_default.exists() else cfg.dataset_csv_fallback)
    df = load_article_highlights_csv(csv_path)
    if args.limit is not None:
        df = df.head(args.limit)

    train_idx, val_idx = _train_val_split_indices(len(df), val_ratio=args.val_ratio, seed=args.seed)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    out_dir = Path(args.out)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    train_rows = train_df[[c for c in ("url", "article", "summary") if c in train_df.columns]].to_dict(orient="records")
    val_rows = val_df[[c for c in ("url", "article", "summary") if c in val_df.columns]].to_dict(orient="records")

    _write_jsonl(train_rows, train_path)
    _write_jsonl(val_rows, val_path)

    print(f"Wrote {len(train_rows)} train rows -> {train_path}")
    print(f"Wrote {len(val_rows)} val rows -> {val_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
