# Ai-pdf-to-notes-summarizer

PDF to notes summarizer (frontend + backend).

## Article summarization (local model)

This repo includes an end-to-end article summarization workflow (dataset cleaning/splitting, fine-tuning, and a backend API).

### 1) Preprocess dataset

From the repo root:

- `cd backend`
- `python -m summarizer.preprocess --csv .venv/article_highlights.csv --out data/processed`

This writes `data/processed/train.jsonl` and `data/processed/val.jsonl`.

### 2) Train / fine-tune model

- `python -m summarizer.train --processed-dir data/processed --output-dir models/summarizer`

### 3) Run backend + summarize

- `uvicorn server:app --reload`

Notes:
- Default base model is `t5-small` (smaller download, faster to start).
- Set `SUMMARIZER_WARM_START=1` to warm-load the model at startup (non-blocking).

API:

- `POST /api/summarize` with JSON `{ "text": "...", "min_length": 30, "max_length": 120 }`

### 4) Evaluate (ROUGE)

- `python -m summarizer.evaluate --val data/processed/val.jsonl --limit 50`

## CNN/DailyMail production summarizer (new module)

This repo also includes a clean-architecture, production-oriented summarizer under `ai_summarizer/` with:
- CNN/DailyMail loading via Hugging Face `datasets`
- Modular preprocessing (cleaning/segmentation/tokenization/filtering/augmentation)
- Fine-tuning with `Seq2SeqTrainer` + ROUGE + BERTScore
- Standalone FastAPI `POST /summarize` returning `summary`, `confidence_score`, `latency_ms`

### 1) Install backend deps

From repo root:

- `cd backend`
- `python -m pip install -r requirements.txt`

### 2) Train (optional)

From repo root:

- `python -m ai_summarizer.training.train --limit-train 2000 --limit-eval 500`

Notes:
- On CPU, `t5-base` / `bart-large-cnn` can be very slow; use `--model-name t5-small` for a quick local run.
- Example quick smoke run: `python -m ai_summarizer.training.train --model-name t5-small --limit-train 200 --limit-eval 50 --output-dir ai_summarizer/models/checkpoints_smoke`

Artifacts are written to `ai_summarizer/models/checkpoints` by default (see `config.yaml`).

### 3) Evaluate (optional)

- `python -m ai_summarizer.evaluation.evaluate --limit 500`

This writes last eval metrics to `ai_summarizer/logs/last_eval.json`.

### 4) Run API

- `python -m uvicorn ai_summarizer.api.app:app --host 127.0.0.1 --port 8001`

Open `http://127.0.0.1:8001/` for the minimal UI, or call:

- `POST http://127.0.0.1:8001/summarize` with JSON `{ "text": "...", "max_length": 120, "num_beams": 4 }`

