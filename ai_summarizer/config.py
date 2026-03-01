from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field


class DatasetSettings(BaseModel):
    source: Literal["huggingface", "kaggle"] = "huggingface"
    hf_name: str = "cnn_dailymail"
    hf_config: str = "3.0.0"


class FilteringSettings(BaseModel):
    min_article_tokens: int = 50
    max_article_tokens: int = 1024
    min_summary_tokens: int = 10
    max_summary_tokens: int = 256


class PreprocessingSettings(BaseModel):
    lowercase: bool = False
    sentence_dropout_prob: float = Field(default=0.0, ge=0.0, le=0.9)


class TokenizationSettings(BaseModel):
    model_name: str = "t5-base"
    max_input_length: int = 512
    max_summary_length: int = 128


class TrainingSettings(BaseModel):
    output_dir: str = "ai_summarizer/models/checkpoints"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = False
    evaluation_strategy: Literal["epoch", "steps", "no"] = "epoch"
    save_strategy: Literal["epoch", "steps", "no"] = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "rougeL"
    greater_is_better: bool = True
    early_stopping_patience: int = 2


class DefaultDecodingSettings(BaseModel):
    strategy: Literal["greedy", "beam", "topk", "topp"] = "beam"
    num_beams: int = 4
    max_length: int = 120
    min_length: int = 30
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.2


class InferenceSettings(BaseModel):
    device: Literal["auto", "cpu", "cuda"] = "auto"
    warmup: bool = True
    default_decoding: DefaultDecodingSettings = Field(default_factory=DefaultDecodingSettings)


class LoggingSettings(BaseModel):
    log_dir: str = "ai_summarizer/logs"
    requests_jsonl: str = "requests.jsonl"
    usage_json: str = "usage.json"
    last_eval_json: str = "last_eval.json"


class EvaluationSettings(BaseModel):
    # BERTScore defaults to roberta-large which is a ~1.4GB download.
    # Use a smaller default for local/dev runs; override in config.yaml if desired.
    bertscore_model_type: str = "distilroberta-base"
    bertscore_lang: str = "en"


class ProjectSettings(BaseModel):
    seed: int = 42


class Settings(BaseModel):
    project: ProjectSettings = Field(default_factory=ProjectSettings)
    dataset: DatasetSettings = Field(default_factory=DatasetSettings)
    filtering: FilteringSettings = Field(default_factory=FilteringSettings)
    preprocessing: PreprocessingSettings = Field(default_factory=PreprocessingSettings)
    tokenization: TokenizationSettings = Field(default_factory=TokenizationSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


def load_settings(config_path: Optional[str | Path] = None) -> Settings:
    path = Path(config_path) if config_path else Path(__file__).resolve().parents[1] / "config.yaml"
    if not path.exists():
        return Settings()
    raw: Any
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Settings.model_validate(raw)
