from __future__ import annotations

from typing import Any


def load_cnn_dailymail(hf_name: str = "cnn_dailymail", hf_config: str = "3.0.0"):
    """Load CNN/DailyMail from Hugging Face Datasets.

    Returns a `datasets.DatasetDict` with train/validation/test splits.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency `datasets`. Install it in your Python environment: pip install datasets"
        ) from e

    ds = load_dataset(hf_name, hf_config)
    if not hasattr(ds, "keys"):
        raise RuntimeError("Expected a DatasetDict with splits.")

    # Standardize column names: article -> input, highlights -> summary
    def _ensure_columns(split: Any) -> Any:
        cols = set(split.column_names)
        if "article" not in cols:
            raise RuntimeError(f"Dataset split missing 'article' column. Columns: {split.column_names}")
        if "highlights" in cols and "summary" not in cols:
            split = split.rename_column("highlights", "summary")
        if "summary" not in set(split.column_names):
            raise RuntimeError(f"Dataset split missing 'highlights'/'summary' column. Columns: {split.column_names}")
        return split

    for k in list(ds.keys()):
        ds[k] = _ensure_columns(ds[k])

    return ds
