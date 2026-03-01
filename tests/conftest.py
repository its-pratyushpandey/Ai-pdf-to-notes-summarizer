from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure():
    # Ensure `backend/` is importable as a top-level path so tests can
    # import `server` and `summarizer.*` regardless of the working directory.
    repo_root = Path(__file__).resolve().parents[1]
    backend_dir = repo_root / "backend"
    if backend_dir.exists():
        sys.path.insert(0, str(backend_dir))
