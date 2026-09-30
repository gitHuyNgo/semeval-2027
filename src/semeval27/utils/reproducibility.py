from __future__ import annotations

import importlib.metadata
import platform
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from semeval27 import GLOBAL_SEED
from semeval27.utils.io import git_commit


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def seed_everything(seed: int = GLOBAL_SEED) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def software_metadata(repo_root: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in ["bert-score", "transformers", "torch", "peft", "openai", "google-genai"]
        },
        "repository_commit": git_commit(repo_root),
    }
    try:
        import torch

        info["torch"] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        info["torch"] = None
    return info
