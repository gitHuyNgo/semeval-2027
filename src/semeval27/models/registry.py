from __future__ import annotations

from pathlib import Path
from typing import Any

from semeval27.data.dataset import DatasetLayout
from semeval27.models.base import ModelRunner
from semeval27.utils.io import load_yaml


def model_spec(alias: str, config_path: Path) -> dict[str, Any]:
    models = load_yaml(config_path)["models"]
    if alias not in models:
        raise KeyError(f"Unknown model alias {alias!r}; choose one of {sorted(models)}")
    return models[alias]


def create_runner(
    alias: str,
    *,
    models_path: Path,
    inference_path: Path,
    layout: DatasetLayout,
    debug_quantized: bool = False,
) -> ModelRunner:
    spec = model_spec(alias, models_path)
    inference = load_yaml(inference_path)
    common = (spec["model_id"], spec.get("revision"), inference["generation"])
    if spec["provider"] == "hf":
        from semeval27.models.hf_vlm_runner import HFVLMRunner

        return HFVLMRunner(*common, layout, debug_quantized=debug_quantized)
    if spec["provider"] == "openai":
        from semeval27.models.openai_runner import OpenAIRunner

        return OpenAIRunner(*common, inference["openai"], layout)
    if spec["provider"] == "gemini":
        from semeval27.models.gemini_runner import GeminiRunner

        return GeminiRunner(*common, inference["gemini"], layout)
    raise ValueError(f"Unsupported provider: {spec['provider']}")
