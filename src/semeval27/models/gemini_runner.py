from __future__ import annotations

import os
import time
from typing import Any

from semeval27.data.dataset import DatasetLayout
from semeval27.models.base import GenerationResult, ModelRunner, bounded_retry
from semeval27.prompting.renderer import RenderedRequest


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _response_metadata(response: Any) -> dict[str, Any]:
    candidates = list(getattr(response, "candidates", None) or [])
    candidate = candidates[0] if candidates else None
    usage = getattr(response, "usage_metadata", None)
    prompt_feedback = getattr(response, "prompt_feedback", None)
    return {
        "candidate_count": len(candidates),
        "finish_reason": _enum_value(getattr(candidate, "finish_reason", None)),
        "finish_message": getattr(candidate, "finish_message", None),
        "prompt_block_reason": _enum_value(getattr(prompt_feedback, "block_reason", None)),
        "usage": {
            field: getattr(usage, field, None)
            for field in (
                "prompt_token_count",
                "candidates_token_count",
                "thoughts_token_count",
                "total_token_count",
                "cached_content_token_count",
            )
        },
    }


class GeminiRunner(ModelRunner):
    def __init__(self, model_id: str, revision: str | None, generation: dict[str, Any], provider_config: dict[str, Any], layout: DatasetLayout) -> None:
        super().__init__(model_id, revision, generation)
        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY is required")
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("Install the Gemini extra: python -m pip install -e '.[gemini]'") from exc
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.provider_config = provider_config
        self.layout = layout

    def _contents(self, request: RenderedRequest):
        from google.genai import types

        contents = []
        for turn in request.turns:
            parts = []
            for block in turn.content:
                if block.type == "text":
                    parts.append(types.Part.from_text(text=block.text))
                else:
                    assert block.image and block.split
                    parts.append(types.Part.from_bytes(data=self.layout.read_image_bytes(block.split, block.image), mime_type="image/jpeg"))
            contents.append(types.Content(role="model" if turn.role == "assistant" else "user", parts=parts))
        return contents

    def generate(self, request: RenderedRequest) -> GenerationResult:
        from google.genai import types

        started = time.perf_counter()
        effective = {
            "temperature": "provider_default; sampling parameters omitted for Gemini 3.8",
            "max_output_tokens": int(
                self.provider_config.get("max_output_tokens", self.generation["max_output_tokens"])
            ),
            "thinking_level": str(self.provider_config["thinking_level"]),
            "tools": None,
        }
        config = types.GenerateContentConfig(
            max_output_tokens=effective["max_output_tokens"],
            thinking_config=types.ThinkingConfig(
                thinking_level=effective["thinking_level"]
            ),
        )
        try:
            response = bounded_retry(
                lambda: self.client.models.generate_content(model=self.model_id, contents=self._contents(request), config=config),
                max_retries=int(self.provider_config["max_retries"]),
                initial_backoff_seconds=float(self.provider_config["initial_backoff_seconds"]),
                max_backoff_seconds=float(self.provider_config["max_backoff_seconds"]),
            )
            model_version = getattr(response, "model_version", None)
            self.resolved_revision = model_version or self.model_id
            output = response.text or ""
            metadata = _response_metadata(response)
            if not output:
                print(
                    f"WARNING {request.sample_id}: Gemini returned no visible text "
                    f"(finish_reason={metadata['finish_reason']}, usage={metadata['usage']})"
                )
            return GenerationResult(
                output,
                "success",
                None,
                time.perf_counter() - started,
                effective,
                metadata,
            )
        except Exception as exc:
            return GenerationResult("", "error", f"{type(exc).__name__}: {exc}", time.perf_counter() - started, effective)

