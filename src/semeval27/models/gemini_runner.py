from __future__ import annotations

import os
import time
from typing import Any

from semeval27.data.dataset import DatasetLayout
from semeval27.models.base import GenerationResult, ModelRunner, bounded_retry
from semeval27.prompting.renderer import RenderedRequest


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
            "temperature": 0,
            "max_output_tokens": int(self.generation["max_output_tokens"]),
            "thinking_budget": int(self.provider_config["thinking_budget"]),
            "tools": None,
        }
        config = types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=effective["max_output_tokens"],
            thinking_config=types.ThinkingConfig(thinking_budget=effective["thinking_budget"]),
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
            return GenerationResult(response.text or "", "success", None, time.perf_counter() - started, effective)
        except Exception as exc:
            return GenerationResult("", "error", f"{type(exc).__name__}: {exc}", time.perf_counter() - started, effective)

