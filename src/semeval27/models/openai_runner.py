from __future__ import annotations

import base64
import os
import time
from typing import Any

from semeval27.data.dataset import DatasetLayout
from semeval27.models.base import GenerationResult, ModelRunner, bounded_retry
from semeval27.prompting.renderer import RenderedRequest


class OpenAIRunner(ModelRunner):
    def __init__(self, model_id: str, revision: str | None, generation: dict[str, Any], provider_config: dict[str, Any], layout: DatasetLayout) -> None:
        super().__init__(model_id, revision, generation)
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the OpenAI extra: python -m pip install -e '.[openai]'") from exc
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.provider_config = provider_config
        self.layout = layout

    def _input(self, request: RenderedRequest) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for turn in request.turns:
            content: list[dict[str, Any]] = []
            for block in turn.content:
                if block.type == "text":
                    content.append({"type": "input_text" if turn.role == "user" else "output_text", "text": block.text})
                else:
                    assert block.image and block.split
                    encoded = base64.b64encode(self.layout.read_image_bytes(block.split, block.image)).decode("ascii")
                    content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{encoded}"})
            messages.append({"role": turn.role, "content": content})
        return messages

    def generate(self, request: RenderedRequest) -> GenerationResult:
        started = time.perf_counter()
        effective = {
            "temperature": 0,
            "max_output_tokens": int(self.generation["max_output_tokens"]),
            "reasoning_effort": self.provider_config["reasoning_effort"],
            "tools": [],
        }
        try:
            def create_response():
                arguments = {
                    "model": self.model_id,
                    "input": self._input(request),
                    "max_output_tokens": effective["max_output_tokens"],
                    "temperature": 0,
                    "reasoning": {"effort": effective["reasoning_effort"]},
                    "tools": [],
                }
                try:
                    return self.client.responses.create(**arguments)
                except Exception as exc:
                    message = str(exc).lower()
                    if "temperature" not in message or not any(term in message for term in ["unsupported", "not support", "invalid"]):
                        raise
                    arguments.pop("temperature")
                    effective["temperature"] = "provider_default; explicit 0 unsupported"
                    return self.client.responses.create(**arguments)

            response = bounded_retry(
                create_response,
                max_retries=int(self.provider_config["max_retries"]),
                initial_backoff_seconds=float(self.provider_config["initial_backoff_seconds"]),
                max_backoff_seconds=float(self.provider_config["max_backoff_seconds"]),
            )
            self.resolved_revision = getattr(response, "model", None) or self.model_id
            return GenerationResult(response.output_text, "success", None, time.perf_counter() - started, effective)
        except Exception as exc:
            return GenerationResult("", "error", f"{type(exc).__name__}: {exc}", time.perf_counter() - started, effective)

