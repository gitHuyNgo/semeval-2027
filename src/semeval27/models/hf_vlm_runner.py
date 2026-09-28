from __future__ import annotations

import os
import time
from typing import Any

from semeval27.data.dataset import DatasetLayout
from semeval27.models.base import GenerationResult, ModelRunner
from semeval27.prompting.renderer import RenderedRequest


class HFVLMRunner(ModelRunner):
    def __init__(
        self,
        model_id: str,
        revision: str | None,
        generation: dict[str, Any],
        layout: DatasetLayout,
        *,
        debug_quantized: bool = False,
    ) -> None:
        super().__init__(model_id, revision, generation)
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("Install the local extra: python -m pip install -e '.[local]'") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("Local scientific benchmark runs require a CUDA GPU")
        if not debug_quantized and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Native BF16 is required; use --debug-quantized only for explicitly labeled debugging")
        self.layout = layout
        hf_token = os.getenv("HF_TOKEN") or None
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision, token=hf_token)
        kwargs: dict[str, Any] = {"revision": revision, "device_map": "auto", "token": hf_token}
        if debug_quantized:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            self.precision_status = "DEBUG_4BIT_NOT_SCIENTIFIC"
        else:
            kwargs["torch_dtype"] = torch.bfloat16
            self.precision_status = "NATIVE_BF16"
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        commit = getattr(getattr(self.model, "config", None), "_commit_hash", None)
        self.resolved_revision = commit or revision

    def _messages(self, request: RenderedRequest) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for turn in request.turns:
            content: list[dict[str, Any]] = []
            for block in turn.content:
                if block.type == "text":
                    content.append({"type": "text", "text": block.text})
                else:
                    assert block.image and block.split
                    content.append({"type": "image", "image": self.layout.open_pil_image(block.split, block.image)})
            messages.append({"role": turn.role, "content": content})
        return messages

    def generate(self, request: RenderedRequest) -> GenerationResult:
        import torch

        started = time.perf_counter()
        effective = {
            "temperature": 0,
            "do_sample": False,
            "max_new_tokens": int(self.generation["max_output_tokens"]),
            "precision_status": self.precision_status,
            "native_chat_template": True,
        }
        try:
            messages = self._messages(request)
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            device = next(self.model.parameters()).device
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
            input_length = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=effective["max_new_tokens"],
                    do_sample=False,
                )
            output_ids = generated[:, input_length:]
            output = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            return GenerationResult(output, "success", None, time.perf_counter() - started, effective)
        except Exception as exc:
            return GenerationResult("", "error", f"{type(exc).__name__}: {exc}", time.perf_counter() - started, effective)

