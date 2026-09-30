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
        dtype_name: str = "bfloat16",
        base_model_id: str | None = None,
        base_revision: str | None = None,
        processor_id: str | None = None,
        processor_revision: str | None = None,
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
        self.layout = layout
        hf_token = os.getenv("HF_TOKEN") or None
        processor_source = processor_id or model_id
        processor_source_revision = processor_revision or revision
        self.processor = AutoProcessor.from_pretrained(
            processor_source,
            revision=processor_source_revision,
            token=hf_token,
        )
        model_source = base_model_id or model_id
        model_source_revision = base_revision if base_model_id else revision
        kwargs: dict[str, Any] = {
            "revision": model_source_revision,
            "device_map": "auto",
            "token": hf_token,
        }
        if debug_quantized:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            self.precision_status = "DEBUG_4BIT_NOT_SCIENTIFIC"
        else:
            if dtype_name not in {"bfloat16", "float16"}:
                raise ValueError(f"Unsupported scientific inference dtype: {dtype_name}")
            if dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported():
                raise RuntimeError(
                    "Native BF16 is required; use --debug-quantized only for explicitly labeled debugging"
                )
            kwargs["torch_dtype"] = getattr(torch, dtype_name)
            self.precision_status = f"NATIVE_{dtype_name.upper()}"
        self.model = AutoModelForImageTextToText.from_pretrained(model_source, **kwargs)
        base_commit = getattr(getattr(self.model, "config", None), "_commit_hash", None)
        self.base_model_id = model_source
        self.base_model_revision = base_commit or model_source_revision
        self.adapter_id: str | None = None
        self.adapter_revision: str | None = None
        if base_model_id:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("Install PEFT support: python -m pip install -e '.[local]'") from exc
            self.model = PeftModel.from_pretrained(
                self.model,
                model_id,
                revision=revision,
                token=hf_token,
                is_trainable=False,
            )
            self.adapter_id = model_id
            self.adapter_revision = revision
            self.resolved_revision = revision
        else:
            self.resolved_revision = self.base_model_revision
        self.model.eval()

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
            "processor_id": getattr(self.processor, "name_or_path", processor_source),
            "base_model_id": self.base_model_id,
            "base_model_revision": self.base_model_revision,
            "adapter_id": self.adapter_id,
            "adapter_revision": self.adapter_revision,
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

