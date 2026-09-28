from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from semeval27.utils.io import load_yaml, sha256_value


@dataclass(frozen=True)
class ContentBlock:
    type: Literal["text", "image"]
    text: str | None = None
    image: str | None = None
    split: Literal["train", "dev"] | None = None


@dataclass(frozen=True)
class ChatTurn:
    role: Literal["user", "assistant"]
    content: tuple[ContentBlock, ...]


@dataclass(frozen=True)
class RenderedRequest:
    sample_id: str
    turns: tuple[ChatTurn, ...]
    prompt_id: str
    prompt_template: str
    prompt_hash: str
    regime: Literal["zero_shot", "fixed_9shot"]
    demo_ids: tuple[str, ...]


def load_candidates(path: Path) -> dict[str, str]:
    config = load_yaml(path)
    prompts = config.get("prompts", {})
    if list(prompts) != ["P1", "P2", "P3", "P4"]:
        raise ValueError("Calibration prompt file must define exactly P1, P2, P3, P4 in order")
    return {key: str(value) for key, value in prompts.items()}


def load_frozen_prompt(path: Path) -> tuple[str, str, str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"No frozen prompt at {path}. Run python -m semeval27.prompting.freeze --prompt-id P#"
        )
    config = load_yaml(path)
    if config.get("status") != "frozen":
        raise ValueError(f"Prompt config is not frozen: {path}")
    prompt_id = str(config["prompt_id"])
    template = str(config["template"])
    prompt_hash = sha256_value({"prompt_id": prompt_id, "template": template, "version": config["version"]})
    if config.get("prompt_hash") != prompt_hash:
        raise ValueError("Frozen prompt hash does not match its content")
    return prompt_id, template, prompt_hash


def prompt_hash(prompt_id: str, template: str, version: int = 1) -> str:
    return sha256_value({"prompt_id": prompt_id, "template": template, "version": version})


def render_request(
    *,
    target: dict[str, str],
    prompt_id: str,
    template: str,
    regime: Literal["zero_shot", "fixed_9shot"],
    demonstrations: list[dict[str, Any]] | None = None,
    target_split: Literal["train", "dev"],
) -> RenderedRequest:
    turns: list[ChatTurn] = []
    demos = demonstrations or []
    if regime == "fixed_9shot":
        if len(demos) != 9:
            raise ValueError(f"fixed_9shot requires exactly 9 demonstrations, got {len(demos)}")
        for demo in demos:
            turns.append(
                ChatTurn(
                    role="user",
                    content=(
                        ContentBlock(type="image", image=str(demo["image"]), split="train"),
                        ContentBlock(type="text", text=f"Question: {demo['question']}\nAnswer:"),
                    ),
                )
            )
            turns.append(
                ChatTurn(role="assistant", content=(ContentBlock(type="text", text=str(demo["answer"])),))
            )
    elif demos:
        raise ValueError("zero_shot requests cannot contain demonstrations")
    turns.append(
        ChatTurn(
            role="user",
            content=(
                ContentBlock(type="image", image=target["image"], split=target_split),
                ContentBlock(type="text", text=template.format(question=target["question"])),
            ),
        )
    )
    request = RenderedRequest(
        sample_id=target["id"],
        turns=tuple(turns),
        prompt_id=prompt_id,
        prompt_template=template,
        prompt_hash=prompt_hash(prompt_id, template),
        regime=regime,
        demo_ids=tuple(str(item["sample_id"]) for item in demos),
    )
    from semeval27.prompting.validation import validate_rendered_request

    validate_rendered_request(request)
    return request


def request_debug_json(request: RenderedRequest) -> str:
    """A stable serialization used only for tests/logging; never sent to a model."""
    return json.dumps(
        {
            "sample_id": request.sample_id,
            "turns": [
                {
                    "role": turn.role,
                    "content": [
                        {key: value for key, value in block.__dict__.items() if value is not None}
                        for block in turn.content
                    ],
                }
                for turn in request.turns
            ],
            "prompt_id": request.prompt_id,
            "prompt_hash": request.prompt_hash,
            "regime": request.regime,
            "demo_ids": request.demo_ids,
        },
        ensure_ascii=False,
        sort_keys=True,
    )

