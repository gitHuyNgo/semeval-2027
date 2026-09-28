from __future__ import annotations

from semeval27.prompting.renderer import RenderedRequest


def validate_rendered_request(request: RenderedRequest) -> None:
    images = [block for turn in request.turns for block in turn.content if block.type == "image"]
    if request.regime == "zero_shot":
        if len(images) != 1 or request.demo_ids:
            raise ValueError("Zero-shot request must contain exactly one target image and no demos")
    elif request.regime == "fixed_9shot":
        if len(images) != 10 or len(request.demo_ids) != 9 or len(set(request.demo_ids)) != 9:
            raise ValueError("9-shot request must contain 9 unique demos and one target image")
        if sum(image.split == "train" for image in images) != 9:
            raise ValueError("All nine demonstration images must come from train")
    else:
        raise ValueError(f"Unknown regime: {request.regime}")
    for turn in request.turns:
        for block in turn.content:
            # ContentBlock intentionally cannot represent dataset metadata fields.
            if block.type == "image" and (not block.image or block.text is not None):
                raise ValueError("Malformed image block")
            if block.type == "text" and (block.text is None or block.image is not None):
                raise ValueError("Malformed text block")

