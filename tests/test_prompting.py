from __future__ import annotations

from semeval27.prompting.renderer import prompt_hash, render_request, request_debug_json


TEMPLATE = "Answer the question about the image.\n\nQuestion: {question}\nAnswer:"


def demos():
    return [
        {"sample_id": f"demo-{index}", "image": f"images/demo-{index}.jpg", "question": f"Q{index}?", "answer": f"A{index}"}
        for index in range(9)
    ]


def test_metadata_fields_never_appear_in_rendered_inputs() -> None:
    request = render_request(
        target={"id": "target", "image": "images/target.jpg", "question": "What is shown?"},
        prompt_id="P1",
        template=TEMPLATE,
        regime="fixed_9shot",
        demonstrations=demos(),
        target_split="dev",
    )
    serialized = request_debug_json(request)
    for forbidden in ["country", "category", "subcategory", "offline-country"]:
        assert forbidden not in serialized


def test_all_models_receive_same_semantic_frozen_prompt() -> None:
    requests = [
        render_request(
            target={"id": "target", "image": "images/target.jpg", "question": "What is shown?"},
            prompt_id="P1",
            template=TEMPLATE,
            regime="zero_shot",
            target_split="dev",
        )
        for _model in ["openai", "gemini", "qwen", "gemma"]
    ]
    assert len({request.prompt_hash for request in requests}) == 1
    assert len({request.prompt_template for request in requests}) == 1


def test_all_fewshot_models_receive_same_demo_ids_and_order() -> None:
    requests = [
        render_request(
            target={"id": "target", "image": "images/target.jpg", "question": "What is shown?"},
            prompt_id="P1",
            template=TEMPLATE,
            regime="fixed_9shot",
            demonstrations=demos(),
            target_split="dev",
        )
        for _model in ["openai", "gemini", "qwen", "gemma"]
    ]
    assert len({request.demo_ids for request in requests}) == 1
    assert requests[0].demo_ids == tuple(f"demo-{index}" for index in range(9))


def test_prompt_changes_produce_different_hash() -> None:
    assert prompt_hash("P1", TEMPLATE) != prompt_hash("P1", TEMPLATE + " More.")

