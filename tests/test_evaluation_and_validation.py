from __future__ import annotations

from pathlib import Path

import pytest

from semeval27.evaluation.provisional_bertscore import _prediction_and_reference, provisional_config
from semeval27.evaluation.sync_official_scorer import sync_files
from semeval27.experiments.benchmark import validate_prediction_coverage
from semeval27.experiments.common import validate_resume_compatibility
from semeval27.utils.io import append_jsonl
from semeval27.models.gemini_runner import _response_metadata
from semeval27.utils.io import REPO_ROOT, load_yaml, sha256_file


def test_official_scorer_copy_hashes_match_originals(dataset_root: Path, tmp_path: Path) -> None:
    scorer = dataset_root / "evaluation" / "score.py"
    scorer.parent.mkdir()
    scorer.write_bytes(b"# organizer bytes\r\nprint('score')\r\n")
    destination = tmp_path / "vendor"
    records = sync_files(dataset_root, destination, [scorer])
    copied = destination / "evaluation" / "score.py"
    assert scorer.read_bytes() == copied.read_bytes()
    assert records[0]["source_sha256"] == records[0]["destination_sha256"] == sha256_file(scorer)


def test_final_validation_catches_missing_and_duplicate_predictions() -> None:
    expected = {"a", "b"}
    with pytest.raises(ValueError, match="missing=1"):
        validate_prediction_coverage([{"sample_id": "a", "inference_status": "success"}], expected)
    with pytest.raises(ValueError, match="duplicates=1"):
        validate_prediction_coverage(
            [
                {"sample_id": "a", "inference_status": "success"},
                {"sample_id": "a", "inference_status": "success"},
                {"sample_id": "b", "inference_status": "success"},
            ],
            expected,
        )


def test_provisional_configuration_is_explicit() -> None:
    config = provisional_config()
    assert config == {
        "metric": "BERTScore-F1",
        "status": "PROVISIONAL",
        "language": "en",
        "model_type": "roberta-large",
        "num_layers": "canonical_default_for_roberta-large",
        "idf": False,
        "rescale_with_baseline": False,
        "use_fast_tokenizer": False,
        "aggregation": "arithmetic_macro_mean",
        "answer_normalization": "none",
        "bert_score_version_pin": "0.3.13",
    }


def test_evaluator_preserves_raw_answer_text() -> None:
    row = {"sample_id": "x", "raw_model_output": "  The Answer!\n", "reference_answer": "The answer."}
    assert _prediction_and_reference(row) == ("  The Answer!\n", "The answer.")


def test_gemini_response_metadata_preserves_finish_reason_and_token_usage() -> None:
    class Value:
        value = "MAX_TOKENS"

    class Candidate:
        finish_reason = Value()
        finish_message = "token limit"

    class Usage:
        prompt_token_count = 10
        candidates_token_count = 0
        thoughts_token_count = 1024
        total_token_count = 1034
        cached_content_token_count = None

    class Feedback:
        block_reason = None

    class Response:
        candidates = [Candidate()]
        usage_metadata = Usage()
        prompt_feedback = Feedback()

    metadata = _response_metadata(Response())
    assert metadata["finish_reason"] == "MAX_TOKENS"
    assert metadata["finish_message"] == "token limit"
    assert metadata["usage"]["thoughts_token_count"] == 1024


def test_finetuned_zero_shot_system_pins_adapter_base_and_processor() -> None:
    config = load_yaml(REPO_ROOT / "configs" / "models.yaml")
    system = config["systems"]["FT01"]
    model = config["models"][system["model"]]
    assert system["regime"] == "zero_shot"
    assert model["provider"] == "hf_peft"
    assert model["model_id"] == "anhbilong/qwen3-vl-4b-mmcultureqa-split512"
    assert model["revision"] == "e62fdf28389cdc11f45f295f63e3e4c354e8eabe"
    assert model["base_model_id"] == "Qwen/Qwen3-VL-4B-Instruct"
    assert model["base_revision"] == "ebb281ec70b05090aa6165b016eac8ec08e71b17"
    assert model["processor_id"] == model["model_id"]
    assert model["processor_revision"] == model["revision"]
    assert model["dtype"] == "float16"


def test_resume_rejects_predictions_from_a_different_adapter(tmp_path: Path) -> None:
    path = tmp_path / "FT01.jsonl"
    append_jsonl(
        path,
        {
            "sample_id": "x",
            "model_id": "anhbilong/wrong-adapter",
            "model_revision": "old-revision",
            "semantic_prompt_hash": "prompt-hash",
            "regime": "zero_shot",
            "demo_ids": [],
            "inference_status": "success",
        },
    )
    with pytest.raises(ValueError, match="incompatible"):
        validate_resume_compatibility(
            path,
            model_id="anhbilong/qwen3-vl-4b-mmcultureqa-split512",
            model_revision="e62fdf28389cdc11f45f295f63e3e4c354e8eabe",
            prompt_hash="prompt-hash",
            regime="zero_shot",
            demo_ids=[],
        )
