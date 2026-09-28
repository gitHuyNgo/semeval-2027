from __future__ import annotations

from pathlib import Path

import pytest

from semeval27.evaluation.provisional_bertscore import _prediction_and_reference, provisional_config
from semeval27.evaluation.sync_official_scorer import sync_files
from semeval27.experiments.benchmark import validate_prediction_coverage
from semeval27.utils.io import sha256_file


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
