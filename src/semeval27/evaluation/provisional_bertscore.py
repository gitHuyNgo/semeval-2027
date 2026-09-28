from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from semeval27.utils.io import read_jsonl, sha256_value, write_json_atomic
from semeval27.utils.reproducibility import package_version, utc_now


WARNING = "THIS IS NOT YET THE OFFICIAL MMCultureQA SCORER."
MODEL_TYPE = "roberta-large"
LANGUAGE = "en"


def _prediction_and_reference(row: dict[str, Any]) -> tuple[str, str]:
    prediction = row.get("raw_model_output", row.get("prediction"))
    reference = row.get("reference_answer", row.get("reference"))
    if not isinstance(prediction, str) or not isinstance(reference, str):
        raise ValueError(f"Row {row.get('sample_id')} lacks string prediction/reference fields")
    # Deliberately return exact stored strings: no case, punctuation, article,
    # extraction, sentence, explanation, or semantic normalization is allowed.
    return prediction, reference


def provisional_config() -> dict[str, Any]:
    return {
        "metric": "BERTScore-F1",
        "status": "PROVISIONAL",
        "language": LANGUAGE,
        "model_type": MODEL_TYPE,
        "num_layers": "canonical_default_for_roberta-large",
        "idf": False,
        "rescale_with_baseline": False,
        "use_fast_tokenizer": False,
        "aggregation": "arithmetic_macro_mean",
        "answer_normalization": "none",
        "bert_score_version_pin": "0.3.13",
    }


def score_rows(rows: list[dict[str, Any]], *, device: str | None = None, batch_size: int = 16) -> dict[str, Any]:
    try:
        from bert_score import BERTScorer
    except ImportError as exc:
        raise RuntimeError("Install project dependencies; bert-score==0.3.13 is required") from exc
    installed = package_version("bert-score")
    if installed != "0.3.13":
        raise RuntimeError(f"Expected bert-score==0.3.13, found {installed!r}")
    successful = [row for row in rows if row.get("inference_status", row.get("status", "success")) == "success"]
    errors = [row for row in rows if row not in successful]
    if not successful:
        raise ValueError("No successful predictions to score")
    pairs = [_prediction_and_reference(row) for row in successful]
    predictions = [pair[0] for pair in pairs]
    references = [pair[1] for pair in pairs]
    scorer = BERTScorer(
        model_type=MODEL_TYPE,
        lang=LANGUAGE,
        num_layers=None,
        idf=False,
        rescale_with_baseline=False,
        use_fast_tokenizer=False,
        device=device,
    )
    precision, recall, f1 = scorer.score(predictions, references, batch_size=batch_size)
    p_values = [float(value) for value in precision.cpu().tolist()]
    r_values = [float(value) for value in recall.cpu().tolist()]
    f_values = [float(value) for value in f1.cpu().tolist()]
    model = getattr(scorer, "_model", None)
    model_config = getattr(model, "config", None)
    resolved_revision = getattr(model_config, "_commit_hash", None)
    layers = getattr(scorer, "_num_layers", None)
    config = provisional_config()
    report = {
        "warning": WARNING,
        "organizer_scorer_status": "The current local dataset release does not contain the organizer evaluation script.",
        **config,
        "configuration_sha256": sha256_value(config),
        "bertscore_hash_code": getattr(scorer, "hash", None),
        "versions": {
            "bert-score": installed,
            "transformers": package_version("transformers"),
            "torch": package_version("torch"),
        },
        "resolved_model_revision": resolved_revision,
        "resolved_num_layers": layers,
        "number_of_examples": len(successful),
        "number_of_inference_errors": len(errors),
        "macro_precision": sum(p_values) / len(p_values),
        "macro_recall": sum(r_values) / len(r_values),
        "macro_f1": sum(f_values) / len(f_values),
        "per_example": [
            {
                "sample_id": row["sample_id"],
                "precision": p_value,
                "recall": r_value,
                "f1": f_value,
            }
            for row, p_value, r_value, f_value in zip(successful, p_values, r_values, f_values, strict=True)
        ],
        "timestamp": utc_now(),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{WARNING} Provisional English BERTScore-F1")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    print(WARNING)
    rows = read_jsonl(Path(args.predictions))
    report = score_rows(rows, device=args.device, batch_size=args.batch_size)
    write_json_atomic(args.output, report)
    print(f"PROVISIONAL macro F1={report['macro_f1']:.6f} over {report['number_of_examples']} examples")
    print(WARNING)


if __name__ == "__main__":
    main()

