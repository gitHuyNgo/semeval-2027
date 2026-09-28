from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from semeval27.data.dataset import DatasetLayout
from semeval27.data.fewshot import load_fewshot
from semeval27.evaluation.provisional_bertscore import WARNING
from semeval27.experiments.benchmark import validate_prediction_coverage
from semeval27.utils.io import REPO_ROOT, load_paths, load_yaml, read_jsonl


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def calibration_report(artifact_root: Path) -> list[dict[str, Any]]:
    rows = []
    for prompt_id in ["P1", "P2", "P3", "P4"]:
        scores = {}
        lengths = []
        n_examples = []
        n_errors = []
        for alias in ["qwen4b", "gemma4b"]:
            score_path = artifact_root / "scores" / "calibration" / f"{alias}_{prompt_id}.json"
            prediction_path = artifact_root / "predictions" / "calibration" / f"{alias}_{prompt_id}.jsonl"
            if not score_path.is_file() or not prediction_path.is_file():
                raise FileNotFoundError(f"Missing calibration score/prediction for {alias} {prompt_id}")
            report = _read_json(score_path)
            if report.get("status") != "PROVISIONAL":
                raise ValueError(f"Expected a PROVISIONAL report: {score_path}")
            predictions = read_jsonl(prediction_path)
            scores[alias] = float(report["macro_f1"])
            n_examples.append(int(report["number_of_examples"]))
            n_errors.append(90 - len({row["sample_id"] for row in predictions}))
            lengths.extend(len(row["raw_model_output"]) for row in predictions)
        rows.append(
            {
                "status": "PROVISIONAL",
                "prompt_id": prompt_id,
                "qwen4b_provisional_bertscore_f1": scores["qwen4b"],
                "gemma4b_provisional_bertscore_f1": scores["gemma4b"],
                "cross_model_mean": (scores["qwen4b"] + scores["gemma4b"]) / 2,
                "num_examples": min(n_examples),
                "num_errors": sum(n_errors),
                "mean_output_length": sum(lengths) / len(lengths) if lengths else 0,
            }
        )
    fields = list(rows[0])
    output = artifact_root / "reports" / "prompt_calibration.csv"
    _write_csv(output, rows, fields)
    ranked = sorted(rows, key=lambda row: (-row["cross_model_mean"], row["prompt_id"]))
    lines = [
        "# Prompt calibration — PROVISIONAL",
        "",
        f"> **{WARNING}**",
        "> The current local dataset release does not contain the organizer evaluation script.",
        "",
        "| Prompt | Qwen4B provisional F1 | Gemma4B provisional F1 | Cross-model mean | Examples/model | Errors | Mean output length |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['prompt_id']} | {row['qwen4b_provisional_bertscore_f1']:.6f} | {row['gemma4b_provisional_bertscore_f1']:.6f} | {row['cross_model_mean']:.6f} | {row['num_examples']} | {row['num_errors']} | {row['mean_output_length']:.1f} |"
        for row in rows
    )
    lines.extend(
        [
            "",
            f"Highest provisional cross-model mean: **{ranked[0]['prompt_id']}**.",
            "",
            "This is a recommendation only. Review the table, then explicitly run `python -m semeval27.prompting.freeze --prompt-id P#`. No prompt was frozen or changed by this report.",
            "If a later official scorer changes the ordering, compare reports and make a new explicit decision; never silently refreeze.",
        ]
    )
    (artifact_root / "reports" / "prompt_calibration.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return rows


def benchmark_report(artifact_root: Path, dataset_root: Path) -> list[dict[str, Any]]:
    layout = DatasetLayout(dataset_root)
    expected = {row["id"] for row in layout.load_split("dev")}
    config = load_yaml(REPO_ROOT / "configs" / "models.yaml")
    fewshot = load_fewshot(artifact_root / "fewshot" / "fixed_9shot_seed42.json")
    fixed_demo_ids = fewshot["selected_ids"]
    rows = []
    for system_id in [f"B{i:02d}" for i in range(1, 11)]:
        prediction_path = artifact_root / "predictions" / "benchmark" / f"{system_id}.jsonl"
        score_path = artifact_root / "scores" / "benchmark" / f"{system_id}.json"
        predictions = read_jsonl(prediction_path)
        expected_demos = fixed_demo_ids if config["systems"][system_id]["regime"] == "fixed_9shot" else []
        validate_prediction_coverage(predictions, expected, expected_demos)
        score = _read_json(score_path)
        if score.get("status") != "PROVISIONAL":
            raise ValueError(f"Expected a PROVISIONAL report: {score_path}")
        system = config["systems"][system_id]
        model = config["models"][system["model"]]
        first = predictions[0]
        metadata_files = sorted((artifact_root / "run_metadata" / "benchmark").glob(f"{system_id}_*.json"))
        metadata = _read_json(metadata_files[-1]) if metadata_files else {}
        rows.append(
            {
                "status": "PROVISIONAL",
                "system_id": system_id,
                "model": model["model_id"],
                "regime": system["regime"],
                "provisional_bertscore_f1": score["macro_f1"],
                "n_samples": len(predictions),
                "n_errors": 0,
                "prompt_hash": first["semantic_prompt_hash"],
                "fewshot_manifest_hash": "" if system["regime"] == "zero_shot" else fewshot["manifest_sha256"],
                "model_revision": first.get("model_revision"),
                "inference_config_hash": metadata.get("inference_config_hash", "missing-run-metadata"),
            }
        )
    fields = list(rows[0])
    _write_csv(artifact_root / "reports" / "benchmark_provisional.csv", rows, fields)
    lines = [
        "# Full dev benchmark — PROVISIONAL",
        "",
        f"> **{WARNING}**",
        "> The current local dataset release does not contain the organizer evaluation script.",
        "",
        "| System | Model | Regime | Provisional BERTScore F1 | N | Errors |",
        "|---|---|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['system_id']} | {row['model']} | {row['regime']} | {float(row['provisional_bertscore_f1']):.6f} | {row['n_samples']} | {row['n_errors']} |"
        for row in rows
    )
    (artifact_root / "reports" / "benchmark_provisional.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["calibration", "benchmark"])
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--paths-config", default="configs/paths.example.yaml")
    args = parser.parse_args()
    artifact_root = (REPO_ROOT / args.artifact_root).resolve() if not Path(args.artifact_root).is_absolute() else Path(args.artifact_root)
    print(WARNING)
    if args.kind == "calibration":
        calibration_report(artifact_root)
    else:
        dataset_root, _ = load_paths(args.paths_config)
        benchmark_report(artifact_root, dataset_root)
    print(WARNING)


if __name__ == "__main__":
    main()

