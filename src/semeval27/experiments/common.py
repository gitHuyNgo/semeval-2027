from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Iterable

from semeval27 import GLOBAL_SEED
from semeval27.evaluation.provisional_bertscore import provisional_config
from semeval27.models.base import ModelRunner
from semeval27.prompting.renderer import RenderedRequest
from semeval27.utils.io import REPO_ROOT, append_jsonl, read_jsonl, sha256_file, sha256_value, write_json_atomic
from semeval27.utils.reproducibility import software_metadata, utc_now


def successful_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        row["sample_id"]
        for row in read_jsonl(path)
        if row.get("inference_status") == "success"
    }


def validate_resume_compatibility(
    path: Path,
    *,
    model_id: str,
    model_revision: str | None,
    prompt_hash: str,
    regime: str,
    demo_ids: list[str],
) -> None:
    """Refuse to mix predictions produced by different benchmark definitions."""
    if not path.is_file():
        return
    rows = read_jsonl(path)
    incompatible = [
        row.get("sample_id")
        for row in rows
        if row.get("model_id") != model_id
        or (model_revision is not None and row.get("model_revision") != model_revision)
        or row.get("semantic_prompt_hash") != prompt_hash
        or row.get("regime") != regime
        or row.get("demo_ids", []) != demo_ids
    ]
    if incompatible:
        raise ValueError(
            f"Existing prediction file is incompatible with this run ({len(incompatible)} row(s)): {path}. "
            "Preserve/archive it, then start this system with a clean prediction path."
        )


def run_requests(
    jobs: Iterable[tuple[RenderedRequest, str]],
    *,
    runner: ModelRunner,
    output: Path,
    model_alias: str,
    force: bool,
) -> tuple[int, int, str]:
    errors_path = output.with_suffix(".errors.jsonl")
    if force and output.exists():
        output.unlink()
    if force and errors_path.exists():
        errors_path.unlink()
    completed = successful_ids(output)
    run_id = uuid.uuid4().hex
    successes = errors = 0
    for request, reference in jobs:
        if request.sample_id in completed:
            continue
        result = runner.generate(request)
        row = {
            "sample_id": request.sample_id,
            "model_alias": model_alias,
            "model_id": runner.model_id,
            "model_revision": runner.resolved_revision,
            "prompt_id": request.prompt_id,
            "semantic_prompt_version": 1,
            "semantic_prompt_hash": request.prompt_hash,
            "regime": request.regime,
            "demo_ids": list(request.demo_ids),
            "raw_model_output": result.output,
            "normalized_output": None,
            "reference_answer": reference,
            "inference_status": result.status,
            "error": result.error,
            "latency_seconds": result.latency_seconds,
            "generation_configuration": result.effective_generation_config,
            "provider_response_metadata": result.provider_response_metadata,
            "timestamp": utc_now(),
            "run_id": run_id,
        }
        if result.status == "success":
            append_jsonl(output, row)
            completed.add(request.sample_id)
            successes += 1
        else:
            append_jsonl(errors_path, row)
            errors += 1
            print(f"ERROR {request.sample_id}: {result.error}")
    return successes, errors, run_id


def scorer_provenance() -> dict[str, Any]:
    provenance = REPO_ROOT / "vendor" / "mmcqa_official_eval" / "PROVENANCE.md"
    if provenance.is_file():
        return {"status": "official_vendor_present_unconfirmed_interface", "provenance_sha256": sha256_file(provenance)}
    config = provisional_config()
    return {
        "status": "organizer_scorer_absent_provisional_only",
        "provisional_configuration_sha256": sha256_value(config),
    }


def write_run_metadata(
    path: Path,
    *,
    system_id: str,
    runner: ModelRunner,
    dataset_revision: str | None,
    split: str,
    sample_count: int,
    prompt_id: str,
    prompt_hash: str,
    fewshot_manifest_hash: str | None,
    inference_parameters: dict[str, Any],
    run_id: str,
    started_at: str,
    ended_at: str,
) -> None:
    write_json_atomic(
        path,
        {
            "system_id": system_id,
            "model_id": runner.model_id,
            "resolved_model_revision": runner.resolved_revision,
            "dataset_source_revision": dataset_revision,
            "split": split,
            "sample_count": sample_count,
            "prompt_id": prompt_id,
            "prompt_hash": prompt_hash,
            "fewshot_manifest_hash": fewshot_manifest_hash,
            "inference_parameters": inference_parameters,
            "inference_config_hash": sha256_value(inference_parameters),
            "seed": GLOBAL_SEED,
            "run_id": run_id,
            "start_time": started_at,
            "end_time": ended_at,
            "software": software_metadata(REPO_ROOT),
            "scorer_provenance": scorer_provenance(),
        },
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

