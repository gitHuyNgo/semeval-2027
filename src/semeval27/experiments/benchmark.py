from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from semeval27.data.dataset import DatasetLayout, safe_model_sample
from semeval27.data.fewshot import DEFAULT_NAME_FEWSHOT, load_fewshot
from semeval27.experiments.common import (
    run_requests,
    successful_ids,
    validate_resume_compatibility,
    write_run_metadata,
)
from semeval27.models.registry import create_runner
from semeval27.prompting.renderer import load_frozen_prompt, render_request
from semeval27.utils.io import REPO_ROOT, load_paths, load_yaml, read_jsonl
from semeval27.utils.reproducibility import utc_now


def validate_prediction_coverage(
    rows: list[dict[str, Any]],
    expected_ids: set[str],
    expected_demo_ids: list[str] | None = None,
) -> None:
    ids = [row.get("sample_id") for row in rows]
    duplicates = sorted({sample_id for sample_id in ids if ids.count(sample_id) > 1})
    missing = sorted(expected_ids - set(ids))
    extras = sorted(set(ids) - expected_ids)
    errors = [row.get("sample_id") for row in rows if row.get("inference_status") != "success"]
    demo_mismatches = []
    if expected_demo_ids is not None:
        demo_mismatches = [row.get("sample_id") for row in rows if row.get("demo_ids", []) != expected_demo_ids]
    if duplicates or missing or extras or errors or demo_mismatches:
        raise ValueError(
            f"Invalid predictions: duplicates={len(duplicates)}, missing={len(missing)}, "
            f"extras={len(extras)}, errors={len(errors)}, demo_mismatches={len(demo_mismatches)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a configured frozen-prompt benchmark system")
    parser.add_argument("--system-id", required=True)
    parser.add_argument("--paths-config", default="configs/paths.example.yaml")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--confirm-full-api", action="store_true")
    parser.add_argument("--debug-quantized", action="store_true")
    args = parser.parse_args()
    dataset_root, artifact_root = load_paths(args.paths_config)
    layout = DatasetLayout(dataset_root)
    model_config = load_yaml(REPO_ROOT / "configs" / "models.yaml")
    if args.system_id not in model_config["systems"]:
        raise SystemExit(
            f"Unknown system ID {args.system_id!r}; choose one of {sorted(model_config['systems'])}"
        )
    system = model_config["systems"][args.system_id]
    spec = model_config["models"][system["model"]]
    prompt_id, template, frozen_hash = load_frozen_prompt(REPO_ROOT / "configs" / "prompts" / "frozen_prompt.yaml")
    demos: list[dict[str, Any]] = []
    fewshot_hash = None
    if system["regime"] == "fixed_9shot":
        fewshot = load_fewshot(artifact_root / "fewshot" / DEFAULT_NAME_FEWSHOT)
        demos = fewshot["demonstrations"]
        fewshot_hash = fewshot["manifest_sha256"]
    dev = layout.load_split("dev")
    if args.limit is not None:
        dev = dev[: args.limit]
    output = artifact_root / "predictions" / "benchmark" / f"{args.system_id}.jsonl"
    if not args.force:
        validate_resume_compatibility(
            output,
            model_id=spec["model_id"],
            model_revision=spec.get("revision"),
            prompt_hash=frozen_hash,
            regime=system["regime"],
            demo_ids=[demo["sample_id"] for demo in demos],
        )
    completed = successful_ids(output)
    completed_in_scope = sum(row["id"] in completed for row in dev)
    remaining = sum(row["id"] not in completed for row in dev)
    print(
        f"model={spec['model_id']} split_size={len(dev)} regime={system['regime']} "
        f"already_completed={completed_in_scope} requests_required={remaining}"
    )
    if spec["provider"] in {"openai", "gemini"} and args.limit is None and not args.confirm_full_api:
        raise SystemExit("Full paid API inference requires the explicit --confirm-full-api flag")
    runner = create_runner(
        system["model"],
        models_path=REPO_ROOT / "configs" / "models.yaml",
        inference_path=REPO_ROOT / "configs" / "inference.yaml",
        layout=layout,
        debug_quantized=args.debug_quantized,
    )
    jobs = [
        (
            render_request(
                target=safe_model_sample(row),
                prompt_id=prompt_id,
                template=template,
                regime=system["regime"],
                demonstrations=demos,
                target_split="dev",
            ),
            row["answer"],
        )
        for row in dev
    ]
    started = utc_now()
    successes, errors, run_id = run_requests(jobs, runner=runner, output=output, model_alias=system["model"], force=args.force)
    inference = load_yaml(REPO_ROOT / "configs" / "inference.yaml")
    write_run_metadata(
        artifact_root / "run_metadata" / "benchmark" / f"{args.system_id}_{run_id}.json",
        system_id=args.system_id,
        runner=runner,
        dataset_revision=layout.revision,
        split="dev",
        sample_count=len(dev),
        prompt_id=prompt_id,
        prompt_hash=frozen_hash,
        fewshot_manifest_hash=fewshot_hash,
        inference_parameters={
            "generation": inference["generation"],
            "provider": inference[spec["provider"] if spec["provider"] in {"openai", "gemini"} else "local"],
            "debug_quantized": args.debug_quantized,
        },
        run_id=run_id,
        started_at=started,
        ended_at=utc_now(),
    )
    print(f"{args.system_id}: wrote {successes} successful predictions; {errors} errors (not marked complete)")
    if args.limit is None:
        validate_prediction_coverage(
            read_jsonl(output),
            {row["id"] for row in layout.load_split("dev")},
            [demo["sample_id"] for demo in demos],
        )


if __name__ == "__main__":
    main()
