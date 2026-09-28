from __future__ import annotations

import argparse
from pathlib import Path

from semeval27.data.calibration import DEFAULT_NAME, load_calibration
from semeval27.data.dataset import DatasetLayout
from semeval27.experiments.common import run_requests, successful_ids, write_run_metadata
from semeval27.models.registry import create_runner
from semeval27.prompting.renderer import load_candidates, render_request
from semeval27.utils.io import REPO_ROOT, load_paths, load_yaml
from semeval27.utils.reproducibility import utc_now


ALLOWED_MODELS = ("qwen4b", "gemma4b")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TRAIN-only prompt calibration on Qwen4B or Gemma4B")
    parser.add_argument("--model", required=True, choices=ALLOWED_MODELS)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt-id", choices=["P1", "P2", "P3", "P4"])
    prompt_group.add_argument("--all-prompts", action="store_true")
    parser.add_argument("--paths-config", default="configs/paths.example.yaml")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--debug-quantized", action="store_true")
    args = parser.parse_args()
    dataset_root, artifact_root = load_paths(args.paths_config)
    layout = DatasetLayout(dataset_root)
    _, samples = load_calibration(artifact_root / "manifests" / DEFAULT_NAME)
    if args.limit is not None:
        samples = samples[: args.limit]
    candidates = load_candidates(REPO_ROOT / "configs" / "prompts" / "calibration_candidates.yaml")
    prompt_ids = list(candidates) if args.all_prompts else [args.prompt_id]
    inference = load_yaml(REPO_ROOT / "configs" / "inference.yaml")
    runner = create_runner(
        args.model,
        models_path=REPO_ROOT / "configs" / "models.yaml",
        inference_path=REPO_ROOT / "configs" / "inference.yaml",
        layout=layout,
        debug_quantized=args.debug_quantized,
    )
    for prompt_id in prompt_ids:
        assert prompt_id is not None
        output = artifact_root / "predictions" / "calibration" / f"{args.model}_{prompt_id}.jsonl"
        completed = successful_ids(output)
        completed_in_scope = sum(sample["sample_id"] in completed for sample in samples)
        remaining = sum(sample["sample_id"] not in completed for sample in samples)
        print(f"model={runner.model_id} split=train(calibration) prompt={prompt_id} size={len(samples)} completed={completed_in_scope} remaining={remaining}")
        jobs = [
            (
                render_request(
                    target={"id": row["sample_id"], "image": row["image"], "question": row["question"]},
                    prompt_id=prompt_id,
                    template=candidates[prompt_id],
                    regime="zero_shot",
                    target_split="train",
                ),
                row["answer"],
            )
            for row in samples
        ]
        started = utc_now()
        successes, errors, run_id = run_requests(jobs, runner=runner, output=output, model_alias=args.model, force=args.force)
        prompt = jobs[0][0] if jobs else None
        if prompt:
            write_run_metadata(
                artifact_root / "run_metadata" / "calibration" / f"{args.model}_{prompt_id}_{run_id}.json",
                system_id=f"CAL-{args.model}-{prompt_id}",
                runner=runner,
                dataset_revision=layout.revision,
                split="train_calibration",
                sample_count=len(samples),
                prompt_id=prompt_id,
                prompt_hash=prompt.prompt_hash,
                fewshot_manifest_hash=None,
                inference_parameters={
                    "generation": inference["generation"],
                    "local": inference["local"],
                    "debug_quantized": args.debug_quantized,
                },
                run_id=run_id,
                started_at=started,
                ended_at=utc_now(),
            )
        print(f"{prompt_id}: wrote {successes} successful predictions; {errors} errors (not marked complete)")


if __name__ == "__main__":
    main()

