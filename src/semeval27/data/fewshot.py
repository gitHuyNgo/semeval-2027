from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from semeval27 import GLOBAL_SEED
from semeval27.data.calibration import DEFAULT_NAME, load_calibration
from semeval27.data.dataset import DatasetLayout, assert_source_unchanged
from semeval27.utils.io import load_paths, sha256_value, write_json_atomic
from semeval27.utils.reproducibility import utc_now


DEFAULT_NAME_FEWSHOT = "fixed_9shot_seed42.json"


def load_fewshot(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    payload = {key: value for key, value in document.items() if key not in {"manifest_sha256", "created_at"}}
    if sha256_value(payload) != document.get("manifest_sha256"):
        raise ValueError("Few-shot manifest hash does not match its content")
    if [row["sample_id"] for row in document["demonstrations"]] != document["selected_ids"]:
        raise ValueError("Few-shot demonstration order/IDs do not match manifest metadata")
    return document


def select_fewshot(
    train: list[dict[str, Any]],
    calibration_ids: set[str],
    dev_ids: set[str],
    seed: int = GLOBAL_SEED,
) -> list[dict[str, Any]]:
    categories = sorted({row["category"] for row in train})
    if len(categories) != 9:
        raise ValueError(f"Expected 9 categories, found {len(categories)}")
    rng = random.Random(seed)
    chosen: list[dict[str, Any]] = []
    for category in categories:
        eligible = sorted(
            (row for row in train if row["category"] == category and row["id"] not in calibration_ids and row["id"] not in dev_ids),
            key=lambda row: row["id"],
        )
        if not eligible:
            raise ValueError(f"No eligible few-shot row for {category}")
        chosen.append(rng.choice(eligible))
    return chosen


def validate_fewshot(rows: list[dict[str, Any]], calibration_ids: set[str], dev_ids: set[str], layout: DatasetLayout) -> None:
    ids = [row["id"] for row in rows]
    categories = [row["category"] for row in rows]
    if len(rows) != 9 or len(set(ids)) != 9 or len(set(categories)) != 9:
        raise ValueError("Few-shot manifest must contain one unique example per each of 9 categories")
    if set(ids) & calibration_ids:
        raise ValueError("Few-shot examples overlap calibration IDs")
    if set(ids) & dev_ids:
        raise ValueError("Few-shot examples overlap dev IDs")
    if categories != sorted(categories):
        raise ValueError("Few-shot order must follow sorted category names")
    for row in rows:
        if not row["question"].strip() or not row["answer"].strip() or not layout.image_exists("train", row["image"]):
            raise ValueError(f"Invalid few-shot example {row['id']}")


def create_fewshot_manifest(dataset_root: Path, calibration_path: Path, output: Path, *, seed: int = GLOBAL_SEED, force: bool = False) -> Path:
    if output.exists() and not force:
        load_fewshot(output)
        print(f"Reusing existing few-shot manifest: {output} (hash verified)")
        return output
    _, calibration = load_calibration(calibration_path)
    calibration_ids = {row["sample_id"] for row in calibration}
    layout = DatasetLayout(dataset_root)
    with assert_source_unchanged(dataset_root):
        train = layout.load_split("train")
        dev_ids = {row["id"] for row in layout.load_split("dev")}
        selected = select_fewshot(train, calibration_ids, dev_ids, seed)
        validate_fewshot(selected, calibration_ids, dev_ids, layout)
    payload = {
        "manifest_type": "fixed_9shot",
        "seed": seed,
        "creation_config": {"strategy": "one_per_sorted_category", "shots": 9, "excluded_calibration_manifest": str(calibration_path)},
        "source_split": "train",
        "source_dataset_revision": layout.revision,
        "category_order": [row["category"] for row in selected],
        "selected_ids": [row["id"] for row in selected],
        "demonstrations": [
            {
                "order": index,
                "sample_id": row["id"],
                "image": row["image"],
                "question": row["question"],
                "answer": row["answer"],
                "category": row["category"],
            }
            for index, row in enumerate(selected, start=1)
        ],
    }
    document = {**payload, "manifest_sha256": sha256_value(payload), "created_at": utc_now()}
    write_json_atomic(output, document)
    print(f"Wrote fixed 9-shot manifest to {output}; manifest_sha256={document['manifest_sha256']}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths-config", default="configs/paths.example.yaml")
    parser.add_argument("--calibration-manifest")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    dataset_root, artifact_root = load_paths(args.paths_config)
    calibration = Path(args.calibration_manifest).resolve() if args.calibration_manifest else artifact_root / "manifests" / DEFAULT_NAME
    output = Path(args.output).resolve() if args.output else artifact_root / "fewshot" / DEFAULT_NAME_FEWSHOT
    create_fewshot_manifest(dataset_root, calibration, output, seed=args.seed, force=args.force)


if __name__ == "__main__":
    main()
