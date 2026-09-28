from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from semeval27 import GLOBAL_SEED
from semeval27.data.dataset import DatasetLayout, assert_source_unchanged
from semeval27.utils.io import load_paths, read_jsonl, sha256_value, write_jsonl_atomic
from semeval27.utils.reproducibility import utc_now


DEFAULT_NAME = "prompt_calibration_seed42.jsonl"


def sample_calibration(rows: list[dict[str, Any]], seed: int = GLOBAL_SEED, per_category: int = 10) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    categories = sorted(grouped)
    if len(categories) != 9:
        raise ValueError(f"Expected 9 categories, found {len(categories)}")
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for category in categories:
        population = sorted(grouped[category], key=lambda item: item["id"])
        if len(population) < per_category:
            raise ValueError(f"Category {category!r} has fewer than {per_category} rows")
        selected.extend(rng.sample(population, per_category))
    return selected


def validate_calibration(rows: list[dict[str, Any]], layout: DatasetLayout) -> None:
    if len(rows) != 90:
        raise ValueError(f"Expected 90 calibration rows, found {len(rows)}")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != 90:
        raise ValueError("Calibration IDs are not unique")
    counts = Counter(row["category"] for row in rows)
    if len(counts) != 9 or set(counts.values()) != {10}:
        raise ValueError(f"Expected 10 rows in each of 9 categories, found {dict(counts)}")
    for row in rows:
        if not row["question"].strip() or not row["answer"].strip():
            raise ValueError(f"Empty question or answer for {row['id']}")
        if not layout.image_exists("train", row["image"]):
            raise ValueError(f"Missing image for {row['id']}: {row['image']}")


def logical_payload(rows: list[dict[str, Any]], layout: DatasetLayout, seed: int) -> dict[str, Any]:
    return {
        "manifest_type": "prompt_calibration",
        "seed": seed,
        "creation_config": {"strategy": "stratified_random", "per_category": 10, "category_order": sorted({r["category"] for r in rows})},
        "source_split": "train",
        "source_dataset_revision": layout.revision,
        "selected_ids": [row["id"] for row in rows],
    }


def create_manifest(dataset_root: Path, output: Path, *, seed: int = GLOBAL_SEED, force: bool = False) -> Path:
    if output.exists() and not force:
        _, samples = load_calibration(output)
        print(f"Reusing existing calibration manifest: {output} ({len(samples)} samples, hash verified)")
        return output
    layout = DatasetLayout(dataset_root)
    with assert_source_unchanged(dataset_root):
        rows = sample_calibration(layout.load_split("train"), seed)
        validate_calibration(rows, layout)
    payload = logical_payload(rows, layout, seed)
    manifest_hash = sha256_value(payload)
    metadata = {
        "record_type": "metadata",
        **payload,
        "manifest_sha256": manifest_hash,
        "created_at": utc_now(),
    }
    samples = [
        {
            "record_type": "sample",
            "sample_id": row["id"],
            "image": row["image"],
            "question": row["question"],
            "answer": row["answer"],
            "category": row["category"],
        }
        for row in rows
    ]
    write_jsonl_atomic(output, [metadata, *samples])
    print(f"Wrote {len(samples)} samples to {output}; manifest_sha256={manifest_hash}")
    return output


def load_calibration(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = read_jsonl(path)
    if not records or records[0].get("record_type") != "metadata":
        raise ValueError("Calibration manifest lacks metadata header")
    metadata = records[0]
    samples = [row for row in records[1:] if row.get("record_type") == "sample"]
    payload = {
        key: metadata[key]
        for key in [
            "manifest_type",
            "seed",
            "creation_config",
            "source_split",
            "source_dataset_revision",
            "selected_ids",
        ]
    }
    if sha256_value(payload) != metadata.get("manifest_sha256"):
        raise ValueError("Calibration manifest hash does not match its metadata")
    if [row["sample_id"] for row in samples] != metadata["selected_ids"]:
        raise ValueError("Calibration sample order/IDs do not match manifest metadata")
    return metadata, samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths-config", default="configs/paths.example.yaml")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    dataset_root, artifact_root = load_paths(args.paths_config)
    output = Path(args.output) if args.output else artifact_root / "manifests" / DEFAULT_NAME
    create_manifest(dataset_root, output.resolve(), seed=args.seed, force=args.force)


if __name__ == "__main__":
    main()

