from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from semeval27.data.dataset import DatasetLayout, EXPECTED_FIELDS, assert_source_unchanged
from semeval27.utils.io import load_paths


def inspect_dataset(dataset_root: Path) -> dict[str, Any]:
    layout = DatasetLayout(dataset_root)
    with assert_source_unchanged(dataset_root):
        train = layout.load_split("train")
        dev = layout.load_split("dev")
        train_ids = {row["id"] for row in train}
        dev_ids = {row["id"] for row in dev}
        categories = Counter(row["category"] for row in train)
        archive_stats: dict[str, Any] = {}
        for split, rows in [("train", train), ("dev", dev)]:
            with zipfile.ZipFile(layout.image_archive(split)) as archive:
                names = {item.filename for item in archive.infolist() if not item.is_dir()}
            missing = sorted(row["image"] for row in rows if row["image"] not in names)
            archive_stats[split] = {"archive_files": len(names), "missing_images": missing}
    report = {
        "track": "qa_mena_en",
        "dataset_root": str(dataset_root),
        "dataset_revision": layout.revision,
        "train_samples": len(train),
        "dev_samples": len(dev),
        "train_unique_ids": len(train_ids),
        "dev_unique_ids": len(dev_ids),
        "cross_split_id_overlap": len(train_ids & dev_ids),
        "fields": sorted(train[0]),
        "fields_match_expected": set(train[0]) == EXPECTED_FIELDS,
        "categories": dict(sorted(categories.items())),
        "category_count": len(categories),
        "exactly_nine_categories": len(categories) == 9,
        "image_path_convention": "images/<sha256-id>.jpg",
        "image_storage": "ZIP archives (or extracted paths when present)",
        "archives": archive_stats,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths-config", default="configs/paths.example.yaml")
    args = parser.parse_args()
    dataset_root, _ = load_paths(args.paths_config)
    report = inspect_dataset(dataset_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["train_samples"] <= 24 or report["dev_samples"] == 0:
        raise SystemExit("Demo/sample release detected; full qa_mena_en data is required")
    if not report["exactly_nine_categories"]:
        raise SystemExit("Expected exactly 9 train categories")
    if any(report["archives"][split]["missing_images"] for split in ["train", "dev"]):
        raise SystemExit("One or more referenced images are missing from the media archives")


if __name__ == "__main__":
    main()

