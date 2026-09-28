from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from semeval27.data.calibration import create_manifest, load_calibration, sample_calibration
from semeval27.data.dataset import DatasetLayout, source_snapshot
from semeval27.data.fewshot import create_fewshot_manifest, select_fewshot
from semeval27.data.inspect_dataset import inspect_dataset
from semeval27.utils.io import sha256_value


def test_hugging_face_source_is_never_written(dataset_root: Path) -> None:
    before = source_snapshot(dataset_root)
    inspect_dataset(dataset_root)
    assert source_snapshot(dataset_root) == before


def test_calibration_sampling_is_deterministic(dataset_root: Path) -> None:
    rows = DatasetLayout(dataset_root).load_split("train")
    assert [row["id"] for row in sample_calibration(rows)] == [row["id"] for row in sample_calibration(list(reversed(rows)))]


def test_calibration_has_ten_per_nine_categories_and_90_unique_ids(dataset_root: Path, tmp_path: Path) -> None:
    output = tmp_path / "calibration.jsonl"
    create_manifest(dataset_root, output)
    metadata, rows = load_calibration(output)
    counts = Counter(row["category"] for row in rows)
    assert len(rows) == len({row["sample_id"] for row in rows}) == 90
    assert len(counts) == 9 and set(counts.values()) == {10}
    assert metadata["seed"] == 42
    assert metadata["manifest_sha256"]


def test_fewshot_one_per_category_no_calibration_or_dev_overlap(dataset_root: Path, tmp_path: Path) -> None:
    calibration = tmp_path / "calibration.jsonl"
    fewshot = tmp_path / "fewshot.json"
    create_manifest(dataset_root, calibration)
    create_fewshot_manifest(dataset_root, calibration, fewshot)
    document = json.loads(fewshot.read_text(encoding="utf-8"))
    _, calibration_rows = load_calibration(calibration)
    calibration_ids = {row["sample_id"] for row in calibration_rows}
    dev_ids = {row["id"] for row in DatasetLayout(dataset_root).load_split("dev")}
    demos = document["demonstrations"]
    assert len(demos) == len({row["category"] for row in demos}) == 9
    assert not ({row["sample_id"] for row in demos} & calibration_ids)
    assert not ({row["sample_id"] for row in demos} & dev_ids)
    assert [row["category"] for row in demos] == sorted(row["category"] for row in demos)


def test_fewshot_explicitly_excludes_train_ids_duplicated_in_dev(dataset_root: Path) -> None:
    layout = DatasetLayout(dataset_root)
    train = layout.load_split("train")
    duplicate = train[0]["id"]
    selected = select_fewshot(train, set(), {duplicate})
    assert duplicate not in {row["id"] for row in selected}


def test_manifest_changes_produce_different_hash() -> None:
    assert sha256_value({"selected_ids": ["a"]}) != sha256_value({"selected_ids": ["b"]})

