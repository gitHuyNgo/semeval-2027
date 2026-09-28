from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest


CATEGORIES = [
    "Culture, Arts & Entertainment",
    "Food & Cooking",
    "Geography, Buildings & Landmarks",
    "History, Geography & National Identity",
    "Objects, Materials & Clothing",
    "People, Society & Education",
    "Religion & Spirituality",
    "Sports & Recreation",
    "Vehicles & Transportation",
]


@pytest.fixture()
def dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    (root / "qa" / "mena").mkdir(parents=True)
    (root / "archives").mkdir()
    (root / ".cache" / "huggingface" / "trees").mkdir(parents=True)
    (root / ".cache" / "huggingface" / "trees" / ("a" * 40 + ".json")).write_text("{}", encoding="utf-8")
    train = []
    for category_index, category in enumerate(CATEGORIES):
        for item_index in range(12):
            sample_id = f"train-{category_index}-{item_index}"
            train.append(
                {
                    "id": sample_id,
                    "image": f"images/{sample_id}.jpg",
                    "country": "offline-country",
                    "category": category,
                    "subcategory": "offline-subcategory",
                    "question": f"Question {sample_id}?",
                    "answer": f"Answer {sample_id}",
                }
            )
    dev = []
    for index in range(10):
        sample_id = f"dev-{index}"
        dev.append(
            {
                "id": sample_id,
                "image": f"images/{sample_id}.jpg",
                "country": "offline-country",
                "category": CATEGORIES[index % 9],
                "subcategory": "offline-subcategory",
                "question": f"Dev question {index}?",
                "answer": f"Dev answer {index}",
            }
        )
    for split, rows in [("train", train), ("dev", dev)]:
        with (root / "qa" / "mena" / f"{split}_en.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        with zipfile.ZipFile(root / "archives" / f"images_mena_{split}.zip", "w") as archive:
            for row in rows:
                archive.writestr(row["image"], b"fake-jpeg")
    return root

