from __future__ import annotations

import io
import json
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


EXPECTED_FIELDS = {"id", "image", "country", "category", "subcategory", "question", "answer"}
FORBIDDEN_MODEL_FIELDS = {"country", "category", "subcategory"}


def dataset_revision(root: Path) -> str | None:
    trees = root / ".cache" / "huggingface" / "trees"
    if not trees.is_dir():
        return None
    revisions = sorted(path.stem for path in trees.glob("*.json") if len(path.stem) == 40)
    return revisions[-1] if revisions else None


def source_snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@contextmanager
def assert_source_unchanged(root: Path) -> Iterator[None]:
    before = source_snapshot(root)
    yield
    after = source_snapshot(root)
    if before != after:
        changed = sorted(set(before) ^ set(after) | {key for key in before.keys() & after if before[key] != after[key]})
        raise RuntimeError(f"Dataset source was modified: {changed}")


@dataclass(frozen=True)
class DatasetLayout:
    root: Path

    def split_path(self, split: str) -> Path:
        if split not in {"train", "dev"}:
            raise ValueError(f"Unsupported split: {split}")
        return self.root / "qa" / "mena" / f"{split}_en.jsonl"

    def image_archive(self, split: str) -> Path:
        return self.root / "archives" / f"images_mena_{split}.zip"

    def load_split(self, split: str) -> list[dict[str, Any]]:
        path = self.split_path(split)
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        for index, row in enumerate(rows):
            missing = EXPECTED_FIELDS - row.keys()
            if missing:
                raise ValueError(f"{path}:{index + 1} missing fields: {sorted(missing)}")
        return rows

    def image_exists(self, split: str, relative_path: str) -> bool:
        direct = self.root / relative_path
        if direct.is_file():
            return True
        archive = self.image_archive(split)
        if not archive.is_file():
            return False
        normalized = relative_path.replace("\\", "/")
        with zipfile.ZipFile(archive) as handle:
            try:
                handle.getinfo(normalized)
                return True
            except KeyError:
                return False

    def read_image_bytes(self, split: str, relative_path: str) -> bytes:
        direct = self.root / relative_path
        if direct.is_file():
            return direct.read_bytes()
        normalized = relative_path.replace("\\", "/")
        with zipfile.ZipFile(self.image_archive(split)) as handle:
            return handle.read(normalized)

    def open_pil_image(self, split: str, relative_path: str):
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Install the local or gemini extra to load images") from exc
        image = Image.open(io.BytesIO(self.read_image_bytes(split, relative_path)))
        return image.convert("RGB")

    @property
    def revision(self) -> str | None:
        return dataset_revision(self.root)


def safe_model_sample(row: dict[str, Any]) -> dict[str, str]:
    """Return the only target fields permitted to reach a model renderer."""
    return {"id": str(row["id"]), "image": str(row["image"]), "question": str(row["question"])}

