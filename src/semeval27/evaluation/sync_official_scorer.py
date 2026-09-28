from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from semeval27.data.dataset import assert_source_unchanged, dataset_revision
from semeval27.utils.io import REPO_ROOT, load_paths, sha256_file
from semeval27.utils.reproducibility import utc_now


SCORER_EXTENSIONS = {".py", ".sh", ".r", ".js", ".ipynb"}
SCORER_TERMS = ("score", "scorer", "scoring", "eval", "evaluation", "metric", "bertscore")


def discover_scorer_files(dataset_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file() or ".cache" in path.parts or path.suffix.lower() not in SCORER_EXTENSIONS:
            continue
        relative = path.relative_to(dataset_root).as_posix().lower()
        if any(term in relative for term in SCORER_TERMS):
            candidates.append(path)
    return candidates


def sync_files(dataset_root: Path, destination: Path, sources: list[Path]) -> list[dict[str, str]]:
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    with assert_source_unchanged(dataset_root):
        for source in sources:
            resolved = source.resolve()
            try:
                relative = resolved.relative_to(dataset_root.resolve())
            except ValueError as exc:
                raise ValueError(f"Official scorer source must be inside dataset root: {source}") from exc
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(resolved, target)
            source_hash = sha256_file(resolved)
            target_hash = sha256_file(target)
            if source_hash != target_hash:
                raise RuntimeError(f"Byte verification failed for {relative}")
            records.append(
                {
                    "source": str(resolved),
                    "destination": str(target.resolve()),
                    "source_sha256": source_hash,
                    "destination_sha256": target_hash,
                }
            )
    lines = [
        "# Official MMCultureQA scorer provenance",
        "",
        f"- Source repository: `{dataset_root.resolve()}`",
        f"- Source revision: `{dataset_revision(dataset_root) or 'unavailable'}`",
        f"- Synchronized at: `{utc_now()}`",
        "- Copy policy: files below were copied byte-for-byte without modification.",
        "",
        "| Source | Destination | Source SHA256 | Destination SHA256 |",
        "|---|---|---|---|",
    ]
    lines.extend(
        f"| `{row['source']}` | `{row['destination']}` | `{row['source_sha256']}` | `{row['destination_sha256']}` |"
        for row in records
    )
    (destination / "PROVENANCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover and byte-copy a future organizer scorer")
    parser.add_argument("--paths-config", default="configs/paths.example.yaml")
    parser.add_argument("--source-file", action="append", default=[])
    parser.add_argument("--destination", default=str(REPO_ROOT / "vendor" / "mmcqa_official_eval"))
    args = parser.parse_args()
    dataset_root, _ = load_paths(args.paths_config)
    sources = [Path(value).resolve() for value in args.source_file] or discover_scorer_files(dataset_root)
    if not sources:
        raise SystemExit(
            "No organizer scorer exists in the current local dataset release. "
            "Development scoring remains PROVISIONAL; nothing was copied."
        )
    records = sync_files(dataset_root, Path(args.destination).resolve(), sources)
    print(f"Copied and verified {len(records)} organizer scorer file(s). Review PROVENANCE.md before use.")


if __name__ == "__main__":
    main()

