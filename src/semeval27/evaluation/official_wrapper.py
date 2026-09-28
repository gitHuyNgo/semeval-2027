from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from semeval27.utils.io import REPO_ROOT


def require_verified_vendor(vendor_root: Path) -> Path:
    provenance = vendor_root / "PROVENANCE.md"
    scorer_files = [path for path in vendor_root.rglob("*") if path.is_file() and path.name != "PROVENANCE.md"]
    if not provenance.is_file() or not scorer_files:
        raise RuntimeError(
            "Verified organizer scorer is unavailable. Run sync_official_scorer after it appears in the dataset release."
        )
    return provenance


def invoke_official(command: str, predictions: Path, output: Path, vendor_root: Path) -> None:
    """Invoke organizer code as-is; the command interface must be set after release."""
    require_verified_vendor(vendor_root)
    tokens = [part.format(predictions=str(predictions), output=str(output), vendor=str(vendor_root)) for part in shlex.split(command)]
    subprocess.run(tokens, check=True, cwd=vendor_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Thin wrapper for future verified organizer scorer")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--command",
        required=True,
        help="Organizer CLI template; placeholders: {predictions}, {output}, {vendor}",
    )
    parser.add_argument("--vendor-root", default=str(REPO_ROOT / "vendor" / "mmcqa_official_eval"))
    args = parser.parse_args()
    invoke_official(args.command, Path(args.predictions).resolve(), Path(args.output).resolve(), Path(args.vendor_root).resolve())


if __name__ == "__main__":
    main()

