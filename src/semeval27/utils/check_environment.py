from __future__ import annotations

import importlib.metadata

from packaging.version import Version


EXACT_VERSIONS = {
    "bert-score": "0.3.13",
    "PyYAML": "6.0.2",
    "transformers": "4.57.5",
}


def installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_errors() -> list[str]:
    errors = []
    for distribution, expected in EXACT_VERSIONS.items():
        installed = installed_version(distribution)
        if installed != expected:
            errors.append(f"{distribution}: expected {expected}, found {installed or 'not installed'}")

    hub = installed_version("huggingface-hub")
    if hub is None:
        errors.append("huggingface-hub: not installed")
    elif not Version("0.34") <= Version(hub) < Version("1"):
        errors.append(f"huggingface-hub: expected >=0.34,<1, found {hub}")
    return errors


def main() -> None:
    distributions = [*EXACT_VERSIONS, "huggingface-hub", "torch"]
    for distribution in distributions:
        print(f"{distribution}: {installed_version(distribution) or 'not installed'}")

    errors = environment_errors()
    if errors:
        print("\nEnvironment check FAILED:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("\nEnvironment check passed.")
    print("Note: bert_score.__version__ may report 0.3.12 for the 0.3.13 distribution;")
    print("the project and evaluator use installed distribution metadata instead.")


if __name__ == "__main__":
    main()
