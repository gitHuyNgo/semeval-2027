from __future__ import annotations

import argparse
from pathlib import Path

from semeval27.prompting.renderer import load_candidates, prompt_hash
from semeval27.utils.io import REPO_ROOT, write_json_atomic
from semeval27.utils.reproducibility import utc_now


def freeze_prompt(prompt_id: str, candidates_path: Path, output: Path) -> Path:
    prompts = load_candidates(candidates_path)
    if prompt_id not in prompts:
        raise ValueError(f"Unknown prompt ID {prompt_id}; choose one of {sorted(prompts)}")
    template = prompts[prompt_id]
    document = {
        "status": "frozen",
        "version": 1,
        "prompt_id": prompt_id,
        "template": template,
        "prompt_hash": prompt_hash(prompt_id, template),
        "frozen_at": utc_now(),
        "decision": "explicit_user_command",
    }
    write_json_atomic(output, document)  # JSON is valid YAML and preserves exact newlines.
    print(f"Frozen {prompt_id} at {output}; prompt_hash={document['prompt_hash']}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Explicitly freeze one reviewed calibration prompt")
    parser.add_argument("--prompt-id", required=True, choices=["P1", "P2", "P3", "P4"])
    parser.add_argument("--candidates", default=str(REPO_ROOT / "configs" / "prompts" / "calibration_candidates.yaml"))
    parser.add_argument("--output", default=str(REPO_ROOT / "configs" / "prompts" / "frozen_prompt.yaml"))
    args = parser.parse_args()
    freeze_prompt(args.prompt_id, Path(args.candidates).resolve(), Path(args.output).resolve())


if __name__ == "__main__":
    main()
