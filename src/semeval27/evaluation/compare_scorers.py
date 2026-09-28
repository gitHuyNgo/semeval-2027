from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from semeval27.evaluation.provisional_bertscore import WARNING
from semeval27.utils.io import write_json_atomic
from semeval27.utils.reproducibility import utc_now


def load_reports(root: Path) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            reports[path.stem] = json.load(handle)
    return reports


def compare(provisional: dict[str, dict[str, Any]], official: dict[str, dict[str, Any]]) -> dict[str, Any]:
    systems = sorted(provisional.keys() & official.keys())
    rows = []
    for system in systems:
        p_report, o_report = provisional[system], official[system]
        p_score = float(p_report["macro_f1"])
        o_score = float(o_report.get("official_score", o_report.get("macro_f1")))
        p_examples = {item["sample_id"]: float(item["f1"]) for item in p_report.get("per_example", [])}
        o_examples = {
            item["sample_id"]: float(item.get("score", item.get("f1")))
            for item in o_report.get("per_example", [])
        }
        common = sorted(p_examples.keys() & o_examples.keys())
        rows.append(
            {
                "system_id": system,
                "provisional_score": p_score,
                "official_score": o_score,
                "absolute_difference": abs(p_score - o_score),
                "per_example_differences": [
                    {
                        "sample_id": sample_id,
                        "provisional": p_examples[sample_id],
                        "official": o_examples[sample_id],
                        "absolute_difference": abs(p_examples[sample_id] - o_examples[sample_id]),
                    }
                    for sample_id in common
                ],
            }
        )
    p_order = sorted(rows, key=lambda row: (-row["provisional_score"], row["system_id"]))
    o_order = sorted(rows, key=lambda row: (-row["official_score"], row["system_id"]))
    p_rank = {row["system_id"]: index for index, row in enumerate(p_order, 1)}
    o_rank = {row["system_id"]: index for index, row in enumerate(o_order, 1)}
    for row in rows:
        row["provisional_rank"] = p_rank[row["system_id"]]
        row["official_rank"] = o_rank[row["system_id"]]
        row["rank_difference"] = abs(row["provisional_rank"] - row["official_rank"])
    return {
        "warning": WARNING,
        "status": "SCORER_COMPARISON",
        "generated_at": utc_now(),
        "preferred_system_changed": bool(rows) and p_order[0]["system_id"] != o_order[0]["system_id"],
        "requires_explicit_prompt_decision": bool(rows) and p_order[0]["system_id"] != o_order[0]["system_id"],
        "systems": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare saved provisional/official scores without model inference")
    parser.add_argument("--provisional-dir", required=True)
    parser.add_argument("--official-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = compare(load_reports(Path(args.provisional_dir)), load_reports(Path(args.official_dir)))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / "comparison.json", result)
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["system_id", "provisional_score", "official_score", "absolute_difference", "provisional_rank", "official_rank", "rank_difference"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in result["systems"]:
            writer.writerow({key: row[key] for key in fields})
    print(json.dumps({key: value for key, value in result.items() if key != "systems"}, indent=2))


if __name__ == "__main__":
    main()

