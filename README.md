# SemEval 2027 — MMCultureQA

Reproducible research code for Task 2 English (`qa_mena_en`). The model is
given only the image and question. Dataset metadata (`country`, `category`, and
`subcategory`) is restricted to offline stratification and analysis.

> **Scoring status:** the current local dataset release does not contain the
> organizer scorer. Development reports use a pinned, explicitly configured
> provisional BERTScore implementation and are marked **PROVISIONAL — THIS IS
> NOT YET THE OFFICIAL MMCultureQA SCORER.** Raw predictions are retained so
> they can be rescored later without rerunning inference.

## Setup

```powershell
cd D:\Documents\A_Research\SemEval\semeval-2027
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
Copy-Item configs/paths.example.yaml configs/paths.yaml
```

Install local-model or API dependencies only when needed:

```powershell
python -m pip install -e ".[local]"       # PyTorch/Transformers VLMs
python -m pip install -e ".[openai]"      # OpenAI runner
python -m pip install -e ".[gemini]"      # Gemini runner
```

The evaluator requires the pinned `bert-score==0.3.13` and
`transformers==4.57.5` combination. Transformers 5.x is intentionally
excluded because it is incompatible with BERTScore 0.3.13's handling of empty
candidate strings. Do not independently upgrade `transformers` or
`huggingface-hub` in this environment. After installation, verify the complete
environment with:

```powershell
python -m pip check
python -m semeval27.utils.check_environment
```

The upstream 0.3.13 BERTScore package may expose
`bert_score.__version__ == "0.3.12"`. This project checks the installed
distribution metadata, which must report 0.3.13.

Set credentials from `.env.example` in the shell. `HF_TOKEN` is optional for
public models unless Hugging Face access terms require it.

## Workflow

All commands accept `--paths-config configs/paths.yaml`; the examples use the
checked-in example config, which points to the sibling dataset folder.

```powershell
# 1. Inspect and validate the current dataset release.
python -m semeval27.data.inspect_dataset --paths-config configs/paths.example.yaml

# 2. Look for a newly released official scorer (currently expected to fail).
python -m semeval27.evaluation.sync_official_scorer --paths-config configs/paths.example.yaml

# 3. Reuse or create the deterministic 90-example TRAIN calibration manifest.
python -m semeval27.data.calibration --paths-config configs/paths.example.yaml

# 4. Inexpensive local smoke tests (downloads model weights if not cached).
python -m semeval27.experiments.prompt_calibration --model qwen4b --prompt-id P1 --limit 2 --paths-config configs/paths.example.yaml
python -m semeval27.experiments.prompt_calibration --model gemma4b --prompt-id P1 --limit 2 --paths-config configs/paths.example.yaml

# 5–6. Run each of P1–P4. Completed successful rows are resumed by default.
python -m semeval27.experiments.prompt_calibration --model qwen4b --all-prompts --paths-config configs/paths.example.yaml
python -m semeval27.experiments.prompt_calibration --model gemma4b --all-prompts --paths-config configs/paths.example.yaml

# 7. Score each saved prediction file provisionally (repeat for all eight).
python -m semeval27.evaluation.provisional_bertscore --predictions artifacts/predictions/calibration/qwen4b_P1.jsonl --output artifacts/scores/calibration/qwen4b_P1.json

# 8. Build the provisional cross-model calibration table/report.
python -m semeval27.evaluation.summarize calibration --artifact-root artifacts

# 9. Explicitly freeze the reviewed prompt (never done automatically).
python -m semeval27.prompting.freeze --prompt-id P2

# 10. Reuse or create the fixed, taxonomy-balanced 9-shot manifest.
python -m semeval27.data.fewshot --paths-config configs/paths.example.yaml

# 11. Smoke-test a benchmark system.
python -m semeval27.experiments.benchmark --system-id B05 --limit 2 --paths-config configs/paths.example.yaml

# 12. Run B01–B10. Paid full API runs require --confirm-full-api.
python -m semeval27.experiments.benchmark --system-id B01 --confirm-full-api --paths-config configs/paths.example.yaml
python -m semeval27.experiments.benchmark --system-id B05 --paths-config configs/paths.example.yaml

# 13. Score a complete system provisionally.
python -m semeval27.evaluation.provisional_bertscore --predictions artifacts/predictions/benchmark/B05.jsonl --output artifacts/scores/benchmark/B05.json

# Inspect the aggregate and per-example provisional F1 values.
python -c "import json; r=json.load(open('artifacts/scores/benchmark/B05.json', encoding='utf-8')); print(r['macro_f1']); print(r['per_example'][:3])"

# 14. Validate completeness and make ranked provisional CSV/Markdown tables
#     for every configured system (currently B01-B10 plus FT01).
python -m semeval27.evaluation.summarize benchmark --artifact-root artifacts --paths-config configs/paths.example.yaml

# View the Markdown table.
cat artifacts/reports/benchmark_provisional.md
```

Gemini uses the provider-specific `gemini.max_output_tokens` setting because
Gemini's limit includes both internal thinking and visible answer tokens. Its
saved prediction rows include finish reason and token-usage metadata so empty
visible responses can be audited without rerunning inference.

The optional `FT01` system evaluates the pinned
`anhbilong/qwen3-vl-4b-mmcultureqa-split512` PEFT adapter zero-shot. It loads
the pinned Qwen3-VL-4B base model, applies the adapter with PEFT, and uses the
adapter repository's 65,536-to-262,144-pixel processor configuration:

```powershell
python -m semeval27.experiments.benchmark --system-id FT01 --limit 5 --paths-config configs/paths.example.yaml
python -m semeval27.experiments.benchmark --system-id FT01 --paths-config configs/paths.example.yaml
python -m semeval27.evaluation.provisional_bertscore --predictions artifacts/predictions/benchmark/FT01.jsonl --output artifacts/scores/benchmark/FT01.json
```

The adapter's published split manifest contains a stratified 9,000/1,000 split
of the 10,000-example MMCultureQA training set. Its IDs were checked against the
local release: both subsets are contained in `train_en.jsonl`, are mutually
exclusive, and have zero overlap with the 1,000-example `dev_en.jsonl`
benchmark split. Preserve the pinned adapter revision and split manifest with
the result provenance.

When organizer code becomes available, sync it verbatim, run the optional
wrapper against the same prediction files, and compare the saved score reports:

```powershell
python -m semeval27.evaluation.sync_official_scorer --paths-config configs/paths.example.yaml
python -m semeval27.evaluation.compare_scorers --provisional-dir artifacts/scores --official-dir artifacts/official_scores --output-dir artifacts/reports/scorer_comparison
```

If official scoring changes the preferred prompt, comparison reports flag it;
they never alter `configs/prompts/frozen_prompt.yaml`. Freezing another prompt
always requires another explicit `freeze` command.

## Safety and precision

- The sibling dataset repository is opened read-only and guarded by before/after
  source snapshots. Images are read directly from the organizer ZIP archives.
- API requests are incremental and resumable. Successful predictions are not
  resent unless `--force` is supplied.
- Scientific open-model runs require native BF16-capable hardware. Quantization
  is available only with the explicit debug flag and is labeled accordingly.
- No paid full API run starts without `--confirm-full-api`.

See [docs/methodology.md](docs/methodology.md) for the experimental rationale.
