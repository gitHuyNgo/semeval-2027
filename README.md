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

# 14. Validate completeness and make provisional CSV/Markdown tables.
python -m semeval27.evaluation.summarize benchmark --artifact-root artifacts --paths-config configs/paths.example.yaml
```

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
