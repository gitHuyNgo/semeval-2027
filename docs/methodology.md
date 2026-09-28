# Methodology

## Inputs and leakage boundary

At test time MMCultureQA supplies the image and question, while country,
category, and subcategory are withheld. The model-facing renderer therefore
accepts only an image path and question for a target. Taxonomy fields are used
offline for sampling and analysis, never serialized into a model request.

## Prompt calibration

Prompt calibration uses TRAIN rather than DEV so the official development set
remains an untouched benchmark. Ninety examples—ten sampled from each of the
nine current train categories with seed 42—give every broad topic equal
representation while keeping the 720-prediction calibration experiment
manageable.

There are exactly four candidate prompts. They deliberately cover a minimal
instruction, concise output, explicit cultural-knowledge allowance, and a more
complete answer style. This small, declared set prevents an open-ended prompt
search. Qwen3-VL-4B-Instruct and Gemma-3-4B-it perform calibration because they
are similarly sized reproducible open models from different families. GPT and
Gemini results cannot influence prompt selection.

The preferred prompt is the arithmetic mean of the two model scores, with both
individual scores shown. A report only recommends a prompt. A human must issue
an explicit freeze command before dev benchmarking. If a future official scorer
changes the ranking, comparison is reported and another explicit decision is
required; the repository never silently changes the frozen prompt.

## Fixed 9-shot regime

The fixed demonstrations contain one TRAIN example from each sorted category,
excluding all calibration IDs and any ID shared with DEV. Category metadata is
legitimate for this offline balancing step but is omitted from the request.
Every benchmark model receives the same nine IDs in the same order, with only
image, question, and gold TRAIN answer. The target follows with its image,
question, and the same frozen semantic instruction used by zero-shot systems.

Provider syntax necessarily differs, but semantics do not: Qwen and Gemma use
their processor chat templates, OpenAI uses its native multimodal message
schema, and Gemini uses native content/part objects. No handwritten special
tokens substitute for an available official template.

## Scoring status and preservation

The current local `QCRI/MMCQA-SemEval27` release has no organizer evaluation
script. Until it appears, development uses canonical upstream
`bert-score==0.3.13`: English `roberta-large`, its default layer, no IDF, no
baseline rescaling, slow tokenizer, and arithmetic macro mean of example-level
F1. Predictions and references are used verbatim without lowercasing,
punctuation/article removal, extraction, truncation, or explanation stripping.

Every provisional artifact says: **THIS IS NOT YET THE OFFICIAL MMCultureQA
SCORER.** It must never be called an official score. Raw predictions are stored
independently, allowing all candidates and systems to be rescored without model
inference. A sync command checks each new local dataset release. Once organizer
code appears, it is copied unchanged, hash-verified, recorded in provenance,
and invoked through a wrapper instead of being reimplemented. A comparison tool
then reports aggregate, per-example, and ranking differences.

## Precision and reproducibility

Official scientific open-model runs use native BF16 when supported. Four-bit
inference is permitted only through an explicit debug flag and is labeled as
non-scientific; it must never silently replace the native-precision result.
Manifests record the seed, creation rule, dataset revision, ordered IDs, and a
content hash. Existing manifests and successful prediction rows are reused by
default. Experiment metadata records model/dataset revisions, prompt and
few-shot hashes, inference settings, software/CUDA versions, timestamps,
repository commit, and scorer status.

