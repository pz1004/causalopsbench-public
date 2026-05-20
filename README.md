# CausalOpsBench Public Reproducibility Release

This repository contains the public reproduction bundle for CausalOpsBench:
source code, experiment scripts, tests, compact documentation, native benchmark
artifacts, local-agent outputs, external-validation summaries, and analysis
tables/figures.

Public URL: <https://github.com/pz1004/causalopsbench-public>

## What Is Included

- `causalopsbench/`: native synthetic benchmark package.
- `cob_ext/`: external-trace portability adapters, runners, scoring, and reporting.
- `scripts/`: native, local-agent, contamination, and analysis scripts.
- `tests/`: unit and smoke tests for the public code path.
- `docs/`, `examples/`, `data/`: benchmark notes, schema examples, and tiny dev fixtures.
- `results/cob_v2*`: v2 native scaffold artifacts, held-out seed artifacts, local Ollama results, ablation/prompt/contamination inputs, model-card audit files, and final analysis outputs.
- `external_validation/`: public-dataset validation configs, processed manifests, and compact output summaries/predictions/scores.

## What Is Omitted

- Article source files are outside this public code bundle.
- Legacy v1, smoke-only, and superseded result directories.
- Raw external datasets under `external_validation/data/raw/`.
- Large processed external episode JSONL files under `external_validation/data/processed/episodes/`.
- Python caches, build metadata, and unrelated logs.

Raw RCAEval and LBNL FDD data should be downloaded separately according to
`external_validation/README.md`; the adapters in `cob_ext.adapters` regenerate
the processed external episodes from those public sources.

## Quick Start

```bash
cd causalopsbench-public
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -v
```

Run a small native benchmark sample:

```bash
python -m causalopsbench generate --count 10 --seed 7 --output data/dev.jsonl
python -m causalopsbench evaluate \
  --episodes data/dev.jsonl \
  --baseline threshold \
  --output data/threshold_results.json
```

## Reproduce Native Scaffold Results

The included v2 native scaffold artifacts live in `results/cob_v2/`.
To regenerate them in place:

```bash
python scripts/run_experiments.py \
  --output-dir results/cob_v2 \
  --seeds 0 1 2 3 4 \
  --count 500 \
  --duration 60 \
  --domain all \
  --topology-variant public \
  --baselines noop random threshold topology_rca oracle
```

Held-out topology audit:

```bash
python scripts/run_experiments.py \
  --output-dir results/cob_v2_hidden_seed \
  --seeds 20260520 \
  --count 500 \
  --duration 60 \
  --domain all \
  --topology-variant heldout_v1 \
  --baselines noop random threshold topology_rca oracle
```

Expected native headline means are in `results/cob_v2/summary.csv`.

## Reproduce Local Ollama Agent Runs

The release-track local-agent outputs are included under `results/cob_v2_ollama/`
and `results/cob_v2_hidden_seed_ollama/`. Re-running them requires Ollama,
the listed model tags, and enough GPU/CPU resources for the selected models.

```bash
ollama pull gemma4:e2b
ollama pull qwen3.5:4b
ollama pull granite4.1:3b
ollama pull gemma4:e4b
ollama pull qwen3.5:9b
ollama pull granite4.1:8b
ollama pull gemma4:26b
```

```bash
python scripts/run_foundation_agent_experiments.py \
  --episodes-dir results/cob_v2/episodes \
  --output-dir results/cob_v2_ollama \
  --models ollama:gemma4:e2b ollama:qwen3.5:4b \
    ollama:granite4.1:3b ollama:gemma4:e4b \
    ollama:qwen3.5:9b ollama:granite4.1:8b \
    ollama:gemma4:26b \
  --policy react-json \
  --max-steps 4 \
  --temperature 0 \
  --num-ctx 8192 \
  --keep-alive 5m \
  --think false \
  --seed 0 \
  --timeout-s 120
```

## Reproduce Analysis Tables And Figures

The compact final analysis outputs are already included in
`results/cob_v2_analysis/`. To regenerate the tables and figures from the
included result inputs:

```bash
python scripts/analyze_revision_artifacts.py \
  --scaffold-dir results/cob_v2 \
  --ollama-dir results/cob_v2_ollama \
  --ablation-root results/cob_v2_ollama_ablation \
  --extra-ablation-root results/cob_v2_ollama_ablation_extra \
  --contamination-dir results/cob_v2_contamination_all_domains \
  --joint-contamination-dir results/cob_v2_contamination_no_topology \
  --prompt-root results/cob_v2_prompt_sweep \
  --hidden-scaffold-dir results/cob_v2_hidden_seed \
  --hidden-ollama-dir results/cob_v2_hidden_seed_ollama \
  --model-card-dir results/cob_v2_ollama_model_cards \
  --output-dir results/cob_v2_analysis \
  --bootstrap 10000 \
  --seed 20260516 \
  --random-weight-samples 1000 \
  --random-weight-seed 20260516
```

## External Validation

External validation is intentionally separate from the native replay benchmark.
It checks portability on public RCAEval RE2 and LBNL FDD traces, and the scorer
refuses unsupported native replay metrics for those data.

Use `external_validation/README.md` for dataset download and conversion
commands. This public bundle includes configs, processed manifests, and compact
output artifacts, but not the raw datasets or large processed episode files.

## Development Checks

```bash
python -m compileall -q causalopsbench cob_ext scripts tests
python -m unittest discover -v
```

The package is pure Python and has no required third-party runtime dependency.
Optional local-agent experiments depend on external Ollama tooling rather than
Python packages.
