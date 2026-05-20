# CausalOpsBench Public Release Verification Report

Verification date: 2026-05-20

Release candidate: <https://github.com/pz1004/causalopsbench-public>

Scope: standalone public repository containing code, tests, documentation,
compact fixtures, native benchmark artifacts, local-agent outputs, external
validation summaries, and analysis tables/figures. Article source files are
outside this public bundle.

## Environment Summary

- Python version observed during verification: `3.13.12`
- Package: `causalopsbench` version `0.1.0`
- Runtime dependencies: Python standard library only
- Optional local-agent dependency: Ollama with the recorded model tags
- Recorded local-agent hardware: NVIDIA Quadro RTX 8000, 49152 MiB VRAM
- Recorded local-agent settings: Ollama 0.23.4, `num_ctx=8192`,
  `temperature=0`, `keep_alive=5m`, and `think=false`

## Checks Performed

- `python -m compileall -q causalopsbench cob_ext scripts tests` passed.
- `python -m unittest discover -v` passed with 38 tests.
- CLI smoke generation passed with 10 deterministic development episodes.
- CLI smoke evaluation passed for the `threshold` and `oracle` baselines.
- `scripts/run_experiments.py` smoke passed on 5 episodes with `noop`,
  `threshold`, and `oracle` baselines.
- `scripts/analyze_revision_artifacts.py` smoke passed against the included
  result artifacts and generated 35 temporary analysis files.
- `python -m pip install --dry-run .` passed; the package resolves as
  `causalopsbench-0.1.0`.
- Static disclosure checks were run for stale workspace paths, old repository
  URLs, private-source references, placeholder markers, Python caches, and
  obvious secret values.

## Included Artifact Summary

Native scaffold artifacts:

- `results/cob_v2/summary.csv`
- `results/cob_v2/summary_by_domain.csv`
- `results/cob_v2/summary_by_seed.csv`
- `results/cob_v2/experiment_manifest.json`
- `results/cob_v2/episodes/*.jsonl`
- `results/cob_v2/scores/*.json`

Local Ollama agent artifacts:

- `results/cob_v2_ollama/summary.csv`
- `results/cob_v2_ollama/experiment_manifest.json`
- `results/cob_v2_ollama/predictions/*.jsonl`
- `results/cob_v2_ollama/predictions/*_raw.jsonl`
- `results/cob_v2_ollama/scores/*.json`
- `results/cob_v2_ollama_model_cards/*.txt`

Additional release artifacts:

- `results/cob_v2_ollama_ablation/*/summary.csv`
- `results/cob_v2_seed01_stratified/episodes/*.jsonl`
- `results/cob_v2_ollama_ablation_extra/*/summary.csv`
- `results/cob_v2_contamination_all_domains/*/delta_summary.csv`
- `results/cob_v2_contamination_no_topology/*/delta_summary.csv`
- `results/cob_v2_prompt_sweep/*/summary.csv`
- `results/cob_v2_hidden_seed/summary.csv`
- `results/cob_v2_hidden_seed_ollama/summary.csv`
- `results/cob_v2_analysis/*.csv`
- `results/cob_v2_analysis/*.tex`
- `results/cob_v2_analysis/*.pdf`
- `external_validation/outputs/**/summary.csv`
- `external_validation/outputs/**/scores/*.json`

## Observed Smoke Results

The CLI smoke run preserved the expected ordering: oracle above threshold.
The 5-episode scaffold smoke run produced the following composite means:

| Baseline | Composite |
| --- | ---: |
| oracle | 0.982075 |
| threshold | 0.767774 |
| noop | 0.149753 |

The included full native scaffold summary in `results/cob_v2/summary.csv`
orders the baselines as oracle, topology-RCA, threshold, random, and no-op.
The included full local-agent summary in `results/cob_v2_ollama/summary.csv`
identifies `gemma4:26b` as the strongest recorded local-agent model.

## Known Limitations

- Raw RCAEval and LBNL FDD datasets are not redistributed; regenerate them
  from public sources using `external_validation/README.md`.
- Large processed external episode JSONL files are intentionally omitted.
- Local-agent reruns require Ollama, model availability, and adequate hardware.
- The native benchmark is a deterministic interventional-replay scaffold, not
  an industrial simulator.
- The public repository URL can only be resolved after the GitHub repository is
  created or pushed.

## Reproduction Notes

Use `README.md` for executable setup, test, native scaffold, Ollama, analysis,
and external-validation commands.
