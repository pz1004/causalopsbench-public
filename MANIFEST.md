# CausalOpsBench Audit Manifest

This manifest indexes the artifacts included in the standalone public
reproducibility release.

## Release Identity

- Public repository URL: `https://github.com/pz1004/causalopsbench-public`
- Release name: CausalOpsBench public reproducibility release
- Release date: 2026-05-20
- Commit SHA: record the final Git commit in the GitHub release metadata.
- Scope: standalone code, tests, documentation, compact fixtures, generated
  benchmark artifacts, local-agent summaries/predictions/scores, and external
  portability outputs. Article source files are outside this repository.

## Native Scaffold Run

- Runner: `scripts/run_experiments.py`
- Manifest: `results/cob_v2/experiment_manifest.json`
- Seeds: `0 1 2 3 4`
- Episodes: 500 per seed, 2500 total
- Duration: 60 steps
- Domains: all five public domains
- Baselines: `noop random threshold topology_rca oracle`
- Degenerate replay episodes: `N_deg=0`

## Local Ollama Agent Run

- Runner: `scripts/run_foundation_agent_experiments.py`
- Manifest: `results/cob_v2_ollama/experiment_manifest.json`
- Ollama version: `0.23.4`
- Hardware: NVIDIA Quadro RTX 8000, 49152 MiB VRAM
- Configuration: `timeout_s=120.0`, `num_ctx=8192`, `temperature=0`, `keep_alive=5m`, `think=false`
- Predictions: `results/cob_v2_ollama/predictions/`
- Scores: `results/cob_v2_ollama/scores/`
- Model-card snapshots: `results/cob_v2_ollama_model_cards/`

## Ollama Model Digests

| Model tag | Ollama ID | Full blob SHA256 |
| --- | --- | --- |
| `gemma4:26b` | `5571076f3d70` | `7121486771cbfe218851513210c40b35dbdee93ab1ef43fe36283c883980f0df` |
| `gemma4:e2b` | `7fbdbf8f5e45` | `4e30e2665218745ef463f722c0bf86be0cab6ee676320f1cfadf91e989107448` |
| `gemma4:e4b` | `c6eb396dbd59` | `4c27e0f5b5adf02ac956c7322bd2ee7636fe3f45a8512c9aba5385242cb6e09a` |
| `granite4.1:3b` | `6fd349357287` | `662b0626cd58f443baea23559b469df6576a81d349649c59413b36a9fb32eb29` |
| `granite4.1:8b` | `444af1c4b2fe` | `ed902ac9eb6adce5a90c6a08c8ea201b50e23fdc5976d1cd0362006afac5309e` |
| `qwen3.5:4b` | `2a654d98e6fb` | `81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490` |
| `qwen3.5:9b` | `6488c96fa5fa` | `dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c` |

## Additional Result Manifests

- Best-model view ablations: `results/cob_v2_ollama_ablation/*/experiment_manifest.json`
- Cross-model ablation episode subsample: `results/cob_v2_seed01_stratified/episodes/*.jsonl`
- Cross-model 100-episode ablations: `results/cob_v2_ollama_ablation_extra/*/experiment_manifest.json`
- Five-domain neutralized-identifier stress: `results/cob_v2_contamination_all_domains/*/experiment_manifest.json`
- Joint neutralized/no-topology stress: `results/cob_v2_contamination_no_topology/*/experiment_manifest.json`
- Prompt sensitivity: `results/cob_v2_prompt_sweep/*/experiment_manifest.json`
- Held-out scaffold seed: `results/cob_v2_hidden_seed/experiment_manifest.json`
- Held-out Ollama seed: `results/cob_v2_hidden_seed_ollama/experiment_manifest.json`
- External public-trace agents: `external_validation/outputs/agents_20260519T150840Z/experiment_manifest.json`

## Rendered Analysis Outputs

- Ollama model table and full digest CSV: `results/cob_v2_analysis/ollama_model_table.tex`, `results/cob_v2_analysis/ollama_model_table.csv`
- External portability table: `results/cob_v2_analysis/external_portability_table.tex`
- Hidden-seed table: `results/cob_v2_analysis/hidden_seed_table.tex`
- Prompt-sensitivity table: `results/cob_v2_analysis/prompt_sensitivity_table.tex`
- Contamination-stress tables: `results/cob_v2_analysis/contamination_stress_table.tex`, `results/cob_v2_analysis/contamination_no_topology_table.tex`
- Consolidated submetric table: `results/cob_v2_analysis/submetric_master_table.tex`
- Scaffold, local-agent, and domain summaries: `results/cob_v2_analysis/table1_scaffold_stats.tex`, `results/cob_v2_analysis/table2_local_agent_stats.tex`, `results/cob_v2_analysis/table3_domain_stats.tex`
- Figures: `results/cob_v2_analysis/*.pdf`

## Optional Full-Scale Cross-Model Ablation Command

```bash
for ablation in no-evidence no-topology no-manuals; do
  python scripts/run_foundation_agent_experiments.py \
    --episodes-dir results/cob_v2/episodes \
    --output-dir "results/cob_v2_ollama_ablation_extra_full/${ablation}" \
    --models ollama:gemma4:e4b ollama:granite4.1:8b \
    --policy react-json \
    --view-ablation "$ablation" \
    --max-steps 4 \
    --temperature 0 \
    --num-ctx 8192 \
    --keep-alive 5m \
    --think false \
    --seed 0 \
    --timeout-s 120
done
```

After running the command, regenerate analysis artifacts with
`--extra-ablation-root results/cob_v2_ollama_ablation_extra_full`.
