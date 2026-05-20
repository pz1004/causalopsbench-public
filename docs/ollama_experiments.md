# Ollama Experiment Protocol for Public Release

This document records the local open-weight SLM/LLM experiment protocol used for the CausalOpsBench public reproducibility release. It is intentionally narrative: executable reproduction commands are kept only in the repository README.

## Purpose

The local-agent track evaluates whether resource-constrained open-weight models can use telemetry summaries, evidence spans, topology, manuals, and action metadata to produce valid incident-response predictions. The track avoids paid proprietary API dependence and supports audit through raw model outputs, score files, model-card snapshots, and run manifests.

## Model Set

The completed release track uses seven Ollama model tags:

- `gemma4:e2b`
- `qwen3.5:4b`
- `granite4.1:3b`
- `gemma4:e4b`
- `qwen3.5:9b`
- `granite4.1:8b`
- `gemma4:26b`

Reports should identify `gemma4:26b` as the strongest completed local-agent result, `qwen3.5:9b` and `qwen3.5:4b` as compact high-performing general-reasoning baselines, and `granite4.1:8b` as the structured-output family comparison. Use `local open-weight` or `local open-weight SLM/LLM` for these baselines rather than universally calling them open-source, because license openness varies by model.

## Experiment Phases

Phase 0 verifies the Python environment, the runner help output, the local Ollama service, and GPU availability. The recorded release-track hardware is a Quadro RTX 8000 with approximately 49152 MiB total VRAM.

Phase 1 regenerates the native scaffold artifacts under `results/cob_v2`. Included analysis tables should not be changed unless this phase is deliberately rerun and the generated summaries are rechecked.

Phase 2 pulls the seven model tags and saves one Ollama model-card snapshot plus one modelfile snapshot per tag under `results/cob_v2_ollama_model_cards`. If a tag is unavailable, record the failure rather than silently substituting another model.

Phase 3 performs a small smoke run under `results/cob_v2_ollama_smoke`. The smoke run should produce score files and parse-status counts for all seven models before the full run is trusted.

Phase 4 performs the full local-agent release run under `results/cob_v2_ollama`. The completed manifest records 2,500 episodes per model, `timeout_s=120.0`, `num_ctx=8192`, `temperature=0`, `keep_alive=5m`, and `think=false`.

Phase 5 performs the best-model ReAct component ablations under `results/cob_v2_ollama_ablation/{no-evidence,no-topology,no-manuals}` using `ollama:gemma4:26b`.

Reviewer-response cross-model view ablations for `gemma4:e4b` and `granite4.1:8b` are reported on a 100-episode stratified subsample. If a full-scale replacement is required, run:

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

Then regenerate analysis tables with `--extra-ablation-root results/cob_v2_ollama_ablation_extra_full`.

Phase 6 performs reviewer-response stress experiments: five-domain renamed/paraphrased contamination under `results/cob_v2_contamination_all_domains`, prompt sensitivity under `results/cob_v2_prompt_sweep`, and joint neutralized-identifier plus no-topology stress under `results/cob_v2_contamination_no_topology`.

Phase 7 performs the held-out topology seed audit under `results/cob_v2_hidden_seed` and `results/cob_v2_hidden_seed_ollama` using seed `20260520`, topology variant `heldout_v1`, and `ollama:gemma4:26b`.

Phase 8 audits score summaries, raw prediction counts, and JSON parse statuses. The completed main run produced direct valid JSON for all 2,500 episodes for each of the seven models.

Phase 9 records the run environment. The completed artifact set reports Ollama 0.23.4, Quadro RTX 8000 with 49152 MiB memory, `num_ctx=8192`, `temperature=0`, `keep_alive=5m`, and `think=false`.

Phase 10 converts the completed run into release-ready tables, statistical artifacts, and figures under `results/cob_v2_analysis`.

## Expected Artifacts

The completed local-agent run should preserve:

- `results/cob_v2_ollama/summary.csv`
- `results/cob_v2_ollama/experiment_manifest.json`
- `results/cob_v2_ollama/predictions/*.jsonl`
- `results/cob_v2_ollama/predictions/*_raw.jsonl`
- `results/cob_v2_ollama/scores/*.json`
- `results/cob_v2_ollama_ablation/*/summary.csv`
- `results/cob_v2_contamination_all_domains/*/delta_summary.csv`
- `results/cob_v2_contamination_no_topology/*/delta_summary.csv`
- `results/cob_v2_prompt_sweep/*/summary.csv`
- `results/cob_v2_hidden_seed/summary.csv`
- `results/cob_v2_hidden_seed_ollama/summary.csv`
- `results/cob_v2_analysis/*.tex`
- `results/cob_v2_analysis/*.csv`
- `results/cob_v2_analysis/*.pdf`
- `results/cob_v2_ollama_model_cards/*.txt`

## Interpretation Guidance

The completed local-agent results should be interpreted as a local open-weight agent baseline, not as industrial validation. `gemma4:26b` approaches the oracle upper bound but remains below it. The Qwen models outperform the threshold baseline on composite score. The Granite models show strong root-cause localization but weak detection and intervention behavior. The smaller Gemma and Granite models expose safety and alarm failures, supporting the benchmark's discriminability.

When reporting local-agent results, include the exact model tags, Ollama version, GPU, context length, temperature, keep-alive setting, thinking setting, prompt or prompt hash, raw prediction artifacts, and parse-failure counts. Do not manually repair predictions before scoring; parser failures or malformed outputs are benchmark evidence.

## Reproduction Commands

All executable commands for environment checks, model pulls, scaffold runs, Ollama smoke runs, full local-agent runs, and artifact checks are maintained in `README.md`. The public implementation and command entrypoint are disclosed at https://github.com/pz1004/causalopsbench-public.

## Sources

- Ollama API documentation: https://github.com/ollama/ollama/blob/main/docs/api.md
- Qwen3.5 Ollama library page: https://ollama.com/library/qwen3.5
- Gemma4 Ollama library page: https://ollama.com/library/gemma4
- Granite4.1 Ollama library page: https://ollama.com/library/granite4.1
