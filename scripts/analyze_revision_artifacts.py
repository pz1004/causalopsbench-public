#!/usr/bin/env python3
"""Generate reviewer-response tables and figures from experiment artifacts."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import re
import sys
import textwrap
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causalopsbench._io import write_csv_rows, write_json_document
from causalopsbench.baselines import get_baseline
from causalopsbench.domains import domain_names, get_domain
from causalopsbench.metrics import INTERVENTION_EPSILON, is_degenerate_intervention
from causalopsbench.schemas import load_episodes_jsonl, load_predictions_jsonl

METRICS = [
    "composite",
    "detection",
    "root_cause",
    "intervention",
    "evidence",
    "calibration",
    "efficiency",
]

WEIGHT_SETS = {
    "v1_default": {
        "detection": 0.15,
        "root_cause": 0.20,
        "intervention": 0.30,
        "evidence": 0.15,
        "calibration": 0.10,
        "efficiency": 0.10,
    },
    "diagnosis_heavy": {
        "detection": 0.15,
        "root_cause": 0.35,
        "intervention": 0.20,
        "evidence": 0.15,
        "calibration": 0.10,
        "efficiency": 0.05,
    },
    "evidence_heavy": {
        "detection": 0.15,
        "root_cause": 0.20,
        "intervention": 0.20,
        "evidence": 0.30,
        "calibration": 0.10,
        "efficiency": 0.05,
    },
    "efficiency_light": {
        "detection": 0.18,
        "root_cause": 0.22,
        "intervention": 0.35,
        "evidence": 0.18,
        "calibration": 0.07,
        "efficiency": 0.00,
    },
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate reviewer-response statistical artifacts."
    )
    parser.add_argument("--scaffold-dir", required=True)
    parser.add_argument("--ollama-dir", required=True)
    parser.add_argument("--ablation-root", required=True)
    parser.add_argument("--extra-ablation-root")
    parser.add_argument("--contamination-dir")
    parser.add_argument("--joint-contamination-dir")
    parser.add_argument("--prompt-root")
    parser.add_argument("--hidden-scaffold-dir")
    parser.add_argument("--hidden-ollama-dir")
    parser.add_argument("--model-card-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--random-weight-samples", type=int, default=1000)
    parser.add_argument("--random-weight-seed", type=int, default=20260516)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = _load_episode_map(Path(args.scaffold_dir) / "episodes")
    scaffold_records = _load_scaffold_records(Path(args.scaffold_dir), episodes)
    ollama_records = _load_agent_records(Path(args.ollama_dir), episodes, "main")
    ablation_records = _load_ablation_records(Path(args.ablation_root), episodes)
    extra_ablation_records = (
        _load_ablation_records(Path(args.extra_ablation_root), episodes)
        if args.extra_ablation_root
        else []
    )

    rng = random.Random(args.seed)
    _write_leaderboard_stats(
        output_dir / "table1_scaffold_stats.tex",
        output_dir / "table1_scaffold_stats.csv",
        scaffold_records,
        "baseline",
        args.bootstrap,
        rng,
    )
    _write_leaderboard_stats(
        output_dir / "table2_local_agent_stats.tex",
        output_dir / "table2_local_agent_stats.csv",
        ollama_records,
        "baseline",
        args.bootstrap,
        rng,
    )
    _write_domain_stats(
        output_dir / "table3_domain_stats.tex",
        output_dir / "table3_domain_stats.csv",
        scaffold_records,
        args.bootstrap,
        rng,
    )
    _write_pairwise(
        output_dir / "pairwise_scaffold_composite.csv",
        scaffold_records,
        "baseline",
        args.bootstrap,
        rng,
    )
    _write_effect_size_summary(
        output_dir / "effect_size_summary.tex",
        output_dir / "effect_size_summary.csv",
        scaffold_records,
        args.bootstrap,
        rng,
    )
    _write_pairwise(
        output_dir / "pairwise_local_agent_composite.csv",
        ollama_records,
        "baseline",
        args.bootstrap,
        rng,
    )
    _write_domain_pairwise(
        output_dir / "pairwise_domain_composite.csv",
        scaffold_records,
        args.bootstrap,
        rng,
    )
    _write_sensitivity(
        output_dir / "sensitivity_table.tex",
        output_dir / "sensitivity_table.csv",
        scaffold_records,
    )
    sweep_row = _write_random_weight_sweep(
        output_dir / "random_weight_sweep_table.tex",
        output_dir / "random_weight_sweep.csv",
        scaffold_records,
        args.random_weight_samples,
        random.Random(args.random_weight_seed),
    )
    _write_ablation_table(
        output_dir / "react_ablation_table.tex",
        output_dir / "react_ablation_table.csv",
        ollama_records,
        ablation_records,
        args.bootstrap,
        rng,
    )
    _write_expanded_ablation_table(
        output_dir / "react_ablation_expanded_table.tex",
        output_dir / "react_ablation_expanded_table.csv",
        ollama_records,
        ablation_records,
        extra_ablation_records,
        args.bootstrap,
        rng,
    )
    if args.contamination_dir:
        _write_contamination_table(
            output_dir / "contamination_stress_table.tex",
            output_dir / "contamination_stress_table.csv",
            Path(args.contamination_dir),
        )
        _write_contamination_delta_ci(
            output_dir / "contamination_delta_ci.tex",
            output_dir / "contamination_delta_ci.csv",
            Path(args.contamination_dir),
            args.bootstrap,
            rng,
        )
    joint_contamination_dir = (
        Path(args.joint_contamination_dir)
        if args.joint_contamination_dir
        else output_dir.parent / "cob_v2_contamination_no_topology"
    )
    if joint_contamination_dir.exists():
        _write_contamination_table(
            output_dir / "contamination_no_topology_table.tex",
            output_dir / "contamination_no_topology_table.csv",
            joint_contamination_dir,
        )
    prompt_root = Path(args.prompt_root) if args.prompt_root else output_dir.parent / "cob_v2_prompt_sweep"
    if prompt_root.exists():
        prompt_records = _load_prompt_records(prompt_root, episodes)
        _write_prompt_sweep_table(
            output_dir / "prompt_sensitivity_table.tex",
            output_dir / "prompt_sensitivity_table.csv",
            prompt_records,
            args.bootstrap,
            rng,
        )
    _write_model_table(
        output_dir / "ollama_model_table.tex",
        output_dir / "ollama_model_table.csv",
        Path(args.model_card_dir),
    )
    _write_submetric_master_table(
        output_dir / "submetric_master_table.tex",
        output_dir / "submetric_master_table.csv",
        scaffold_records,
        ollama_records,
    )
    if args.hidden_scaffold_dir and args.hidden_ollama_dir:
        hidden_episodes = _load_episode_map(Path(args.hidden_scaffold_dir) / "episodes")
        hidden_records = _load_agent_records(Path(args.hidden_ollama_dir), hidden_episodes, "hidden")
        _write_hidden_seed_table(
            output_dir / "hidden_seed_table.tex",
            output_dir / "hidden_seed_table.csv",
            hidden_records,
            Path(args.hidden_ollama_dir),
            args.bootstrap,
            rng,
        )
    _write_figures(output_dir, Path(args.scaffold_dir), Path(args.ollama_dir), episodes)
    _write_analysis_manifest(
        output_dir / "experiment_manifest.json",
        episodes,
        scaffold_records,
        args,
        sweep_row,
    )

    print(json.dumps({"output_dir": str(output_dir), "records": len(scaffold_records) + len(ollama_records)}))
    return 0


def _load_episode_map(episodes_dir: Path) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    if not episodes_dir.exists():
        return by_id
    for path in sorted(episodes_dir.glob("*.jsonl")):
        for episode in load_episodes_jsonl(path):
            by_id[episode.episode_id] = {
                "episode": episode,
                "domain": episode.domain,
                "seed": episode.metadata.get("seed"),
                "fault_start": episode.ground_truth.fault.start_time,
                "oracle_action_ids": set(episode.ground_truth.oracle_action_ids),
            }
    return by_id


def _load_scaffold_records(scaffold_dir: Path, episodes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((scaffold_dir / "scores").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        baseline = str(payload.get("baseline") or path.stem.split("_seed_", 1)[0])
        seed = payload.get("seed")
        for score in payload.get("scores", []):
            record = dict(score)
            record["baseline"] = baseline
            record["seed"] = record.get("seed", seed)
            record["domain"] = record.get("domain") or episodes.get(record["episode_id"], {}).get("domain", "")
            records.append(record)
    return records


def _load_agent_records(agent_dir: Path, episodes: dict[str, dict[str, Any]], run_label: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    scores_dir = agent_dir / "scores"
    if not scores_dir.exists():
        return records
    for path in sorted(scores_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        baseline = _short_model_name(str(payload.get("model_spec") or payload.get("baseline") or path.stem))
        for score in payload.get("scores", []):
            episode_meta = episodes.get(score["episode_id"], {})
            record = dict(score)
            record["baseline"] = baseline
            record["model_spec"] = payload.get("model_spec", "")
            record["run_label"] = run_label
            record["seed"] = episode_meta.get("seed")
            record["domain"] = episode_meta.get("domain", "")
            records.append(record)
    return records


def _load_ablation_records(ablation_root: Path, episodes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not ablation_root.exists():
        return records
    for child in sorted(path for path in ablation_root.iterdir() if path.is_dir()):
        child_records = _load_agent_records(child, episodes, child.name)
        for record in child_records:
            record["ablation"] = child.name
        records.extend(child_records)
    return records


def _load_prompt_records(prompt_root: Path, episodes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not prompt_root.exists():
        return records
    for child in sorted(path for path in prompt_root.iterdir() if path.is_dir()):
        child_records = _load_agent_records(child, episodes, child.name)
        for record in child_records:
            record["prompt_style"] = child.name
        records.extend(child_records)
    return records


def _write_leaderboard_stats(
    tex_path: Path,
    csv_path: Path,
    records: list[dict[str, Any]],
    group_key: str,
    bootstrap: int,
    rng: random.Random,
) -> None:
    rows: list[dict[str, Any]] = []
    for name, group in sorted(_groups(records, group_key).items(), key=lambda item: _mean(item[1], "composite"), reverse=True):
        row = {
            group_key: name,
            "count": len(group),
            "safety_violations": sum(int(record.get("safety_violations", 0)) for record in group),
        }
        seed_means = [
            _mean(seed_group, "composite")
            for seed_group in _groups(group, "seed").values()
            if seed_group
        ]
        row["composite_seed_sd"] = _stdev(seed_means)
        for metric in METRICS:
            values = [float(record[metric]) for record in group]
            mean, low, high = _bootstrap_ci(values, bootstrap, rng)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)

    _write_csv(csv_path, rows)
    if not rows:
        tex_path.write_text("% No rows generated.\n", encoding="utf-8")
        return

    lines = [
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"System & Composite (95\% CI) & Seed SD & Detect. & Root & Interv. & Evidence & Calib. & Eff. & Safety \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _tex_escape(str(row[group_key])),
                    _fmt_ci(row["composite_mean"], row["composite_ci_low"], row["composite_ci_high"]),
                    f"{row['composite_seed_sd']:.3f}",
                    _fmt_ci(row["detection_mean"], row["detection_ci_low"], row["detection_ci_high"]),
                    _fmt_ci(row["root_cause_mean"], row["root_cause_ci_low"], row["root_cause_ci_high"]),
                    _fmt_ci(row["intervention_mean"], row["intervention_ci_low"], row["intervention_ci_high"]),
                    _fmt_ci(row["evidence_mean"], row["evidence_ci_low"], row["evidence_ci_high"]),
                    _fmt_ci(row["calibration_mean"], row["calibration_ci_low"], row["calibration_ci_high"]),
                    _fmt_ci(row["efficiency_mean"], row["efficiency_ci_low"], row["efficiency_ci_high"]),
                    str(int(row["safety_violations"])),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")


def _write_domain_stats(
    tex_path: Path,
    csv_path: Path,
    records: list[dict[str, Any]],
    bootstrap: int,
    rng: random.Random,
) -> None:
    domains = sorted({record.get("domain", "") for record in records if record.get("domain")})
    baselines = sorted(
        {record["baseline"] for record in records},
        key=lambda baseline: _mean([r for r in records if r["baseline"] == baseline], "composite"),
        reverse=True,
    )
    rows: list[dict[str, Any]] = []
    by_key = {(row["baseline"], row.get("domain", "")): [] for row in records}
    for record in records:
        by_key.setdefault((record["baseline"], record.get("domain", "")), []).append(record)
    for baseline in baselines:
        for domain in domains:
            group = by_key.get((baseline, domain), [])
            values = [float(record["composite"]) for record in group]
            mean, low, high = _bootstrap_ci(values, bootstrap, rng)
            seed_sd = _stdev([
                _mean(seed_group, "composite")
                for seed_group in _groups(group, "seed").values()
                if seed_group
            ])
            rows.append(
                {
                    "baseline": baseline,
                    "domain": domain,
                    "count": len(group),
                    "composite_mean": mean,
                    "composite_ci_low": low,
                    "composite_ci_high": high,
                    "composite_seed_sd": seed_sd,
                }
            )
    _write_csv(csv_path, rows)
    lines = [
        r"\begin{tabular}{l" + "r" * len(domains) + "}",
        r"\toprule",
        "Baseline & " + " & ".join(_tex_escape(domain) for domain in domains) + r" \\",
        r"\midrule",
    ]
    lookup = {(row["baseline"], row["domain"]): row for row in rows}
    for baseline in baselines:
        values = []
        for domain in domains:
            row = lookup.get((baseline, domain))
            values.append(
                _fmt_ci(row["composite_mean"], row["composite_ci_low"], row["composite_ci_high"])
                if row and row["count"]
                else "--"
            )
        lines.append(_tex_escape(baseline) + " & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")


def _write_pairwise(
    path: Path,
    records: list[dict[str, Any]],
    group_key: str,
    bootstrap: int,
    rng: random.Random,
) -> None:
    groups = _groups(records, group_key)
    rows: list[dict[str, Any]] = []
    names = sorted(groups, key=lambda name: _mean(groups[name], "composite"), reverse=True)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            paired = _paired_values(groups[left], groups[right], "episode_id", "composite")
            if not paired:
                continue
            left_values = [item[0] for item in paired]
            right_values = [item[1] for item in paired]
            delta = sum(a - b for a, b in paired) / len(paired)
            p_value = _paired_bootstrap_p(left_values, right_values, bootstrap, rng)
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "n": len(paired),
                    "mean_delta": round(delta, 6),
                    "paired_bootstrap_p": round(p_value, 6),
                }
            )
    _write_csv(path, rows)


def _write_effect_size_summary(
    tex_path: Path,
    csv_path: Path,
    records: list[dict[str, Any]],
    bootstrap: int,
    rng: random.Random,
) -> None:
    groups = _groups(records, "baseline")
    names = sorted(groups, key=lambda name: _mean(groups[name], "composite"), reverse=True)
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            paired = _paired_values(groups[left], groups[right], "episode_id", "composite")
            if not paired:
                continue
            diffs = [a - b for a, b in paired]
            mean_delta, low, high = _bootstrap_ci(diffs, bootstrap, rng)
            sd = _population_stdev(diffs)
            dz = mean_delta / sd if sd else math.inf
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "n": len(paired),
                    "mean_delta": mean_delta,
                    "delta_ci_low": low,
                    "delta_ci_high": high,
                    "paired_cohens_dz": round(dz, 6) if math.isfinite(dz) else "inf",
                    "paired_delta_sd": round(sd, 6),
                }
            )
    _write_csv(csv_path, rows)
    finite = [
        abs(float(row["paired_cohens_dz"]))
        for row in rows
        if row["paired_cohens_dz"] != "inf" and abs(float(row["paired_cohens_dz"])) <= 50
    ]
    if not rows:
        tex_path.write_text("% Effect-size artifacts were not found.\n", encoding="utf-8")
        return
    min_delta = min(float(row["mean_delta"]) for row in rows)
    max_delta = max(float(row["mean_delta"]) for row in rows)
    if finite:
        summary = f"Pairwise gaps span {min_delta:.3f}--{max_delta:.3f}; finite paired Cohen's $d_z$ spans {min(finite):.2f}--{max(finite):.2f}."
    else:
        summary = f"Pairwise gaps span {min_delta:.3f}--{max_delta:.3f}; all paired effect sizes are deterministic."
    tex_path.write_text(summary + "\n", encoding="utf-8")


def _write_domain_pairwise(
    path: Path,
    records: list[dict[str, Any]],
    bootstrap: int,
    rng: random.Random,
) -> None:
    rows: list[dict[str, Any]] = []
    for domain, domain_records in sorted(_groups(records, "domain").items()):
        groups = _groups(domain_records, "baseline")
        names = sorted(groups, key=lambda name: _mean(groups[name], "composite"), reverse=True)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                paired = _paired_values(groups[left], groups[right], "episode_id", "composite")
                if not paired:
                    continue
                left_values = [item[0] for item in paired]
                right_values = [item[1] for item in paired]
                delta = sum(a - b for a, b in paired) / len(paired)
                rows.append(
                    {
                        "domain": domain,
                        "left": left,
                        "right": right,
                        "n": len(paired),
                        "mean_delta": round(delta, 6),
                        "paired_bootstrap_p": round(_paired_bootstrap_p(left_values, right_values, bootstrap, rng), 6),
                    }
                )
    _write_csv(path, rows)


def _write_sensitivity(tex_path: Path, csv_path: Path, records: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for weighting, weights in WEIGHT_SETS.items():
        scored: list[tuple[str, float]] = []
        for baseline, group in _groups(records, "baseline").items():
            values = [_weighted_score(record, weights) for record in group]
            scored.append((baseline, sum(values) / len(values) if values else 0.0))
        scored.sort(key=lambda item: item[1], reverse=True)
        rows.append(
            {
                "weighting": weighting,
                "ranking": " > ".join(name for name, _ in scored),
                **{f"{name}_composite": round(value, 6) for name, value in scored},
            }
        )
    _write_csv(csv_path, rows)
    baselines = sorted({record["baseline"] for record in records})
    lines = [
        r"\begin{tabular}{l" + "r" * len(baselines) + "l}",
        r"\toprule",
        "Weighting & " + " & ".join(_tex_escape(baseline) for baseline in baselines) + r" & Ranking \\",
        r"\midrule",
    ]
    for row in rows:
        values = [f"{float(row.get(f'{baseline}_composite', 0.0)):.3f}" for baseline in baselines]
        lines.append(
            _tex_escape(row["weighting"])
            + " & "
            + " & ".join(values)
            + " & "
            + _tex_escape(row["ranking"])
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")


def _write_random_weight_sweep(
    tex_path: Path,
    csv_path: Path,
    records: list[dict[str, Any]],
    samples: int,
    rng: random.Random,
) -> dict[str, Any]:
    target = ["oracle", "topology_rca", "threshold", "random", "noop"]
    metrics = ["detection", "root_cause", "intervention", "evidence", "calibration", "efficiency"]
    groups = _groups(records, "baseline")
    preserved = 0
    for _ in range(max(0, samples)):
        draws = [rng.expovariate(1.0) for _ in metrics]
        total = sum(draws) or 1.0
        weights = {metric: draw / total for metric, draw in zip(metrics, draws)}
        scored = []
        for baseline, group in groups.items():
            values = [_weighted_score(record, weights) for record in group]
            scored.append((baseline, sum(values) / len(values) if values else 0.0))
        ranking = [name for name, _ in sorted(scored, key=lambda item: item[1], reverse=True)]
        if ranking[: len(target)] == target:
            preserved += 1
    row = {
        "samples": samples,
        "preserved": preserved,
        "fraction": round(preserved / samples, 6) if samples else 0.0,
        "target_ranking": " > ".join(target),
        "weight_distribution": "Dirichlet(1,1,1,1,1,1) via normalized exponential draws",
    }
    _write_csv(csv_path, [row])
    lines = [
        r"\begin{tabular}{rrrl}",
        r"\toprule",
        r"Samples & Preserved & Fraction & Target ranking \\",
        r"\midrule",
        f"{samples} & {preserved} & {row['fraction']:.3f} & {_tex_escape(row['target_ranking'])} " + r"\\",
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    return row


def _write_ablation_table(
    tex_path: Path,
    csv_path: Path,
    main_records: list[dict[str, Any]],
    ablation_records: list[dict[str, Any]],
    bootstrap: int,
    rng: random.Random,
) -> None:
    strongest = "gemma4:26b"
    main_group = [record for record in main_records if record["baseline"] == strongest]
    rows: list[dict[str, Any]] = []
    if main_group:
        rows.append(_ablation_row("full-view", main_group, bootstrap, rng))
    for name, group in sorted(_groups(ablation_records, "ablation").items()):
        rows.append(_ablation_row(name, group, bootstrap, rng))
    _write_csv(csv_path, rows)
    if not rows:
        tex_path.write_text("% ReAct ablations were not found.\n", encoding="utf-8")
        return
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"View & Composite (95\% CI) & Detect. & Root & Interv. & Evidence & Calib. & Safety \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _tex_escape(row["view"]),
                    _fmt_ci(row["composite_mean"], row["composite_ci_low"], row["composite_ci_high"]),
                    f"{row['detection_mean']:.3f}",
                    f"{row['root_cause_mean']:.3f}",
                    f"{row['intervention_mean']:.3f}",
                    f"{row['evidence_mean']:.3f}",
                    f"{row['calibration_mean']:.3f}",
                    str(int(row["safety_violations"])),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")


def _write_expanded_ablation_table(
    tex_path: Path,
    csv_path: Path,
    main_records: list[dict[str, Any]],
    ablation_records: list[dict[str, Any]],
    extra_ablation_records: list[dict[str, Any]],
    bootstrap: int,
    rng: random.Random,
) -> None:
    rows: list[dict[str, Any]] = []
    for model in ["gemma4:26b", "gemma4:e4b", "granite4.1:8b"]:
        model_main = [record for record in main_records if record["baseline"] == model]
        if model == "gemma4:26b":
            if model_main:
                row = _ablation_row("full-view", model_main, bootstrap, rng)
                row["model"] = model
                rows.append(row)
            for view, group in sorted(_groups(ablation_records, "ablation").items()):
                model_group = [record for record in group if record["baseline"] == model]
                if model_group:
                    row = _ablation_row(view, model_group, bootstrap, rng)
                    row["model"] = model
                    rows.append(row)
        else:
            extra_model_records = [
                record for record in extra_ablation_records if record["baseline"] == model
            ]
            full_group = [
                record
                for record in extra_model_records
                if record.get("ablation") == "full-view"
            ]
            if not full_group and extra_model_records:
                episode_ids = {record["episode_id"] for record in extra_model_records}
                full_group = [
                    record
                    for record in main_records
                    if record["baseline"] == model and record["episode_id"] in episode_ids
                ]
            if full_group:
                row = _ablation_row("full-view", full_group, bootstrap, rng)
                row["model"] = model
                rows.append(row)
            for view, group in sorted(_groups(extra_ablation_records, "ablation").items()):
                if view == "full-view":
                    continue
                model_group = [record for record in group if record["baseline"] == model]
                if model_group:
                    row = _ablation_row(view, model_group, bootstrap, rng)
                    row["model"] = model
                    rows.append(row)

    view_order = {"full-view": 0, "no-evidence": 1, "no-topology": 2, "no-manuals": 3}
    model_order = {"gemma4:26b": 0, "gemma4:e4b": 1, "granite4.1:8b": 2}
    rows.sort(key=lambda row: (model_order.get(row["model"], 99), view_order.get(row["view"], 99)))
    _write_csv(csv_path, rows)
    if not rows:
        tex_path.write_text("% Expanded ReAct ablations were not found.\n", encoding="utf-8")
        return
    lines = [
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Model & View & $N$ & Composite (95\% CI) & Detect. & Root & Interv. & Evidence & Safety \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _tex_escape(row["model"]),
                    _tex_escape(row["view"]),
                    str(int(row["count"])),
                    _fmt_ci(row["composite_mean"], row["composite_ci_low"], row["composite_ci_high"]),
                    f"{row['detection_mean']:.3f}",
                    f"{row['root_cause_mean']:.3f}",
                    f"{row['intervention_mean']:.3f}",
                    f"{row['evidence_mean']:.3f}",
                    str(int(row["safety_violations"])),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")


def _ablation_row(view: str, group: list[dict[str, Any]], bootstrap: int, rng: random.Random) -> dict[str, Any]:
    mean, low, high = _bootstrap_ci([float(record["composite"]) for record in group], bootstrap, rng)
    return {
        "view": view,
        "count": len(group),
        "composite_mean": mean,
        "composite_ci_low": low,
        "composite_ci_high": high,
        "detection_mean": _mean(group, "detection"),
        "root_cause_mean": _mean(group, "root_cause"),
        "intervention_mean": _mean(group, "intervention"),
        "evidence_mean": _mean(group, "evidence"),
        "calibration_mean": _mean(group, "calibration"),
        "safety_violations": sum(int(record.get("safety_violations", 0)) for record in group),
    }


def _write_model_table(tex_path: Path, csv_path: Path, model_card_dir: Path) -> None:
    rows = _model_rows(model_card_dir)
    _write_csv(csv_path, rows)
    if not rows:
        tex_path.write_text("% Ollama model-card artifacts were not found.\n", encoding="utf-8")
        return
    lines = [
        r"\begin{tabular}{lllll}",
        r"\toprule",
        r"Model tag & Pull command & Ollama ID & Blob digest (first 12 hex) & Source \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _tex_escape(row["model"]),
                    r"\texttt{" + _tex_escape(row["pull"]) + "}",
                    r"\texttt{" + _tex_escape(row["ollama_id"]) + "}",
                    r"\texttt{" + _tex_escape(row["sha256_short"]) + "}",
                    row["source_cite"],
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")


def _write_prompt_sweep_table(
    tex_path: Path,
    csv_path: Path,
    records: list[dict[str, Any]],
    bootstrap: int,
    rng: random.Random,
) -> None:
    rows: list[dict[str, Any]] = []
    for style, group in sorted(_groups(records, "prompt_style").items()):
        if not group:
            continue
        mean, low, high = _bootstrap_ci([float(record["composite"]) for record in group], bootstrap, rng)
        rows.append(
            {
                "prompt_style": style,
                "count": len(group),
                "composite_mean": mean,
                "composite_ci_low": low,
                "composite_ci_high": high,
                "detection_mean": _mean(group, "detection"),
                "root_cause_mean": _mean(group, "root_cause"),
                "intervention_mean": _mean(group, "intervention"),
                "evidence_mean": _mean(group, "evidence"),
                "safety_violations": sum(int(record.get("safety_violations", 0)) for record in group),
            }
        )
    style_order = {"terse-json": 0, "react-verbose": 1, "evidence-first": 2, "react-json": 3}
    rows.sort(key=lambda row: style_order.get(str(row["prompt_style"]), 99))
    _write_csv(csv_path, rows)
    if not rows:
        tex_path.write_text("% Prompt-sensitivity artifacts were not found.\n", encoding="utf-8")
        return
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Prompt style & $N$ & Composite (95\% CI) & Detect. & Root & Interv. & Evidence & Safety \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _tex_escape(str(row["prompt_style"])),
                    str(int(row["count"])),
                    _fmt_ci(row["composite_mean"], row["composite_ci_low"], row["composite_ci_high"]),
                    f"{row['detection_mean']:.3f}",
                    f"{row['root_cause_mean']:.3f}",
                    f"{row['intervention_mean']:.3f}",
                    f"{row['evidence_mean']:.3f}",
                    str(int(row["safety_violations"])),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")


def _write_contamination_table(tex_path: Path, csv_path: Path, contamination_dir: Path) -> None:
    rows = _load_contamination_delta_rows(contamination_dir)
    if not rows:
        tex_path.write_text("% Contamination stress artifacts were not found.\n", encoding="utf-8")
        csv_path.write_text("", encoding="utf-8")
        return
    rows_with_aggregate = rows + _aggregate_contamination_rows(rows)
    _write_csv(csv_path, rows_with_aggregate)
    domains = ["bioprocess", "hvac", "manufacturing", "microservice", "water_grid"]
    domain_labels = {
        "bioprocess": "bioprocess",
        "hvac": "HVAC",
        "manufacturing": "manufacturing",
        "microservice": "microservice",
        "water_grid": "water-grid",
    }
    systems = sorted(
        {row["system"] for row in rows},
        key=lambda system: (0 if system == "threshold" else 1, system),
    )
    by_key = {(row["system"], row.get("domain", "")): row for row in rows_with_aggregate}
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"System & $N$/domain & "
        + " & ".join(domain_labels[domain] for domain in domains)
        + r" & Aggregate \\",
        r"\midrule",
    ]
    for system in systems:
        domain_ns = [
            int(float(by_key.get((system, domain), {}).get("n", 0) or 0))
            for domain in domains
        ]
        n_label = str(domain_ns[0]) if domain_ns and len(set(domain_ns)) == 1 else "mixed"
        values = []
        for domain in domains + ["aggregate"]:
            row = by_key.get((system, domain))
            values.append(_fmt_delta_cell(row))
        lines.append(
            " & ".join(
                [
                    _tex_escape(system),
                    n_label,
                    *values,
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")


def _write_contamination_delta_ci(
    tex_path: Path,
    csv_path: Path,
    contamination_dir: Path,
    bootstrap: int,
    rng: random.Random,
) -> None:
    rows = _contamination_score_deltas(contamination_dir, bootstrap, rng)
    _write_csv(csv_path, rows)
    if not rows:
        tex_path.write_text("% Contamination CI artifacts were not found.\n", encoding="utf-8")
        return
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"System & $N$ & $\Delta$ composite & 95\% CI & $\Delta$ root \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _tex_escape(str(row["system"])),
                    str(int(row["n"])),
                    f"{float(row['delta_composite']):+.3f}",
                    f"[{float(row['delta_ci_low']):+.3f}, {float(row['delta_ci_high']):+.3f}]",
                    f"{float(row['delta_root_cause']):+.3f}",
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")


def _contamination_score_deltas(
    contamination_dir: Path,
    bootstrap: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    by_system: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for child in sorted(path for path in contamination_dir.iterdir() if path.is_dir()):
        scores_dir = child / "scores"
        if not scores_dir.exists():
            continue
        for original_path in sorted(scores_dir.glob("original_*.json")):
            suffix = original_path.name.removeprefix("original_").removesuffix(".json")
            neutral_path = scores_dir / f"neutral-identifiers_{suffix}.json"
            if not neutral_path.exists():
                continue
            original_payload = json.loads(original_path.read_text(encoding="utf-8"))
            neutral_payload = json.loads(neutral_path.read_text(encoding="utf-8"))
            system = str(original_payload.get("system") or suffix.replace("_", ":"))
            by_system.setdefault(system, {"original": [], "neutral": []})
            by_system[system]["original"].extend(original_payload.get("scores", []))
            by_system[system]["neutral"].extend(neutral_payload.get("scores", []))
    rows: list[dict[str, Any]] = []
    for system, paired_scores in sorted(by_system.items()):
        original = paired_scores["original"]
        neutral = paired_scores["neutral"]
        count = min(len(original), len(neutral))
        if not count:
            continue
        composite_deltas = [
            float(neutral[index]["composite"]) - float(original[index]["composite"])
            for index in range(count)
        ]
        root_deltas = [
            float(neutral[index]["root_cause"]) - float(original[index]["root_cause"])
            for index in range(count)
        ]
        delta, low, high = _bootstrap_ci(composite_deltas, bootstrap, rng)
        rows.append(
            {
                "system": system,
                "n": count,
                "delta_composite": delta,
                "delta_ci_low": low,
                "delta_ci_high": high,
                "delta_root_cause": round(sum(root_deltas) / len(root_deltas), 6),
            }
        )
    return rows


def _write_submetric_master_table(
    tex_path: Path,
    csv_path: Path,
    scaffold_records: list[dict[str, Any]],
    ollama_records: list[dict[str, Any]],
) -> None:
    rows: list[dict[str, Any]] = []
    rows.extend(_submetric_rows(scaffold_records, "scaffold", "baseline"))
    rows.extend(_submetric_rows(ollama_records, "local-agent", "baseline"))
    _write_csv(csv_path, rows)
    if not rows:
        tex_path.write_text("% Submetric master artifacts were not found.\n", encoding="utf-8")
        return
    lines = [
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Track & System & $N$ & Composite & Detect. & Root & Interv. & Evidence & Calib. & Safety \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _tex_escape(str(row["track"])),
                    _tex_escape(str(row["system"])),
                    str(int(row["n"])),
                    f"{float(row['composite']):.3f}",
                    f"{float(row['detection']):.3f}",
                    f"{float(row['root_cause']):.3f}",
                    f"{float(row['intervention']):.3f}",
                    f"{float(row['evidence']):.3f}",
                    f"{float(row['calibration']):.3f}",
                    str(int(row["safety_violations"])),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")


def _submetric_rows(records: list[dict[str, Any]], track: str, group_key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for system, group in sorted(
        _groups(records, group_key).items(),
        key=lambda item: _mean(item[1], "composite"),
        reverse=True,
    ):
        row: dict[str, Any] = {
            "track": track,
            "system": system,
            "n": len(group),
            "safety_violations": sum(int(record.get("safety_violations", 0)) for record in group),
        }
        for metric in METRICS:
            row[metric] = round(_mean(group, metric), 6)
        rows.append(row)
    return rows


def _write_hidden_seed_table(
    tex_path: Path,
    csv_path: Path,
    hidden_records: list[dict[str, Any]],
    hidden_ollama_dir: Path,
    bootstrap: int,
    rng: random.Random,
) -> None:
    parse = _parse_success_counts(hidden_ollama_dir)
    rows: list[dict[str, Any]] = []
    for model, group in sorted(_groups(hidden_records, "baseline").items()):
        row: dict[str, Any] = {
            "model": model,
            "n": len(group),
            "parse_success": parse.get(model, {}).get("direct_json", 0),
            "parse_total": parse.get(model, {}).get("total", 0),
            "safety_violations": sum(int(record.get("safety_violations", 0)) for record in group),
        }
        for metric in ["composite", "detection", "root_cause", "intervention", "evidence"]:
            values = [float(record[metric]) for record in group]
            mean, low, high = _bootstrap_ci(values, bootstrap, rng)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    _write_csv(csv_path, rows)
    if not rows:
        tex_path.write_text("% Hidden-seed artifacts were not found.\n", encoding="utf-8")
        return
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Model & $N$ & Composite (95\% CI) & Detect. & Root & Interv. & Evidence & Safety / parse \\",
        r"\midrule",
    ]
    for row in rows:
        parse_label = f"{int(row['safety_violations'])} / {int(row['parse_success'])}/{int(row['parse_total'])}"
        lines.append(
            " & ".join(
                [
                    _tex_escape(str(row["model"])),
                    str(int(row["n"])),
                    _fmt_ci(row["composite_mean"], row["composite_ci_low"], row["composite_ci_high"]),
                    f"{row['detection_mean']:.3f}",
                    f"{row['root_cause_mean']:.3f}",
                    f"{row['intervention_mean']:.3f}",
                    f"{row['evidence_mean']:.3f}",
                    parse_label,
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_success_counts(agent_dir: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for path in sorted((agent_dir / "predictions").glob("*_raw.jsonl")):
        model = _model_from_raw_filename(path.name)
        total = 0
        direct = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            total += 1
            record = json.loads(line)
            if record.get("json_parse_status") == "direct_json":
                direct += 1
        counts[model] = {"total": total, "direct_json": direct}
    return counts


def _model_from_raw_filename(name: str) -> str:
    safe = name.removeprefix("ollama_").removesuffix("_react_raw.jsonl")
    return _model_from_safe_name(safe)


def _load_contamination_delta_rows(contamination_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not contamination_dir.exists():
        return rows
    direct = contamination_dir / "delta_summary.csv"
    if direct.exists():
        domain = _contamination_domain(contamination_dir)
        rows.extend(_read_contamination_delta_csv(direct, domain))
    for child in sorted(path for path in contamination_dir.iterdir() if path.is_dir()):
        delta_path = child / "delta_summary.csv"
        if delta_path.exists():
            rows.extend(_read_contamination_delta_csv(delta_path, _contamination_domain(child)))
    return rows


def _read_contamination_delta_csv(path: Path, domain: str) -> list[dict[str, Any]]:
    rows = []
    for row in csv.DictReader(path.open(encoding="utf-8")):
        updated = dict(row)
        updated["domain"] = domain
        rows.append(updated)
    return rows


def _contamination_domain(path: Path) -> str:
    manifest_path = path / "experiment_manifest.json"
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            domain = payload.get("configuration", {}).get("domain")
            if domain:
                return str(domain)
        except json.JSONDecodeError:
            pass
    return path.name


def _aggregate_contamination_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregate_rows = []
    for system, group in _groups(rows, "system").items():
        total_n = sum(float(row.get("n", 0) or 0) for row in group)
        if total_n <= 0:
            continue
        aggregate: dict[str, Any] = {
            "system": system,
            "domain": "aggregate",
            "original_condition": "original",
            "neutral_condition": group[0].get("neutral_condition", "neutral-identifiers"),
            "n": int(total_n),
            "n_deg": sum(float(row.get("n_deg", 0) or 0) for row in group),
        }
        for metric in [
            "composite",
            "detection",
            "root_cause",
            "intervention",
            "evidence",
            "calibration",
            "efficiency",
            "safety_violations",
        ]:
            if metric == "safety_violations":
                original = sum(float(row.get(f"original_{metric}", 0) or 0) for row in group)
                neutral = sum(float(row.get(f"neutral_{metric}", 0) or 0) for row in group)
            else:
                original = sum(
                    float(row.get(f"original_{metric}", 0) or 0) * float(row.get("n", 0) or 0)
                    for row in group
                ) / total_n
                neutral = sum(
                    float(row.get(f"neutral_{metric}", 0) or 0) * float(row.get("n", 0) or 0)
                    for row in group
                ) / total_n
            aggregate[f"original_{metric}"] = round(original, 6)
            aggregate[f"neutral_{metric}"] = round(neutral, 6)
            aggregate[f"delta_{metric}"] = round(neutral - original, 6)
        aggregate_rows.append(aggregate)
    return aggregate_rows


def _fmt_delta_cell(row: dict[str, Any] | None) -> str:
    if not row:
        return "--"
    return f"{float(row.get('delta_composite', 0) or 0):+.3f}"


def _write_figures(output_dir: Path, scaffold_dir: Path, ollama_dir: Path, episodes: dict[str, dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        plt = None
    if plt is not None:
        _plot_topologies(output_dir / "figure2_domain_panels.pdf", plt)
        _plot_domain_scores(output_dir / "score_by_domain_bars.pdf", scaffold_dir, plt)
        _plot_latency_cdf(output_dir / "detection_latency_cdf.pdf", scaffold_dir, plt)
        _plot_reliability(output_dir / "llm_reliability_diagram.pdf", ollama_dir, episodes, plt)
        _plot_episode_walkthrough(output_dir / "episode_walkthrough.pdf", scaffold_dir, ollama_dir, plt)
    else:
        _write_episode_walkthrough_pdf(output_dir / "episode_walkthrough.pdf", scaffold_dir, ollama_dir)


def _plot_topologies(path: Path, plt: Any) -> None:
    panel_order = ["microservice", "hvac", "water_grid", "manufacturing", "bioprocess"]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.8))
    for ax, name in zip(axes.flat[:5], panel_order):
        template = get_domain(name)
        _draw_topology_panel(ax, template, _topology_panel_layout(name), _domain_axis_label(name))
    _draw_topology_key(axes.flat[5])
    fig.tight_layout(pad=1.25, w_pad=1.0, h_pad=1.6)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _topology_panel_layout(name: str) -> dict[str, tuple[float, float]]:
    layouts = {
        "microservice": {
            "api-gateway": (0.12, 0.50),
            "auth": (0.42, 0.78),
            "checkout": (0.42, 0.38),
            "payments": (0.72, 0.56),
            "database": (0.90, 0.30),
        },
        "hvac": {
            "chiller": (0.12, 0.62),
            "air-handler": (0.36, 0.62),
            "supply-fan": (0.60, 0.62),
            "vav-zone-a": (0.86, 0.78),
            "vav-zone-b": (0.86, 0.46),
        },
        "water_grid": {
            "reservoir": (0.12, 0.62),
            "pump-1": (0.38, 0.78),
            "pump-2": (0.38, 0.46),
            "main-line": (0.64, 0.62),
            "north-zone": (0.88, 0.62),
        },
        "manufacturing": {
            "feeder": (0.14, 0.58),
            "press": (0.38, 0.58),
            "cooling-loop": (0.62, 0.78),
            "vision-station": (0.62, 0.40),
            "packager": (0.88, 0.40),
        },
        "bioprocess": {
            "feed-pump": (0.12, 0.76),
            "air-sparger": (0.12, 0.54),
            "ph-control": (0.12, 0.32),
            "bioreactor": (0.56, 0.54),
            "harvest": (0.88, 0.54),
        },
    }
    return layouts[name]


def _draw_topology_panel(ax: Any, template: Any, positions: dict[str, tuple[float, float]], title: str) -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    node_width = 0.15
    node_height = 0.112
    node_pad = 0.010
    edge_width = node_width + 2 * node_pad
    edge_height = node_height + 2 * node_pad
    for left, right in template.topology:
        start = _edge_endpoint(positions[left], positions[right], edge_width, edge_height)
        end = _edge_endpoint(positions[right], positions[left], edge_width, edge_height)
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.5,
            color="#3b3b3b",
            shrinkA=0,
            shrinkB=0,
            zorder=1,
        )
        ax.add_patch(arrow)
    for node in template.components:
        x, y = positions[node]
        box = FancyBboxPatch(
            (x - node_width / 2, y - node_height / 2),
            node_width,
            node_height,
            boxstyle=f"round,pad={node_pad},rounding_size=0.022",
            linewidth=1.25,
            edgecolor="#315f7d",
            facecolor="#dceefb",
            zorder=3,
        )
        ax.add_patch(box)
        ax.text(x, y, _topology_display_label(node), ha="center", va="center", fontsize=8.0, zorder=4)
    ax.set_title(title, fontsize=12.5, fontweight="semibold", pad=8)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(0.12, 0.94)
    ax.set_axis_off()


def _edge_endpoint(
    source: tuple[float, float],
    target: tuple[float, float],
    node_width: float,
    node_height: float,
) -> tuple[float, float]:
    sx, sy = source
    tx, ty = target
    dx = tx - sx
    dy = ty - sy
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return source
    scale = 0.5 / max(abs(dx) / node_width, abs(dy) / node_height)
    return sx + dx * scale, sy + dy * scale


def _topology_display_label(node: str) -> str:
    labels = {
        "api-gateway": "API\ngateway",
        "air-handler": "air\nhandler",
        "air-sparger": "air\nsparger",
        "cooling-loop": "cooling\nloop",
        "feed-pump": "feed\npump",
        "main-line": "main\nline",
        "north-zone": "north\nzone",
        "ph-control": "pH\ncontrol",
        "pump-1": "pump 1",
        "pump-2": "pump 2",
        "supply-fan": "supply\nfan",
        "vav-zone-a": "VAV\nzone A",
        "vav-zone-b": "VAV\nzone B",
        "vision-station": "vision\nstation",
    }
    return labels.get(node, node.replace("-", "\n"))


def _draw_topology_key(ax: Any) -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    ax.set_title("visual key", fontsize=12.5, fontweight="semibold", pad=8)
    node_width = 0.20
    node_height = 0.115
    node_pad = 0.010
    left = (0.26, 0.58)
    right = (0.72, 0.58)
    arrow = FancyArrowPatch(
        _edge_endpoint(left, right, node_width + 2 * node_pad, node_height + 2 * node_pad),
        _edge_endpoint(right, left, node_width + 2 * node_pad, node_height + 2 * node_pad),
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.5,
        color="#3b3b3b",
        shrinkA=0,
        shrinkB=0,
        zorder=1,
    )
    ax.add_patch(arrow)
    for center, label in [(left, "component"), (right, "downstream\ncomponent")]:
        x, y = center
        box = FancyBboxPatch(
            (x - node_width / 2, y - node_height / 2),
            node_width,
            node_height,
            boxstyle=f"round,pad={node_pad},rounding_size=0.022",
            linewidth=1.25,
            edgecolor="#315f7d",
            facecolor="#dceefb",
            zorder=3,
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=8.0, zorder=4)
    ax.text(0.49, 0.69, "declared dependency /\npropagation edge", ha="center", va="bottom", fontsize=8.2)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(0.12, 0.94)
    ax.set_axis_off()


def _plot_domain_scores(path: Path, scaffold_dir: Path, plt: Any) -> None:
    summary = scaffold_dir / "summary_by_domain.csv"
    if not summary.exists():
        return
    rows = list(csv.DictReader(summary.open(encoding="utf-8")))
    baselines = sorted({row["baseline"] for row in rows})
    domains = sorted({row["domain"] for row in rows})
    lookup = {(row["baseline"], row["domain"]): float(row["composite_mean"]) for row in rows}
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    width = 0.8 / max(1, len(baselines))
    x = list(range(len(domains)))
    for offset, baseline in enumerate(baselines):
        values = [lookup.get((baseline, domain), 0.0) for domain in domains]
        label = _baseline_axis_label(baseline)
        ax.bar([item + offset * width for item in x], values, width=width, label=label)
    ax.set_xticks([item + width * (len(baselines) - 1) / 2 for item in x])
    ax.set_xticklabels([_domain_axis_label(domain) for domain in domains], fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_xlabel("Domain", fontsize=13, labelpad=10)
    ax.set_ylabel("Composite score", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(
        fontsize=10,
        ncol=len(baselines),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        frameon=False,
        handlelength=1.5,
        columnspacing=1.0,
    )
    ax.grid(True, axis="y", alpha=0.20, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _domain_axis_label(domain: str) -> str:
    labels = {
        "hvac": "HVAC",
        "water_grid": "water grid",
    }
    return labels.get(domain, domain.replace("_", " "))


def _baseline_axis_label(baseline: str) -> str:
    labels = {
        "noop": "no-op",
        "topology_rca": "topology-RCA",
    }
    return labels.get(baseline, baseline.replace("_", "-"))


def _plot_latency_cdf(path: Path, scaffold_dir: Path, plt: Any) -> None:
    episodes_dir = scaffold_dir / "episodes"
    if not episodes_dir.exists():
        return
    baselines = ["random", "threshold", "topology_rca", "oracle"]
    latencies: dict[str, list[int]] = {baseline: [] for baseline in baselines}
    for episode_path in sorted(episodes_dir.glob("seed_*.jsonl")):
        seed_match = re.search(r"seed_(\d+)", episode_path.stem)
        seed = int(seed_match.group(1)) if seed_match else 0
        episodes = load_episodes_jsonl(episode_path)
        for baseline_name in baselines:
            baseline = get_baseline(baseline_name, seed=seed)
            for episode in episodes:
                prediction = baseline.predict(episode)
                if prediction.alarm_time is None:
                    continue
                latencies[baseline_name].append(prediction.alarm_time - episode.ground_truth.fault.start_time)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    styles = {
        "random": {"color": "#1f77b4", "linestyle": "-", "linewidth": 2.0, "zorder": 2, "label": "random"},
        "topology_rca": {
            "color": "#2ca02c",
            "linestyle": "-",
            "linewidth": 3.0,
            "zorder": 3,
            "label": "topology-RCA",
        },
        "threshold": {
            "color": "#ff7f0e",
            "linestyle": (0, (5, 2)),
            "linewidth": 2.2,
            "marker": "o",
            "markersize": 2.6,
            "markevery": 140,
            "zorder": 4,
            "label": "threshold",
        },
        "oracle": {"color": "#d62728", "linestyle": "-", "linewidth": 2.0, "zorder": 1, "label": "oracle"},
    }
    plot_order = ["random", "topology_rca", "threshold", "oracle"]
    for baseline in plot_order:
        values = latencies[baseline]
        if not values:
            continue
        ordered = sorted(values)
        y = [(index + 1) / len(ordered) for index in range(len(ordered))]
        ax.step(ordered, y, where="post", **styles[baseline])
    ax.set_xlabel("Detection latency (time steps)")
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1.02)
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_reliability(path: Path, ollama_dir: Path, episodes: dict[str, dict[str, Any]], plt: Any) -> None:
    prediction_dir = ollama_dir / "predictions"
    if not prediction_dir.exists() or not episodes:
        return
    bins = [(index / 10, (index + 1) / 10) for index in range(10)]
    by_bin = {index: [] for index in range(10)}
    for prediction_path in sorted(prediction_dir.glob("*.jsonl")):
        if prediction_path.name.endswith("_raw.jsonl"):
            continue
        for prediction in load_predictions_jsonl(prediction_path):
            meta = episodes.get(prediction.episode_id)
            if not meta:
                continue
            correct = bool(set(prediction.action_ids) & meta["oracle_action_ids"])
            conf = max(0.0, min(1.0, prediction.action_confidence))
            index = min(9, int(conf * 10))
            by_bin[index].append((conf, 1.0 if correct else 0.0))
    xs: list[float] = []
    ys: list[float] = []
    counts: list[int] = []
    for index, values in by_bin.items():
        if not values:
            continue
        xs.append(sum(conf for conf, _ in values) / len(values))
        ys.append(sum(correct for _, correct in values) / len(values))
        counts.append(len(values))
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], "--", color="#777777", lw=1.2, zorder=1)
    max_count = max(counts) if counts else 1
    sizes = [70 + 430 * math.sqrt(count / max_count) for count in counts]
    ax.scatter(
        xs,
        ys,
        s=sizes,
        color="#315f7d",
        edgecolors="white",
        linewidths=0.9,
        alpha=0.82,
        zorder=3,
    )
    ax.set_xlabel("Mean action confidence", fontsize=12)
    ax.set_ylabel("Empirical action correctness", fontsize=12)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.22, linewidth=0.6, zorder=0)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_episode_walkthrough(path: Path, scaffold_dir: Path, ollama_dir: Path, plt: Any) -> None:
    payload = _load_walkthrough_artifacts(scaffold_dir, ollama_dir)
    if payload is None:
        return
    episode, prediction, score = payload
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.4))
    _draw_walkthrough_topology(axes[0, 0], episode, prediction)
    _draw_walkthrough_traces(axes[0, 1], episode, prediction)
    _draw_walkthrough_evidence_action(axes[1, 0], episode, prediction)
    _draw_walkthrough_losses(axes[1, 1], episode, score)
    fig.tight_layout(pad=1.1, w_pad=1.8, h_pad=1.4)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _style_walkthrough_panel(ax: Any, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, fontweight="semibold", pad=8)
    ax.set_facecolor("#fbfbfb")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#c7c7c7")
        spine.set_linewidth(0.8)


def _draw_walkthrough_topology(ax: Any, episode: Any, prediction: Any) -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    _style_walkthrough_panel(ax, "A. Topology and fault path")
    positions = {
        "api-gateway": (0.12, 0.55),
        "auth": (0.38, 0.82),
        "checkout": (0.40, 0.55),
        "payments": (0.68, 0.66),
        "database": (0.84, 0.34),
    }
    node_width = 0.20
    node_height = 0.12
    edge_width = node_width + 0.03
    edge_height = node_height + 0.03
    path_nodes = episode.ground_truth.fault.root_cause_path
    path_node_set = set(path_nodes)

    for left, right in episode.topology:
        start = _edge_endpoint(positions[left], positions[right], edge_width, edge_height)
        end = _edge_endpoint(positions[right], positions[left], edge_width, edge_height)
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.2,
                color="#666666",
                alpha=0.78,
                shrinkA=0,
                shrinkB=0,
                zorder=1,
            )
        )
    for left, right in zip(path_nodes, path_nodes[1:]):
        if left not in positions or right not in positions:
            continue
        start = _edge_endpoint(positions[left], positions[right], edge_width, edge_height)
        end = _edge_endpoint(positions[right], positions[left], edge_width, edge_height)
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.8,
                linestyle=(0, (4, 2)),
                color="#b24a4a",
                connectionstyle="arc3,rad=0.16",
                shrinkA=0,
                shrinkB=0,
                zorder=2,
            )
        )
    for node, (x, y) in positions.items():
        is_path = node in path_node_set
        box = FancyBboxPatch(
            (x - node_width / 2, y - node_height / 2),
            node_width,
            node_height,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=1.35,
            edgecolor="#9b2d2d" if is_path else "#315f7d",
            facecolor="#f4dada" if is_path else "#dceefb",
            zorder=3,
        )
        ax.add_patch(box)
        ax.text(x, y, _topology_display_label(node), ha="center", va="center", fontsize=8.3, zorder=4)
    cause = prediction.root_cause_topk[0] if prediction.root_cause_topk else "none"
    ax.text(0.06, 0.11, f"Predicted RCA: {cause}", fontsize=8.6, color="#222222", va="center")
    ax.plot([0.06, 0.17], [0.22, 0.22], color="#666666", lw=1.2)
    ax.text(0.19, 0.22, "declared dependency", fontsize=7.8, va="center", color="#333333")
    ax.plot([0.55, 0.66], [0.22, 0.22], color="#b24a4a", lw=1.8, linestyle=(0, (4, 2)))
    ax.text(0.68, 0.22, "stored fault path", fontsize=7.8, va="center", color="#333333")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_walkthrough_traces(ax: Any, episode: Any, prediction: Any) -> None:
    _style_walkthrough_panel(ax, "B. Propagated sensor drift")
    sensors = [
        ("db.lock_wait_ms", "DB lock wait", "#9b2d2d"),
        ("checkout.queue_depth", "Checkout queue", "#315f7d"),
        ("api.latency_ms", "API latency", "#5c8432"),
    ]
    times = [frame.timestamp for frame in episode.observations]
    fault_start = episode.ground_truth.fault.start_time
    all_values: list[float] = []
    for sensor, label, color in sensors:
        values = [frame.sensors[sensor] for frame in episode.observations]
        baseline = [frame.sensors[sensor] for frame in episode.observations if frame.timestamp < fault_start]
        mean = sum(baseline) / len(baseline)
        variance = sum((value - mean) ** 2 for value in baseline) / max(1, len(baseline))
        std = math.sqrt(variance) or 1.0
        normalized = [(value - mean) / std for value in values]
        all_values.extend(normalized)
        ax.plot(times, normalized, color=color, lw=2.0, label=label)
    if prediction.alarm_time == fault_start:
        ax.axvline(fault_start, color="#8a6d1d", linestyle=(0, (4, 3)), lw=1.4)
        ax.text(
            fault_start + 0.6,
            0.95,
            f"fault/alarm t={fault_start}",
            transform=ax.get_xaxis_transform(),
            fontsize=8.2,
            color="#5f4b17",
            ha="left",
            va="top",
        )
    else:
        ax.axvline(fault_start, color="#8a6d1d", linestyle=(0, (4, 3)), lw=1.2, label=f"fault t={fault_start}")
        if prediction.alarm_time is not None:
            ax.axvline(prediction.alarm_time, color="#333333", linestyle=":", lw=1.2, label=f"alarm t={prediction.alarm_time}")
    y_min = min(all_values)
    y_max = max(all_values)
    pad = max(0.5, (y_max - y_min) * 0.08)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlabel("time step", fontsize=9)
    ax.set_ylabel("baseline-normalized drift", fontsize=9)
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.tick_params(labelsize=8)


def _draw_walkthrough_evidence_action(ax: Any, episode: Any, prediction: Any) -> None:
    from matplotlib.patches import FancyBboxPatch

    _style_walkthrough_panel(ax, "C. Cited evidence and action")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    records = {record.span_id: record for record in episode.manuals}
    for frame in episode.observations:
        for record in frame.evidence:
            records[record.span_id] = record
    row_y = [0.76, 0.56, 0.36]
    for y, span_id in zip(row_y, prediction.evidence_spans[:3]):
        record = records.get(span_id)
        timestamp = "" if record is None else ("manual" if record.timestamp is None else f"t={record.timestamp}")
        text = span_id if record is None else record.text
        ax.add_patch(
            FancyBboxPatch(
                (0.06, y - 0.07),
                0.88,
                0.14,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                linewidth=0.7,
                edgecolor="#d0d0d0",
                facecolor="#ffffff",
            )
        )
        ax.text(0.09, y + 0.025, timestamp, fontsize=8.8, fontweight="bold", color="#333333", va="center")
        ax.text(0.24, y + 0.025, "\n".join(textwrap.wrap(text, width=48)[:2]), fontsize=8.4, color="#222222", va="center")
    action = ", ".join(prediction.action_ids) if prediction.action_ids else "none"
    ax.add_patch(
        FancyBboxPatch(
            (0.06, 0.09),
            0.88,
            0.15,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=0.9,
            edgecolor="#5c8432",
            facecolor="#edf5e8",
        )
    )
    ax.text(0.09, 0.165, f"Selected action: {action}", fontsize=9.2, fontweight="bold", color="#273d16", va="center")


def _draw_walkthrough_losses(ax: Any, episode: Any, score: dict[str, Any]) -> None:
    _style_walkthrough_panel(ax, "D. Replay loss contrast")
    labels = ["No-op", "gemma4:26b\npolicy", "Oracle"]
    values = [episode.ground_truth.noop_loss, float(score["policy_loss"]), episode.ground_truth.oracle_loss]
    colors = ["#b24a4a", "#315f7d", "#5c8432"]
    bars = ax.bar(range(len(values)), values, color=colors, width=0.56, edgecolor="none")
    for bar, value, color in zip(bars, values, colors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.035,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=color,
        )
    ax.text(
        0.98,
        0.95,
        "lower loss is better",
        transform=ax.transAxes,
        fontsize=8.2,
        color="#444444",
        ha="right",
        va="top",
    )
    ax.set_ylabel("incident loss", fontsize=9)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.6)
    ax.tick_params(axis="y", labelsize=8)


def _write_episode_walkthrough_pdf(path: Path, scaffold_dir: Path, ollama_dir: Path) -> None:
    payload = _load_walkthrough_artifacts(scaffold_dir, ollama_dir)
    if payload is None:
        return
    episode, prediction, score = payload
    pdf = _PdfCanvas(path, 792, 540)

    pdf.text(36, 510, "Example episode: database lock contention in microservice operations", 13, bold=True)
    _draw_pdf_panel(pdf, 32, 282, 330, 202, "Topology and propagation path")
    _draw_pdf_panel(pdf, 390, 282, 360, 202, "Propagated sensor ramp")
    _draw_pdf_panel(pdf, 32, 34, 330, 218, "Retrieved evidence and action")
    _draw_pdf_panel(pdf, 390, 34, 360, 218, "Replay loss contrast")

    _draw_pdf_topology(pdf, 45, 300, episode, prediction)
    _draw_pdf_traces(pdf, 412, 310, 304, 126, episode, prediction)
    _draw_pdf_evidence(pdf, 48, 58, episode, prediction)
    _draw_pdf_losses(pdf, 424, 72, 278, 122, episode, score)
    pdf.save()


def _load_walkthrough_artifacts(scaffold_dir: Path, ollama_dir: Path) -> tuple[Any, Any, dict[str, Any]] | None:
    episode_id = "cob-microservice-0003-8fc7a49d"
    episode = None
    for episode_path in sorted((scaffold_dir / "episodes").glob("*.jsonl")):
        for candidate in load_episodes_jsonl(episode_path):
            if candidate.episode_id == episode_id:
                episode = candidate
                break
        if episode is not None:
            break
    if episode is None:
        return None

    prediction = None
    prediction_path = ollama_dir / "predictions" / "ollama_gemma4_26b_react.jsonl"
    if prediction_path.exists():
        for candidate in load_predictions_jsonl(prediction_path):
            if candidate.episode_id == episode_id:
                prediction = candidate
                break
    if prediction is None:
        return None

    score_path = ollama_dir / "scores" / "ollama_gemma4_26b_react.json"
    if not score_path.exists():
        return None
    score = None
    payload = json.loads(score_path.read_text(encoding="utf-8"))
    for candidate in payload.get("scores", []):
        if candidate.get("episode_id") == episode_id:
            score = candidate
            break
    if score is None:
        return None
    return episode, prediction, score


def _draw_pdf_panel(pdf: Any, x: float, y: float, width: float, height: float, title: str) -> None:
    pdf.rect(x, y, width, height, stroke=(0.78, 0.78, 0.78), fill=(1, 1, 1), lw=0.8)
    pdf.text(x + 10, y + height - 18, title, 10, bold=True)


def _draw_pdf_topology(pdf: Any, x: float, y: float, episode: Any, prediction: Any) -> None:
    positions = {
        "api-gateway": (x + 8, y + 102),
        "auth": (x + 108, y + 152),
        "checkout": (x + 130, y + 102),
        "payments": (x + 250, y + 132),
        "database": (x + 250, y + 58),
    }
    for left, right in episode.topology:
        pdf.arrow(*positions[left], *positions[right], color=(0.55, 0.55, 0.55), lw=0.9)
    path_nodes = episode.ground_truth.fault.root_cause_path
    for left, right in zip(path_nodes, path_nodes[1:]):
        pdf.arrow(*positions[left], *positions[right], color=(0.69, 0.29, 0.29), lw=2.1)
    for node, (nx, ny) in positions.items():
        fill = (0.96, 0.84, 0.84) if node in path_nodes else (0.86, 0.93, 0.98)
        stroke = (0.61, 0.18, 0.18) if node in path_nodes else (0.19, 0.37, 0.49)
        pdf.rect(nx - 35, ny - 13, 70, 26, stroke=stroke, fill=fill, lw=1.0)
        pdf.text(nx - 28, ny - 4, node, 7)
    cause = prediction.root_cause_topk[0] if prediction.root_cause_topk else "none"
    pdf.text(x, y + 14, f"Predicted root cause: {cause}", 8)


def _draw_pdf_traces(pdf: Any, x: float, y: float, width: float, height: float, episode: Any, prediction: Any) -> None:
    sensors = [
        ("db.lock_wait_ms", "DB lock wait", (0.61, 0.18, 0.18)),
        ("checkout.queue_depth", "Checkout queue", (0.19, 0.37, 0.49)),
        ("api.latency_ms", "API latency", (0.36, 0.50, 0.21)),
    ]
    times = [frame.timestamp for frame in episode.observations]
    fault_start = episode.ground_truth.fault.start_time
    all_series: list[tuple[str, tuple[float, float, float], list[float]]] = []
    all_values: list[float] = []
    for sensor, label, color in sensors:
        values = [frame.sensors[sensor] for frame in episode.observations]
        baseline = [frame.sensors[sensor] for frame in episode.observations if frame.timestamp < fault_start]
        mean = sum(baseline) / len(baseline)
        variance = sum((value - mean) ** 2 for value in baseline) / max(1, len(baseline))
        std = math.sqrt(variance) or 1.0
        normalized = [(value - mean) / std for value in values]
        all_series.append((label, color, normalized))
        all_values.extend(normalized)
    y_min = min(all_values)
    y_max = max(all_values)
    pad = max(0.5, (y_max - y_min) * 0.08)
    y_min -= pad
    y_max += pad
    pdf.line(x, y, x + width, y, color=(0.25, 0.25, 0.25), lw=0.8)
    pdf.line(x, y, x, y + height, color=(0.25, 0.25, 0.25), lw=0.8)

    def sx(t: float) -> float:
        return x + (t - min(times)) / (max(times) - min(times)) * width

    def sy(value: float) -> float:
        return y + (value - y_min) / (y_max - y_min) * height

    for label, color, values in all_series:
        points = [(sx(t), sy(value)) for t, value in zip(times, values)]
        pdf.polyline(points, color=color, lw=1.4)
    pdf.line(sx(fault_start), y, sx(fault_start), y + height, color=(0.25, 0.25, 0.25), lw=0.8, dash=True)
    if prediction.alarm_time is not None:
        pdf.line(sx(prediction.alarm_time), y, sx(prediction.alarm_time), y + height, color=(0.61, 0.49, 0.18), lw=1.0, dash=True)
    pdf.text(x, y - 17, "time step", 7)
    pdf.text(x - 4, y + height + 8, "baseline-normalized drift", 7)
    legend_x = x + 6
    legend_y = y + height - 12
    for label, color, _ in all_series:
        pdf.line(legend_x, legend_y, legend_x + 16, legend_y, color=color, lw=1.5)
        pdf.text(legend_x + 20, legend_y - 3, label, 7)
        legend_y -= 12


def _draw_pdf_evidence(pdf: Any, x: float, y: float, episode: Any, prediction: Any) -> None:
    records = {record.span_id: record for record in episode.manuals}
    for frame in episode.observations:
        for record in frame.evidence:
            records[record.span_id] = record
    current_y = y + 155
    for span_id in prediction.evidence_spans[:3]:
        record = records.get(span_id)
        if record is None:
            timestamp = ""
            text = span_id
        else:
            timestamp = "manual" if record.timestamp is None else f"t={record.timestamp}"
            text = record.text
        pdf.text(x, current_y, timestamp, 8, bold=True)
        lines = textwrap.wrap(text, width=50)
        for line in lines[:3]:
            pdf.text(x + 48, current_y, line, 8)
            current_y -= 11
        current_y -= 8
    action = ", ".join(prediction.action_ids) if prediction.action_ids else "none"
    pdf.rect(x, y + 8, 266, 30, stroke=(0.36, 0.50, 0.21), fill=(0.93, 0.96, 0.90), lw=0.9)
    pdf.text(x + 8, y + 20, f"Selected action: {action}", 9, bold=True)


def _draw_pdf_losses(pdf: Any, x: float, y: float, width: float, height: float, episode: Any, score: dict[str, Any]) -> None:
    labels = ["No-op", "gemma4:26b policy", "Oracle"]
    values = [episode.ground_truth.noop_loss, float(score["policy_loss"]), episode.ground_truth.oracle_loss]
    colors = [(0.69, 0.29, 0.29), (0.19, 0.37, 0.49), (0.36, 0.50, 0.21)]
    max_value = max(values)
    bar_width = 48
    gap = (width - 3 * bar_width) / 4
    pdf.line(x, y, x + width, y, color=(0.25, 0.25, 0.25), lw=0.8)
    pdf.line(x, y, x, y + height, color=(0.25, 0.25, 0.25), lw=0.8)
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        bx = x + gap + index * (bar_width + gap)
        bar_height = value / max_value * height
        pdf.rect(bx, y, bar_width, bar_height, stroke=color, fill=color, lw=0.8)
        pdf.text(bx + 3, y + bar_height + 8, f"{value:.3f}", 8)
        for line_index, line in enumerate(textwrap.wrap(label, width=12)):
            pdf.text(bx - 3, y - 15 - line_index * 10, line, 7)
    pdf.text(x - 2, y + height + 8, "incident loss", 7)


class _PdfCanvas:
    def __init__(self, path: Path, width: int, height: int) -> None:
        self.path = path
        self.width = width
        self.height = height
        self.commands: list[str] = []

    def text(self, x: float, y: float, text: str, size: int = 9, bold: bool = False) -> None:
        font = "F2" if bold else "F1"
        self.commands.append(
            f"BT /{font} {size} Tf 1 0 0 1 {_num(x)} {_num(y)} Tm ({_pdf_escape(text)}) Tj ET"
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: tuple[float, float, float] = (0, 0, 0),
        lw: float = 1.0,
        dash: bool = False,
    ) -> None:
        dash_cmd = "[3 3] 0 d " if dash else "[] 0 d "
        self.commands.append(
            f"{dash_cmd}{_rgb(color, stroke=True)} {_num(lw)} w {_num(x1)} {_num(y1)} m {_num(x2)} {_num(y2)} l S"
        )

    def arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: tuple[float, float, float] = (0, 0, 0),
        lw: float = 1.0,
    ) -> None:
        self.line(x1, y1, x2, y2, color=color, lw=lw)
        angle = math.atan2(y2 - y1, x2 - x1)
        length = 8.0
        spread = 0.45
        p1 = (x2 - length * math.cos(angle - spread), y2 - length * math.sin(angle - spread))
        p2 = (x2 - length * math.cos(angle + spread), y2 - length * math.sin(angle + spread))
        self.commands.append(
            f"{_rgb(color, stroke=False)} {_num(x2)} {_num(y2)} m {_num(p1[0])} {_num(p1[1])} l {_num(p2[0])} {_num(p2[1])} l f"
        )

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        stroke: tuple[float, float, float] = (0, 0, 0),
        fill: tuple[float, float, float] | None = None,
        lw: float = 1.0,
    ) -> None:
        if fill is not None:
            self.commands.append(
                f"{_rgb(fill, stroke=False)} {_num(x)} {_num(y)} {_num(width)} {_num(height)} re f"
            )
        self.commands.append(
            f"{_rgb(stroke, stroke=True)} {_num(lw)} w {_num(x)} {_num(y)} {_num(width)} {_num(height)} re S"
        )

    def polyline(
        self,
        points: list[tuple[float, float]],
        color: tuple[float, float, float] = (0, 0, 0),
        lw: float = 1.0,
    ) -> None:
        if len(points) < 2:
            return
        commands = [f"{_rgb(color, stroke=True)} {_num(lw)} w {_num(points[0][0])} {_num(points[0][1])} m"]
        commands.extend(f"{_num(x)} {_num(y)} l" for x, y in points[1:])
        commands.append("S")
        self.commands.append(" ".join(commands))

    def save(self) -> None:
        stream = "\n".join(self.commands).encode("latin-1", errors="replace")
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width} {self.height}] "
                f"/Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> /Contents 4 0 R >>"
            ).encode("ascii"),
            b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        ]
        payload = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for index, obj in enumerate(objects, start=1):
            offsets.append(len(payload))
            payload.extend(f"{index} 0 obj\n".encode("ascii"))
            payload.extend(obj)
            payload.extend(b"\nendobj\n")
        xref_offset = len(payload)
        payload.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        payload.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            payload.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
        payload.extend(
            (
                f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
                f"startxref\n{xref_offset}\n%%EOF\n"
            ).encode("ascii")
        )
        self.path.write_bytes(bytes(payload))


def _pdf_escape(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _num(value: float) -> str:
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def _rgb(color: tuple[float, float, float], stroke: bool) -> str:
    op = "RG" if stroke else "rg"
    return " ".join(_num(max(0.0, min(1.0, item))) for item in color) + f" {op}"


def _write_analysis_manifest(
    path: Path,
    episodes: dict[str, dict[str, Any]],
    scaffold_records: list[dict[str, Any]],
    args: argparse.Namespace,
    sweep_row: dict[str, Any],
) -> None:
    episode_values = [meta["episode"] for meta in episodes.values()]
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner": "scripts/analyze_revision_artifacts.py",
        "configuration": {
            "scaffold_dir": args.scaffold_dir,
            "ollama_dir": args.ollama_dir,
            "ablation_root": args.ablation_root,
            "extra_ablation_root": args.extra_ablation_root,
            "contamination_dir": args.contamination_dir,
            "joint_contamination_dir": args.joint_contamination_dir,
            "prompt_root": args.prompt_root,
            "hidden_scaffold_dir": args.hidden_scaffold_dir,
            "hidden_ollama_dir": args.hidden_ollama_dir,
            "model_card_dir": args.model_card_dir,
            "bootstrap": args.bootstrap,
            "seed": args.seed,
            "random_weight_samples": args.random_weight_samples,
            "random_weight_seed": args.random_weight_seed,
        },
        "N": len(episode_values),
        "N_deg": sum(1 for episode in episode_values if is_degenerate_intervention(episode)),
        "intervention_epsilon": INTERVENTION_EPSILON,
        "scaffold_records": len(scaffold_records),
        "random_weight_sweep": sweep_row,
    }
    write_json_document(path, payload)


def _model_rows(model_card_dir: Path) -> list[dict[str, str]]:
    ids = _ollama_ids(model_card_dir / "ollama_list.txt")
    rows: list[dict[str, str]] = []
    for path in sorted(model_card_dir.glob("*_modelfile.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        model = _model_from_safe_name(path.name.replace("_modelfile.txt", ""))
        digest = _sha_from_modelfile(text)
        rows.append(
            {
                "model": model,
                "pull": f"ollama pull {model}",
                "ollama_id": ids.get(model, "")[:12],
                "sha256": digest,
                "sha256_short": digest[:12],
                "source_cite": _model_source_cite(model),
            }
        )
    if rows:
        return rows
    for model, model_id in sorted(ids.items()):
        rows.append(
            {
                "model": model,
                "pull": f"ollama pull {model}",
                "ollama_id": model_id[:12],
                "sha256": "",
                "sha256_short": "",
                "source_cite": _model_source_cite(model),
            }
        )
    return rows


def _model_source_cite(model: str) -> str:
    return {
        "gemma4:e2b": r"\cite{google2026gemma4e2b}",
        "gemma4:e4b": r"\cite{google2026gemma4e4b}",
        "gemma4:26b": r"\cite{google2026gemma426ba4bit}",
        "qwen3.5:4b": r"\cite{qwen2026qwen354b}",
        "qwen3.5:9b": r"\cite{qwen2026qwen359b}",
        "granite4.1:3b": r"\cite{ibm2026granite413b}",
        "granite4.1:8b": r"\cite{ibm2026granite418b}",
    }.get(model, "")


def _ollama_ids(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    ids: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2:
            ids[parts[0]] = parts[1]
    return ids


def _sha_from_modelfile(text: str) -> str:
    match = re.search(r"sha256-([0-9a-f]{64})", text)
    return match.group(1) if match else ""


def _model_from_safe_name(value: str) -> str:
    known = {
        "gemma4_e2b": "gemma4:e2b",
        "qwen3.5_4b": "qwen3.5:4b",
        "granite4.1_3b": "granite4.1:3b",
        "gemma4_e4b": "gemma4:e4b",
        "qwen3.5_9b": "qwen3.5:9b",
        "granite4.1_8b": "granite4.1:8b",
        "gemma4_26b": "gemma4:26b",
    }
    return known.get(value, value.replace("__", "/").replace("_", ":", 1))


def _weighted_score(record: dict[str, Any], weights: dict[str, float]) -> float:
    value = sum(float(record[metric]) * weight for metric, weight in weights.items())
    if int(record.get("safety_violations", 0)):
        value = min(value, 0.45)
    return max(0.0, min(1.0, value))


def _bootstrap_ci(values: list[float], bootstrap: int, rng: random.Random) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1 or bootstrap <= 0:
        return round(mean, 6), round(mean, 6), round(mean, 6)
    try:
        import numpy as np

        array = np.asarray(values, dtype=float)
        np_rng = np.random.default_rng(rng.randrange(2**32))
        indices = np_rng.integers(0, len(array), size=(bootstrap, len(array)))
        means = array[indices].mean(axis=1)
        low, high = np.quantile(means, [0.025, 0.975])
    except Exception:
        bootstrap = min(bootstrap, 1000)
        means = []
        for _ in range(bootstrap):
            total = 0.0
            for _ in values:
                total += values[rng.randrange(len(values))]
            means.append(total / len(values))
        means.sort()
        low = means[int(0.025 * len(means))]
        high = means[int(0.975 * len(means))]
    return round(mean, 6), round(float(low), 6), round(float(high), 6)


def _paired_bootstrap_p(left: list[float], right: list[float], bootstrap: int, rng: random.Random) -> float:
    if not left or len(left) != len(right):
        return 1.0
    diffs = [a - b for a, b in zip(left, right)]
    observed = sum(diffs) / len(diffs)
    if observed == 0:
        return 1.0
    try:
        import numpy as np

        array = np.asarray(diffs, dtype=float)
        np_rng = np.random.default_rng(rng.randrange(2**32))
        indices = np_rng.integers(0, len(array), size=(bootstrap, len(array)))
        means = array[indices].mean(axis=1)
        if observed > 0:
            tail = (means <= 0).mean()
        else:
            tail = (means >= 0).mean()
        return min(1.0, float(2 * tail))
    except Exception:
        bootstrap = min(bootstrap, 1000)
        opposite = 0
        for _ in range(bootstrap):
            total = 0.0
            for _ in diffs:
                total += diffs[rng.randrange(len(diffs))]
            mean = total / len(diffs)
            if (observed > 0 and mean <= 0) or (observed < 0 and mean >= 0):
                opposite += 1
        return min(1.0, 2 * opposite / bootstrap)


def _paired_values(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    key: str,
    metric: str,
) -> list[tuple[float, float]]:
    right_by_key = {record[key]: record for record in right}
    pairs: list[tuple[float, float]] = []
    for record in left:
        other = right_by_key.get(record[key])
        if other:
            pairs.append((float(record[metric]), float(other[metric])))
    return pairs


def _groups(records: Iterable[dict[str, Any]], key: str) -> dict[Any, list[dict[str, Any]]]:
    groups: dict[Any, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record.get(key), []).append(record)
    return groups


def _mean(records: list[dict[str, Any]], metric: str) -> float:
    if not records:
        return 0.0
    return sum(float(record[metric]) for record in records) / len(records)


def _stdev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _population_stdev(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _short_model_name(value: str) -> str:
    if value.startswith("ollama:"):
        return value.split(":", 1)[1]
    match = re.match(r"Ollama-(.+)-ReAct", value)
    return match.group(1) if match else value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    write_csv_rows(path, rows, fieldnames=fieldnames)


def _fmt_ci(mean: float, low: float, high: float) -> str:
    return f"{mean:.3f} [{low:.3f}, {high:.3f}]"


def _tex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


if __name__ == "__main__":
    raise SystemExit(main())
