#!/usr/bin/env python3
"""Run renamed/paraphrased contamination stress tests for CausalOpsBench."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causalopsbench._io import write_csv_rows, write_json_document
from causalopsbench.baselines import get_baseline
from causalopsbench.evaluate import evaluate_predictions
from causalopsbench.metrics import INTERVENTION_EPSILON, is_degenerate_intervention
from causalopsbench.schemas import (
    ActionSpec,
    Episode,
    EvidenceRecord,
    FaultSpec,
    GroundTruth,
    ObservationFrame,
    Prediction,
    load_episodes_jsonl,
    to_dict,
    write_jsonl,
)
from scripts.run_foundation_agent_experiments import FoundationAgent, _slug


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run microservice original-vs-neutralized contamination stress tests."
    )
    parser.add_argument("--episodes-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--domain", default="microservice")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--baselines", nargs="*", default=[])
    parser.add_argument("--transform", default="neutral-identifiers", choices=["neutral-identifiers"])
    parser.add_argument("--policy", default="react-json", choices=["react-json"])
    parser.add_argument(
        "--prompt-style",
        default="react-json",
        choices=["react-json", "terse-json", "react-verbose", "evidence-first"],
    )
    parser.add_argument(
        "--view-ablation",
        default="none",
        choices=["none", "no-evidence", "no-topology", "no-manuals"],
        help="Remove one public prompt component for model predictions.",
    )
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--keep-alive", default="5m")
    parser.add_argument("--think", type=_parse_bool, default=False)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "episodes").mkdir(exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    (output_dir / "scores").mkdir(exist_ok=True)

    original = _load_domain_seed_episodes(Path(args.episodes_dir), args.domain, set(args.seeds))
    if not original:
        raise SystemExit("No matching episodes found")
    neutral = [_neutralize_episode(episode, index) for index, episode in enumerate(original)]

    conditions = {
        "original": original,
        args.transform: neutral,
    }
    for condition, episodes in conditions.items():
        write_jsonl(output_dir / "episodes" / f"{condition}.jsonl", episodes)

    summary_rows: list[dict[str, Any]] = []
    for baseline_name in args.baselines:
        for condition, episodes in conditions.items():
            baseline = get_baseline(baseline_name, seed=0)
            predictions = [baseline.predict(episode) for episode in episodes]
            row = _score_and_write(
                output_dir=output_dir,
                condition=condition,
                system=baseline_name,
                system_kind="baseline",
                episodes=episodes,
                predictions=predictions,
            )
            summary_rows.append(row)

    for model_spec in args.models:
        runner = FoundationAgent(
            model_spec=model_spec,
            policy=args.policy,
            prompt_style=args.prompt_style,
            view_ablation=args.view_ablation,
            max_steps=args.max_steps,
            temperature=args.temperature,
            timeout_s=args.timeout_s,
            ollama_host=args.ollama_host,
            num_ctx=args.num_ctx,
            keep_alive=args.keep_alive,
            think=args.think,
        )
        for condition, episodes in conditions.items():
            predictions: list[Prediction] = []
            raw_records: list[dict[str, Any]] = []
            for episode in episodes:
                prediction, raw = runner.predict(episode)
                predictions.append(prediction)
                raw_records.append(raw)
            raw_path = output_dir / "predictions" / f"{condition}_{_slug(runner.display_name)}_raw.jsonl"
            raw_path.write_text(
                "\n".join(json.dumps(record, sort_keys=True) for record in raw_records) + "\n",
                encoding="utf-8",
            )
            row = _score_and_write(
                output_dir=output_dir,
                condition=condition,
                system=runner.model,
                system_kind="model",
                episodes=episodes,
                predictions=predictions,
            )
            summary_rows.append(row)

    delta_rows = _delta_rows(summary_rows, args.transform)
    _write_csv(output_dir / "summary_by_condition.csv", summary_rows)
    _write_csv(output_dir / "delta_summary.csv", delta_rows)
    _write_manifest(output_dir / "experiment_manifest.json", args, original, neutral)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "episodes": len(original),
                "summary_csv": str(output_dir / "summary_by_condition.csv"),
            },
            sort_keys=True,
        )
    )
    return 0


def _load_domain_seed_episodes(episodes_dir: Path, domain: str, seeds: set[int]) -> list[Episode]:
    episodes: list[Episode] = []
    for seed in sorted(seeds):
        path = episodes_dir / f"seed_{seed}.jsonl"
        if not path.exists():
            continue
        episodes.extend(
            episode
            for episode in load_episodes_jsonl(path)
            if episode.domain == domain and int(episode.metadata.get("seed", seed)) in seeds
        )
    return episodes


def _neutralize_episode(episode: Episode, index: int) -> Episode:
    components = _ordered_components(episode)
    sensors = sorted(episode.observations[0].sensors) if episode.observations else []
    actions = [action.action_id for action in episode.actions]
    fault_types = _ordered_fault_types(episode)

    component_map = {name: f"C_{idx:03d}" for idx, name in enumerate(components, start=1)}
    sensor_map = {name: f"S_{idx:03d}" for idx, name in enumerate(sensors, start=1)}
    action_map = {name: f"A_{idx:03d}" for idx, name in enumerate(actions, start=1)}
    fault_map = {name: f"F_{idx:03d}" for idx, name in enumerate(fault_types, start=1)}
    span_map = _span_map(episode)
    replacements = {
        **component_map,
        **sensor_map,
        **action_map,
        **fault_map,
    }

    manuals = [
        _neutralize_evidence_record(manual, span_map, replacements, source_prefix="SRC_MANUAL")
        for manual in episode.manuals
    ]
    observations = [
        ObservationFrame(
            timestamp=frame.timestamp,
            sensors={sensor_map[sensor]: value for sensor, value in frame.sensors.items()},
            evidence=[
                _neutralize_evidence_record(record, span_map, replacements, source_prefix="SRC_OBS")
                for record in frame.evidence
            ],
        )
        for frame in episode.observations
    ]
    actions_neutral = [
        ActionSpec(
            action_id=action_map[action.action_id],
            target_component=component_map[action.target_component],
            action_type=action.action_type,
            cost=action.cost,
            safety_risk=action.safety_risk,
            expected_faults=[
                _neutralize_label(label, component_map, fault_map)
                for label in action.expected_faults
            ],
            description=_replace_terms(action.description, replacements),
        )
        for action in episode.actions
    ]
    fault = episode.ground_truth.fault
    neutral_fault = FaultSpec(
        component=component_map[fault.component],
        fault_type=fault_map[fault.fault_type],
        start_time=fault.start_time,
        severity=fault.severity,
        root_cause_path=[component_map[item] for item in fault.root_cause_path],
    )
    truth = GroundTruth(
        fault=neutral_fault,
        gold_evidence_spans=[span_map[span] for span in episode.ground_truth.gold_evidence_spans],
        oracle_action_ids=[action_map[action] for action in episode.ground_truth.oracle_action_ids],
        noop_loss=episode.ground_truth.noop_loss,
        oracle_loss=episode.ground_truth.oracle_loss,
        safety_critical_actions=[
            action_map[action] for action in episode.ground_truth.safety_critical_actions
        ],
    )
    metadata = {
        "seed": episode.metadata.get("seed"),
        "generator": episode.metadata.get("generator"),
        "original_episode_id": episode.episode_id,
        "contamination_transform": "neutral-identifiers",
        "sensor_components": {
            sensor_map[sensor]: component_map[component]
            for sensor, component in _episode_sensor_components(episode).items()
            if sensor in sensor_map and component in component_map
        },
        "component_map": component_map,
        "sensor_map": sensor_map,
        "action_map": action_map,
        "fault_type_map": fault_map,
    }
    return replace(
        episode,
        episode_id=f"neutral-{index:04d}",
        topology=[(component_map[left], component_map[right]) for left, right in episode.topology],
        manuals=manuals,
        observations=observations,
        actions=actions_neutral,
        ground_truth=truth,
        metadata=metadata,
    )


def _ordered_components(episode: Episode) -> list[str]:
    components: list[str] = []
    for left, right in episode.topology:
        for component in (left, right):
            if component not in components:
                components.append(component)
    for action in episode.actions:
        if action.target_component not in components:
            components.append(action.target_component)
    return components


def _ordered_fault_types(episode: Episode) -> list[str]:
    values: list[str] = [episode.ground_truth.fault.fault_type]
    for action in episode.actions:
        for label in action.expected_faults:
            if ":" not in label:
                continue
            fault_type = label.split(":", 1)[1]
            if fault_type not in values:
                values.append(fault_type)
    return values


def _span_map(episode: Episode) -> dict[str, str]:
    spans: list[str] = [manual.span_id for manual in episode.manuals]
    for frame in episode.observations:
        spans.extend(record.span_id for record in frame.evidence)
    return {span: f"E_{idx:03d}" for idx, span in enumerate(spans, start=1)}


def _episode_sensor_components(episode: Episode) -> dict[str, str]:
    try:
        from causalopsbench.domains import get_domain

        return dict(get_domain(episode.domain).sensor_components)
    except Exception:
        return {}


def _neutralize_evidence_record(
    record: EvidenceRecord,
    span_map: dict[str, str],
    replacements: dict[str, str],
    source_prefix: str,
) -> EvidenceRecord:
    return EvidenceRecord(
        span_id=span_map[record.span_id],
        source_id=f"{source_prefix}_{span_map[record.span_id].split('_', 1)[1]}",
        kind=record.kind,
        timestamp=record.timestamp,
        text=_replace_terms(record.text, replacements),
    )


def _neutralize_label(label: str, component_map: dict[str, str], fault_map: dict[str, str]) -> str:
    if ":" not in label:
        return label
    component, fault_type = label.split(":", 1)
    return f"{component_map.get(component, component)}:{fault_map.get(fault_type, fault_type)}"


def _replace_terms(text: str, replacements: dict[str, str]) -> str:
    updated = text
    variants: dict[str, str] = {}
    for source, target in replacements.items():
        variants[source] = target
        variants[source.replace("-", " ")] = target
        variants[source.replace("-", "_")] = target
        variants[source.replace("_", "-")] = target
    for source, target in sorted(variants.items(), key=lambda item: len(item[0]), reverse=True):
        if not source:
            continue
        updated = re.sub(re.escape(source), target, updated, flags=re.IGNORECASE)
    return updated


def _score_and_write(
    output_dir: Path,
    condition: str,
    system: str,
    system_kind: str,
    episodes: list[Episode],
    predictions: list[Prediction],
) -> dict[str, Any]:
    slug = _safe_slug(f"{condition}_{system}")
    prediction_path = output_dir / "predictions" / f"{slug}.jsonl"
    score_path = output_dir / "scores" / f"{slug}.json"
    write_jsonl(prediction_path, predictions)
    scores, summary = evaluate_predictions(episodes, predictions)
    payload = {
        "condition": condition,
        "system": system,
        "system_kind": system_kind,
        "count": len(predictions),
        "summary": summary,
        "scores": [to_dict(score) for score in scores],
    }
    write_json_document(score_path, payload)
    row: dict[str, Any] = {
        "condition": condition,
        "system": system,
        "system_kind": system_kind,
    }
    row.update(summary)
    return row


def _delta_rows(rows: list[dict[str, Any]], neutral_condition: str) -> list[dict[str, Any]]:
    by_key = {(row["system"], row["condition"]): row for row in rows}
    deltas: list[dict[str, Any]] = []
    for row in rows:
        if row["condition"] != "original":
            continue
        neutral = by_key.get((row["system"], neutral_condition))
        if not neutral:
            continue
        delta = {
            "system": row["system"],
            "original_condition": "original",
            "neutral_condition": neutral_condition,
            "n": int(row.get("n_scored", row.get("count", 0))),
            "n_deg": int(row.get("n_deg", 0)) + int(neutral.get("n_deg", 0)),
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
            delta[f"original_{metric}"] = row.get(metric, 0.0)
            delta[f"neutral_{metric}"] = neutral.get(metric, 0.0)
            delta[f"delta_{metric}"] = round(float(neutral.get(metric, 0.0)) - float(row.get(metric, 0.0)), 6)
        deltas.append(delta)
    return deltas


def _write_manifest(path: Path, args: argparse.Namespace, original: list[Episode], neutral: list[Episode]) -> None:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner": "scripts/run_contamination_stress.py",
        "configuration": {
            "domain": args.domain,
            "seeds": args.seeds,
            "models": args.models,
            "baselines": args.baselines,
            "transform": args.transform,
            "policy": args.policy,
            "prompt_style": args.prompt_style,
            "view_ablation": args.view_ablation,
            "temperature": args.temperature,
            "timeout_s": args.timeout_s,
            "num_ctx": args.num_ctx,
            "keep_alive": args.keep_alive,
            "think": args.think,
        },
        "N": len(original),
        "N_deg": sum(1 for episode in original if is_degenerate_intervention(episode)),
        "neutral_N": len(neutral),
        "neutral_N_deg": sum(1 for episode in neutral if is_degenerate_intervention(episode)),
        "intervention_epsilon": INTERVENTION_EPSILON,
        "transform_policy": "Public names are mapped to neutral identifiers while numeric traces, topology shape, replay losses, costs, and safety metadata are preserved.",
    }
    write_json_document(path, payload)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    write_csv_rows(path, rows, fieldnames=fieldnames)


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


if __name__ == "__main__":
    raise SystemExit(main())
