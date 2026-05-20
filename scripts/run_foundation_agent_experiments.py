#!/usr/bin/env python3
"""Run foundation-agent baselines for CausalOpsBench.

The runner keeps the benchmark contract simple: each model receives a compact
public episode view and must emit the standard CausalOpsBench prediction JSON.
No hidden ground truth is included in the prompt.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causalopsbench._io import write_csv_rows, write_json_document
from causalopsbench.evaluate import evaluate_predictions
from causalopsbench.metrics import INTERVENTION_EPSILON, is_degenerate_intervention
from causalopsbench.schemas import (
    Episode,
    Prediction,
    load_episodes_jsonl,
    to_dict,
    write_jsonl,
)
from cob_ext import model_client

_MODEL_CLIENT_POST_JSON = model_client.post_json


SYSTEM_PROMPT = """You are an operational incident-response agent.
Use only the public episode fields supplied in the prompt. Do not assume hidden
ground truth. Return exactly one JSON object matching this schema:
{
  "episode_id": string,
  "alarm_time": integer or null,
  "alarm_confidence": number between 0 and 1,
  "root_cause_topk": [string],
  "causal_path": [string],
  "evidence_spans": [string],
  "action_ids": [string],
  "action_confidence": number between 0 and 1,
  "postmortem": string
}
Root-cause labels must have the form component:fault-type. Use only evidence
span IDs and action IDs visible in the prompt. Prefer safe mitigations unless
the prompt provides strong evidence that a risky override is necessary."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate ReAct JSON agents on generated CausalOpsBench episodes."
    )
    parser.add_argument("--episodes-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--policy", default="react-json", choices=["react-json"])
    parser.add_argument(
        "--prompt-style",
        default="react-json",
        choices=["react-json", "terse-json", "react-verbose", "evidence-first"],
        help="Prompt wording variant for prompt-sensitivity experiments.",
    )
    parser.add_argument(
        "--view-ablation",
        default="none",
        choices=["none", "no-evidence", "no-topology", "no-manuals"],
        help="Remove one public prompt component for ReAct ablation studies.",
    )
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama host URL for ollama: model specs.",
    )
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--keep-alive", default="5m")
    parser.add_argument("--think", type=_parse_bool, default=False)
    args = parser.parse_args(argv)

    if args.max_steps <= 0:
        raise SystemExit("--max-steps must be positive")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive when provided")
    if args.num_ctx <= 0:
        raise SystemExit("--num-ctx must be positive")
    _validate_credentials(args.models)

    episodes = _load_episodes(Path(args.episodes_dir), limit=args.limit)
    if not episodes:
        raise SystemExit("No episodes found")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    (output_dir / "scores").mkdir(exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
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
        predictions: list[Prediction] = []
        raw_records: list[dict[str, Any]] = []
        for episode in episodes:
            prediction, raw = runner.predict(episode)
            predictions.append(prediction)
            raw_records.append(raw)

        slug = _slug(runner.display_name)
        prediction_path = output_dir / "predictions" / f"{slug}.jsonl"
        raw_path = output_dir / "predictions" / f"{slug}_raw.jsonl"
        score_path = output_dir / "scores" / f"{slug}.json"
        write_jsonl(prediction_path, predictions)
        raw_path.write_text(
            "\n".join(json.dumps(record, sort_keys=True) for record in raw_records) + "\n",
            encoding="utf-8",
        )

        scores, summary = evaluate_predictions(episodes, predictions)
        payload = {
            "baseline": runner.display_name,
            "model_spec": model_spec,
            "policy": args.policy,
            "count": len(predictions),
            "summary": summary,
            "scores": [to_dict(score) for score in scores],
        }
        write_json_document(score_path, payload)
        row = {"baseline": runner.display_name, "model_spec": model_spec}
        row.update(summary)
        summary_rows.append(row)

    _write_summary(output_dir / "summary.csv", summary_rows)
    _write_manifest(
        output_dir / "experiment_manifest.json",
        args,
        len(episodes),
        sum(1 for episode in episodes if is_degenerate_intervention(episode)),
        summary_rows,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "episodes": len(episodes),
                "models": args.models,
                "summary_csv": str(output_dir / "summary.csv"),
            },
            sort_keys=True,
        )
    )
    return 0


class FoundationAgent:
    def __init__(
        self,
        model_spec: str,
        policy: str,
        prompt_style: str,
        view_ablation: str,
        max_steps: int,
        temperature: float,
        timeout_s: float,
        ollama_host: str = "http://localhost:11434",
        num_ctx: int = 8192,
        keep_alive: str = "5m",
        think: bool = False,
    ) -> None:
        self.provider, self.model = _parse_model_spec(model_spec)
        self.policy = policy
        self.prompt_style = prompt_style
        self.view_ablation = view_ablation
        self.max_steps = max_steps
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.ollama_host = ollama_host.rstrip("/")
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.think = think
        self.display_name = _display_name(self.provider, self.model, policy)

    def predict(self, episode: Episode) -> tuple[Prediction, dict[str, Any]]:
        prompt = _episode_prompt(
            episode,
            self.max_steps,
            self.view_ablation,
            prompt_style=self.prompt_style,
        )
        started = time.time()
        raw_text, call_metadata = self._call_model(prompt)
        elapsed = time.time() - started

        prediction_data, parse_status = _extract_prediction_data_with_status(raw_text)
        prediction = _coerce_prediction(
            episode=episode,
            data=prediction_data,
            wall_time_s=elapsed,
            tool_calls=self.max_steps,
            raw_text=raw_text,
            token_count=_metadata_token_count(call_metadata),
            view_ablation=self.view_ablation,
        )
        raw = {
            "episode_id": episode.episode_id,
            "baseline": self.display_name,
            "provider": self.provider,
            "model": self.model,
            "raw_text": raw_text,
            "wall_time_s": round(elapsed, 4),
            "view_ablation": self.view_ablation,
            "prompt_style": self.prompt_style,
            "json_parse_status": parse_status,
            "call_metadata": call_metadata,
        }
        return prediction, raw

    def _call_model(self, prompt: str) -> tuple[str, dict[str, Any]]:
        if self.provider == "openai":
            return (
                _call_openai(
                    model=self.model,
                    system=SYSTEM_PROMPT,
                    prompt=prompt,
                    temperature=self.temperature,
                    timeout_s=self.timeout_s,
                ),
                {},
            )
        if self.provider == "anthropic":
            return (
                _call_anthropic(
                    model=self.model,
                    system=SYSTEM_PROMPT,
                    prompt=prompt,
                    temperature=self.temperature,
                    timeout_s=self.timeout_s,
                ),
                {},
            )
        if self.provider == "ollama":
            return _call_ollama(
                model=self.model,
                system=SYSTEM_PROMPT,
                prompt=prompt,
                temperature=self.temperature,
                timeout_s=self.timeout_s,
                host=self.ollama_host,
                num_ctx=self.num_ctx,
                keep_alive=self.keep_alive,
                think=self.think,
            )
        raise ValueError(f"Unsupported provider: {self.provider}")


def _load_episodes(path: Path, limit: int | None) -> list[Episode]:
    episodes: list[Episode] = []
    for jsonl_path in sorted(path.glob("*.jsonl")):
        episodes.extend(load_episodes_jsonl(jsonl_path))
        if limit is not None and len(episodes) >= limit:
            return episodes[:limit]
    return episodes


def _episode_prompt(
    episode: Episode,
    max_steps: int,
    view_ablation: str = "none",
    prompt_style: str = "react-json",
) -> str:
    payload = _public_episode_view(episode, view_ablation=view_ablation)
    ablation_note = "" if view_ablation == "none" else f" View ablation: {view_ablation}."
    payload_text = json.dumps(payload, sort_keys=True)
    if prompt_style == "terse-json":
        return (
            "Return only the prediction JSON. Use visible span IDs, action IDs, "
            f"and component:fault labels. Max internal steps: {max_steps}.{ablation_note}\n\n"
            f"{payload_text}"
        )
    if prompt_style == "react-verbose":
        return (
            f"Policy: react-json. Use at most {max_steps} internal steps: "
            "inspect_sensor_summary, retrieve_evidence, map_topology, select_action. "
            "Internally compare symptoms, evidence, topology, and intervention cost before answering. "
            f"Then emit only the final JSON prediction.{ablation_note}\n\n"
            f"Public episode view:\n{payload_text}"
        )
    if prompt_style == "evidence-first":
        return (
            f"Policy: evidence-first react-json. First identify the strongest visible evidence spans, "
            "then infer the root cause, propagation path, alarm time, and safest useful action. "
            f"Use at most {max_steps} internal steps and emit only the final JSON prediction."
            f"{ablation_note}\n\nPublic episode view:\n{payload_text}"
        )
    return (
        f"Policy: react-json. Use at most {max_steps} internal steps: "
        "inspect_sensor_summary, retrieve_evidence, map_topology, select_action. "
        f"Then emit only the final JSON prediction.{ablation_note}\n\n"
        f"Public episode view:\n{payload_text}"
    )


def _public_episode_view(episode: Episode, view_ablation: str = "none") -> dict[str, Any]:
    topology = [] if view_ablation == "no-topology" else episode.topology
    return {
        "episode_id": episode.episode_id,
        "domain": episode.domain,
        "duration": episode.duration,
        "topology": topology,
        "sensor_summary": _sensor_summary(episode),
        "sensor_components": _sensor_components(episode),
        "manuals": [] if view_ablation == "no-manuals" else _manual_records(episode),
        "evidence": [] if view_ablation == "no-evidence" else _observation_evidence_records(episode),
        "actions": [
            {
                "action_id": action.action_id,
                "target_component": action.target_component,
                "action_type": action.action_type,
                "cost": action.cost,
                "safety_risk": action.safety_risk,
                "expected_faults": action.expected_faults,
                "description": action.description,
            }
            for action in episode.actions
        ],
        "valid_components": sorted(
            {component for edge in topology for component in edge}
            | {action.target_component for action in episode.actions}
        ),
    }


def _sensor_summary(episode: Episode) -> list[dict[str, Any]]:
    if not episode.observations:
        return []
    baseline = episode.observations[0].sensors
    summary: list[dict[str, Any]] = []
    for sensor in sorted(baseline):
        values = [(frame.timestamp, frame.sensors[sensor]) for frame in episode.observations]
        base = baseline[sensor]
        peak_t, peak_v = max(values, key=lambda item: abs(item[1] - base))
        final_v = values[-1][1]
        summary.append(
            {
                "sensor": sensor,
                "initial": round(base, 4),
                "final": round(final_v, 4),
                "min": round(min(value for _, value in values), 4),
                "max": round(max(value for _, value in values), 4),
                "peak_abs_delta": round(peak_v - base, 4),
                "peak_time": peak_t,
            }
        )
    return summary


def _sensor_components(episode: Episode) -> dict[str, str]:
    metadata_components = episode.metadata.get("sensor_components")
    if isinstance(metadata_components, dict):
        return {str(key): str(value) for key, value in metadata_components.items()}
    try:
        from causalopsbench.domains import get_domain

        return dict(get_domain(episode.domain).sensor_components)
    except Exception:
        return {}


def _manual_records(episode: Episode) -> list[dict[str, Any]]:
    return [
        {
            "span_id": manual.span_id,
            "source_id": manual.source_id,
            "kind": manual.kind,
            "timestamp": manual.timestamp,
            "text": manual.text,
        }
        for manual in episode.manuals
    ]


def _observation_evidence_records(episode: Episode) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for frame in episode.observations:
        for evidence in frame.evidence:
            records.append(
                {
                    "span_id": evidence.span_id,
                    "source_id": evidence.source_id,
                    "kind": evidence.kind,
                    "timestamp": evidence.timestamp,
                    "text": evidence.text,
                }
            )
    return records


def _evidence_records(episode: Episode, view_ablation: str = "none") -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if view_ablation != "no-manuals":
        records.extend(_manual_records(episode))
    if view_ablation != "no-evidence":
        records.extend(_observation_evidence_records(episode))
    return records


def _with_local_post_json(call: Any) -> Any:
    original = model_client.post_json
    model_client.post_json = _post_json
    try:
        return call()
    finally:
        model_client.post_json = original


def _call_openai(
    model: str,
    system: str,
    prompt: str,
    temperature: float,
    timeout_s: float,
) -> str:
    return _with_local_post_json(
        lambda: model_client.call_openai(
            model=model,
            system=system,
            prompt=prompt,
            temperature=temperature,
            timeout_s=timeout_s,
        )
    )


def _call_anthropic(
    model: str,
    system: str,
    prompt: str,
    temperature: float,
    timeout_s: float,
) -> str:
    return _with_local_post_json(
        lambda: model_client.call_anthropic(
            model=model,
            system=system,
            prompt=prompt,
            temperature=temperature,
            timeout_s=timeout_s,
        )
    )


def _call_ollama(
    model: str,
    system: str,
    prompt: str,
    temperature: float,
    timeout_s: float,
    host: str,
    num_ctx: int,
    keep_alive: str,
    think: bool,
) -> tuple[str, dict[str, Any]]:
    return _with_local_post_json(
        lambda: model_client.call_ollama(
            model=model,
            system=system,
            prompt=prompt,
            temperature=temperature,
            timeout_s=timeout_s,
            host=host,
            num_ctx=num_ctx,
            keep_alive=keep_alive,
            think=think,
        )
    )


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
    return _MODEL_CLIENT_POST_JSON(url, payload, headers, timeout_s)


def _collect_text(value: Any) -> str:
    return model_client.collect_text(value)


def _extract_prediction_data(text: str) -> dict[str, Any]:
    parsed, _ = _extract_prediction_data_with_status(text)
    return parsed


def _extract_prediction_data_with_status(text: str) -> tuple[dict[str, Any], str]:
    return model_client.extract_prediction_data_with_status(text)


def _coerce_prediction(
    episode: Episode,
    data: dict[str, Any],
    wall_time_s: float,
    tool_calls: int,
    raw_text: str,
    token_count: int | None = None,
    view_ablation: str = "none",
) -> Prediction:
    valid_spans = {
        record["span_id"]
        for record in _evidence_records(episode, view_ablation=view_ablation)
    }
    valid_actions = {action.action_id for action in episode.actions}
    alarm_time = data.get("alarm_time")
    if alarm_time is not None:
        try:
            alarm_time = int(alarm_time)
        except (TypeError, ValueError):
            alarm_time = None
        if alarm_time is not None and not 0 <= alarm_time < episode.duration:
            alarm_time = None

    postmortem = str(data.get("postmortem") or "").strip()
    if not data:
        postmortem = "Invalid or empty model JSON output."

    return Prediction(
        episode_id=episode.episode_id,
        alarm_time=alarm_time,
        alarm_confidence=_prob(data.get("alarm_confidence")),
        root_cause_topk=_str_list(data.get("root_cause_topk"))[:3],
        causal_path=_str_list(data.get("causal_path")),
        evidence_spans=[span for span in _str_list(data.get("evidence_spans")) if span in valid_spans],
        action_ids=[action for action in _str_list(data.get("action_ids")) if action in valid_actions],
        action_confidence=_prob(data.get("action_confidence")),
        postmortem=postmortem[:1000],
        token_count=token_count if token_count is not None and token_count > 0 else max(1, len(raw_text) // 4),
        tool_calls=tool_calls,
        wall_time_s=round(wall_time_s, 4),
    )


def _prob(value: Any) -> float:
    return model_client.prob(value)


def _str_list(value: Any) -> list[str]:
    return model_client.str_list(value)


def _parse_model_spec(model_spec: str) -> tuple[str, str]:
    return model_client.parse_model_spec(model_spec)


def _validate_credentials(model_specs: list[str]) -> None:
    model_client.validate_credentials(model_specs)


def _display_name(provider: str, model: str, policy: str) -> str:
    return model_client.display_name(provider, model, policy)


def _metadata_token_count(metadata: dict[str, Any]) -> int | None:
    return model_client.metadata_token_count(metadata)


def _int_or_none(value: Any) -> int | None:
    return model_client._int_or_none(value)


def _parse_bool(value: Any) -> bool:
    return model_client.parse_bool(value)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv_rows(path, rows)


def _write_manifest(
    path: Path,
    args: argparse.Namespace,
    episode_count: int,
    degenerate_count: int,
    rows: list[dict[str, Any]],
) -> None:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner": "scripts/run_foundation_agent_experiments.py",
        "configuration": {
            "episodes_dir": args.episodes_dir,
            "models": args.models,
            "policy": args.policy,
            "prompt_style": args.prompt_style,
            "view_ablation": args.view_ablation,
            "max_steps": args.max_steps,
            "temperature": args.temperature,
            "seed": args.seed,
            "limit": args.limit,
            "timeout_s": args.timeout_s,
            "ollama_host": args.ollama_host,
            "num_ctx": args.num_ctx,
            "keep_alive": args.keep_alive,
            "think": args.think,
        },
        "episodes": episode_count,
        "N": episode_count,
        "N_deg": degenerate_count,
        "intervention_epsilon": INTERVENTION_EPSILON,
        "summary": rows,
        "claim_policy": "foundation-agent results; report only generated prediction artifacts.",
    }
    write_json_document(path, payload)


if __name__ == "__main__":
    raise SystemExit(main())
