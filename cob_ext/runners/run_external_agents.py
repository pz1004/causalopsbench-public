"""Run local or hosted JSON agents on external validation episodes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import resources
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from causalopsbench._io import write_csv_rows, write_json_document
from cob_ext import model_client
from cob_ext.scoring.score_external import evaluate_predictions
from cob_ext.schemas import (
    ExternalEpisode,
    ExternalPrediction,
    load_episodes_jsonl,
    to_dict,
    write_jsonl,
)

EXTERNAL_SYSTEM_PROMPT_FALLBACK = """You are an operational incident-response agent evaluating external public traces.
Use only the public fields supplied in the prompt. Public external datasets do
not provide CausalOpsBench no-op/oracle replay trajectories, so replay
intervention scoring is disabled. Return exactly one JSON object:
{
  "episode_id": string,
  "alarm_time": integer or null,
  "alarm_confidence": number between 0 and 1,
  "root_cause_topk": [string],
  "evidence_spans": [string],
  "action_ids": [string],
  "action_confidence": number between 0 and 1,
  "postmortem": string
}
Use only visible evidence span IDs and action IDs. Root-cause labels should use
component:fault-type when possible."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run external validation agents.")
    parser.add_argument("--episodes", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--policy", default="react-json", choices=["react-json"])
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--keep-alive", default="5m")
    parser.add_argument("--think", type=model_client.parse_bool, default=False)
    args = parser.parse_args(argv)

    model_client.validate_credentials(args.models)
    episodes = []
    for path in args.episodes:
        episodes.extend(load_episodes_jsonl(path))
    if args.limit is not None:
        episodes = episodes[: args.limit]
    if not episodes:
        raise SystemExit("No external episodes found")

    output_dir = Path(args.output_dir)
    prediction_dir = output_dir / "predictions"
    score_dir = output_dir / "scores"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for model_spec in args.models:
        runner = ExternalAgent(
            model_spec=model_spec,
            policy=args.policy,
            max_steps=args.max_steps,
            temperature=args.temperature,
            timeout_s=args.timeout_s,
            ollama_host=args.ollama_host,
            num_ctx=args.num_ctx,
            keep_alive=args.keep_alive,
            think=args.think,
        )
        predictions: list[ExternalPrediction] = []
        raw_records: list[dict[str, Any]] = []
        for episode in episodes:
            prediction, raw = runner.predict(episode)
            predictions.append(prediction)
            raw_records.append(raw)

        slug = _slug(runner.display_name)
        write_jsonl(prediction_dir / f"{slug}.jsonl", predictions)
        (prediction_dir / f"{slug}_raw.jsonl").write_text(
            "\n".join(json.dumps(record, sort_keys=True) for record in raw_records) + "\n",
            encoding="utf-8",
        )
        scores, summary = evaluate_predictions(episodes, predictions)
        write_json_document(
            score_dir / f"{slug}.json",
            {
                "baseline": runner.display_name,
                "model_spec": model_spec,
                "summary": summary,
                "scores": [to_dict(score) for score in scores],
            },
        )
        row = {"system": runner.display_name, "model_spec": model_spec}
        row.update(summary)
        rows.append(row)
    _write_summary(output_dir / "summary.csv", rows)
    _write_manifest(output_dir / "experiment_manifest.json", args, len(episodes), rows)
    print(json.dumps({"output_dir": str(output_dir), "episodes": len(episodes)}, sort_keys=True))
    return 0


class ExternalAgent:
    def __init__(
        self,
        model_spec: str,
        policy: str,
        max_steps: int,
        temperature: float,
        timeout_s: float,
        ollama_host: str,
        num_ctx: int,
        keep_alive: str,
        think: bool,
    ) -> None:
        self.provider, self.model = model_client.parse_model_spec(model_spec)
        self.model_spec = model_spec
        self.policy = policy
        self.max_steps = max_steps
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.ollama_host = ollama_host.rstrip("/")
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.think = think
        self.display_name = model_client.display_name(self.provider, self.model, policy)

    def predict(self, episode: ExternalEpisode) -> tuple[ExternalPrediction, dict[str, Any]]:
        prompt = _episode_prompt(episode, self.max_steps)
        started = time.time()
        try:
            raw_text, metadata = self._call_model(prompt)
            elapsed = time.time() - started
            call_error = ""
        except Exception as exc:
            elapsed = time.time() - started
            raw_text = ""
            metadata = {"error_type": type(exc).__name__, "error": str(exc)}
            call_error = f"{type(exc).__name__}: {exc}"
        parsed, status = model_client.extract_prediction_data_with_status(raw_text)
        if call_error:
            status = "call_error"
        prediction = _coerce_prediction(episode, parsed, elapsed, self.max_steps, raw_text, metadata, status)
        raw = {
            "episode_id": episode.episode_id,
            "baseline": self.display_name,
            "provider": self.provider,
            "model": self.model,
            "raw_text": raw_text,
            "wall_time_s": round(elapsed, 4),
            "json_parse_status": status,
            "call_error": call_error,
            "call_metadata": metadata,
        }
        return prediction, raw

    def _call_model(self, prompt: str) -> tuple[str, dict[str, Any]]:
        if self.provider == "ollama":
            return model_client.call_ollama(
                model=self.model,
                system=_system_prompt(),
                prompt=prompt,
                temperature=self.temperature,
                timeout_s=self.timeout_s,
                host=self.ollama_host,
                num_ctx=self.num_ctx,
                keep_alive=self.keep_alive,
                think=self.think,
            )
        if self.provider == "openai":
            return (
                model_client.call_openai(
                    model=self.model,
                    system=_system_prompt(),
                    prompt=prompt,
                    temperature=self.temperature,
                    timeout_s=self.timeout_s,
                ),
                {},
            )
        if self.provider == "anthropic":
            return (
                model_client.call_anthropic(
                    model=self.model,
                    system=_system_prompt(),
                    prompt=prompt,
                    temperature=self.temperature,
                    timeout_s=self.timeout_s,
                ),
                {},
            )
        raise ValueError(f"Unsupported provider: {self.provider}")


def _episode_prompt(episode: ExternalEpisode, max_steps: int) -> str:
    payload = {
        "episode_id": episode.episode_id,
        "source_dataset": episode.source_dataset,
        "domain": episode.domain,
        "duration": episode.duration,
        "topology": episode.topology,
        "observations": _compact_observations(episode.observations),
        "candidate_actions": [to_dict(action) for action in episode.candidate_actions],
        "supported_metrics": episode.supported_metrics,
        "metric_note": "External portability study: no replay intervention score is available.",
    }
    return (
        f"Policy: react-json. Use at most {max_steps} internal steps: inspect external sensors, "
        "retrieve evidence, map topology, and select an audit action. Emit only JSON.\n\n"
        f"Public external episode view:\n{json.dumps(payload, sort_keys=True)}"
    )


def _compact_observations(observations: dict[str, Any]) -> dict[str, Any]:
    return {
        "sensor_summary": _sensor_summary(observations.get("sensors", []), limit=80),
        "evidence": _span_records(observations.get("evidence", []), limit=60),
        "logs": _span_records(observations.get("logs", []), limit=40),
        "traces": _span_records(observations.get("traces", []), limit=40),
        "manuals": _span_records(observations.get("manuals", []), limit=20),
    }


def _sensor_summary(frames: Any, limit: int) -> list[dict[str, Any]]:
    if not isinstance(frames, list) or not frames:
        return []
    series: dict[str, list[tuple[int, float]]] = {}
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        timestamp = _int_or_default(frame.get("timestamp"), index)
        values = frame.get("values", {})
        if not isinstance(values, dict):
            continue
        for sensor, raw_value in values.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            series.setdefault(str(sensor), []).append((timestamp, value))

    summaries: list[dict[str, Any]] = []
    for sensor, values in series.items():
        if not values:
            continue
        initial = values[0][1]
        final = values[-1][1]
        min_value = min(value for _, value in values)
        max_value = max(value for _, value in values)
        peak_time, peak_value = max(values, key=lambda item: abs(item[1] - initial))
        summaries.append(
            {
                "sensor": sensor,
                "initial": round(initial, 6),
                "final": round(final, 6),
                "min": round(min_value, 6),
                "max": round(max_value, 6),
                "peak_abs_delta": round(peak_value - initial, 6),
                "peak_time": peak_time,
            }
        )
    summaries.sort(key=lambda item: abs(item["peak_abs_delta"]), reverse=True)
    return summaries[:limit]


def _span_records(records: Any, limit: int) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        return []
    compact: list[dict[str, Any]] = []
    for record in records[:limit]:
        if not isinstance(record, dict):
            continue
        compact.append(
            {
                "span_id": record.get("span_id"),
                "source_id": record.get("source_id"),
                "kind": record.get("kind"),
                "timestamp": record.get("timestamp"),
                "component": record.get("component"),
                "proxy": record.get("proxy", False),
                "text": str(record.get("text", ""))[:600],
            }
        )
    return compact


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _coerce_prediction(
    episode: ExternalEpisode,
    data: dict[str, Any],
    wall_time_s: float,
    tool_calls: int,
    raw_text: str,
    metadata: dict[str, Any],
    parse_status: str,
) -> ExternalPrediction:
    visible_spans = {
        str(item.get("span_id"))
        for records in episode.observations.values()
        if isinstance(records, list)
        for item in records
        if isinstance(item, dict) and item.get("span_id")
    }
    visible_actions = {action.action_id for action in episode.candidate_actions}
    alarm_time = data.get("alarm_time")
    if alarm_time is not None:
        try:
            alarm_time = int(alarm_time)
        except (TypeError, ValueError):
            alarm_time = None
    if alarm_time is not None and not 0 <= alarm_time <= max(0, episode.duration):
        alarm_time = None
    parse_success = parse_status in {"direct_json", "embedded_json"} and bool(data)
    postmortem = str(data.get("postmortem") or "").strip()
    if not data:
        error = metadata.get("error")
        postmortem = f"Invalid or empty model JSON output. {error}".strip()
    return ExternalPrediction(
        episode_id=episode.episode_id,
        alarm_time=alarm_time,
        alarm_confidence=model_client.prob(data.get("alarm_confidence")),
        root_cause_topk=model_client.str_list(data.get("root_cause_topk"))[:3],
        evidence_spans=[
            span for span in model_client.str_list(data.get("evidence_spans")) if span in visible_spans
        ],
        action_ids=[
            action for action in model_client.str_list(data.get("action_ids")) if action in visible_actions
        ],
        action_confidence=model_client.prob(data.get("action_confidence")),
        postmortem=postmortem[:1000],
        token_count=model_client.metadata_token_count(metadata) or max(1, len(raw_text) // 4),
        tool_calls=tool_calls,
        wall_time_s=round(wall_time_s, 4),
        parse_success=parse_success,
    )


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv_rows(path, rows)


def _write_manifest(path: Path, args: argparse.Namespace, episode_count: int, rows: list[dict[str, Any]]) -> None:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner": "cob_ext.runners.run_external_agents",
        "episode_count": episode_count,
        "configuration": {
            "episodes": args.episodes,
            "models": args.models,
            "policy": args.policy,
            "max_steps": args.max_steps,
            "temperature": args.temperature,
            "limit": args.limit,
            "timeout_s": args.timeout_s,
            "ollama_host": args.ollama_host,
            "num_ctx": args.num_ctx,
            "keep_alive": args.keep_alive,
            "think": args.think,
        },
        "summary": rows,
    }
    write_json_document(path, payload)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _system_prompt() -> str:
    try:
        return resources.files("cob_ext.prompts").joinpath("external_agent_prompt.txt").read_text(
            encoding="utf-8"
        ).strip()
    except (FileNotFoundError, ModuleNotFoundError):
        return EXTERNAL_SYSTEM_PROMPT_FALLBACK


if __name__ == "__main__":
    raise SystemExit(main())
