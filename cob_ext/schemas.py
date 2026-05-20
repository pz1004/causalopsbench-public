"""Schemas for public-dataset external trace validation.

External validation episodes intentionally do not carry CausalOpsBench replay
fields such as no-op loss, oracle loss, or oracle actions. Public datasets can
validate schema portability and diagnostic behavior, but not native replay
intervention value unless they provide matched intervention trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from causalopsbench._io import dumps_sorted_json, load_jsonl, to_jsonable, write_jsonl as _write_jsonl

UNSUPPORTED_REPLAY_METRICS = {
    "intervention",
    "intervention_improvement",
    "native_composite",
    "composite",
    "oracle_replay",
    "noop_replay",
}

DEFAULT_SUPPORTED_METRICS = [
    "detection",
    "root_cause_top1",
    "root_cause_topk",
    "evidence_f1",
    "calibration",
    "parsing_success",
    "runtime_efficiency",
    "external_portability_score",
]


@dataclass(frozen=True)
class ExternalEvidenceRecord:
    span_id: str
    source_id: str
    kind: str
    timestamp: int | None
    text: str
    component: str | None = None
    proxy: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExternalActionSpec:
    action_id: str
    target_component: str
    action_type: str
    description: str
    cost: float = 0.0
    safety_risk: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExternalGroundTruth:
    fault_component: str
    fault_type: str
    root_cause_label: str
    is_faulted: bool
    fault_start_time: int | None = None
    severity: float | None = None
    gold_evidence_spans: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExternalEpisode:
    episode_id: str
    source_dataset: str
    domain: str
    split: str
    duration: int
    topology: list[tuple[str, str]]
    observations: dict[str, Any]
    candidate_actions: list[ExternalActionSpec]
    ground_truth: ExternalGroundTruth
    supported_metrics: list[str] = field(default_factory=lambda: list(DEFAULT_SUPPORTED_METRICS))
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.duration < 0:
            raise ValueError(f"{self.episode_id}: duration must be non-negative")
        unsupported = set(self.supported_metrics) & UNSUPPORTED_REPLAY_METRICS
        if unsupported:
            listed = ", ".join(sorted(unsupported))
            raise ValueError(
                f"{self.episode_id}: external episodes cannot request replay metrics: {listed}"
            )


@dataclass(frozen=True)
class ExternalPrediction:
    episode_id: str
    alarm_time: int | None
    alarm_confidence: float
    root_cause_topk: list[str]
    evidence_spans: list[str]
    action_ids: list[str] = field(default_factory=list)
    action_confidence: float = 0.0
    postmortem: str = ""
    token_count: int = 0
    tool_calls: int = 0
    wall_time_s: float = 0.0
    parse_success: bool = True


@dataclass(frozen=True)
class ExternalScoreBreakdown:
    episode_id: str
    detection: float
    root_cause_top1: float
    root_cause_topk: float
    evidence_f1: float
    calibration: float
    parsing_success: float
    runtime_efficiency: float
    external_portability_score: float
    notes: list[str] = field(default_factory=list)


def to_dict(value: Any) -> Any:
    return to_jsonable(value)


def dumps_json(value: Any) -> str:
    return dumps_sorted_json(value)


def evidence_from_dict(data: dict[str, Any]) -> ExternalEvidenceRecord:
    return ExternalEvidenceRecord(
        span_id=str(data["span_id"]),
        source_id=str(data.get("source_id", "")),
        kind=str(data.get("kind", "evidence")),
        timestamp=_optional_int(data.get("timestamp")),
        text=str(data.get("text", "")),
        component=_optional_str(data.get("component")),
        proxy=bool(data.get("proxy", False)),
        metadata=dict(data.get("metadata", {})),
    )


def action_from_dict(data: dict[str, Any]) -> ExternalActionSpec:
    return ExternalActionSpec(
        action_id=str(data["action_id"]),
        target_component=str(data.get("target_component", "")),
        action_type=str(data.get("action_type", "investigate")),
        description=str(data.get("description", "")),
        cost=float(data.get("cost", 0.0)),
        safety_risk=float(data.get("safety_risk", 0.0)),
        metadata=dict(data.get("metadata", {})),
    )


def ground_truth_from_dict(data: dict[str, Any]) -> ExternalGroundTruth:
    return ExternalGroundTruth(
        fault_component=str(data.get("fault_component", "")),
        fault_type=str(data.get("fault_type", "")),
        root_cause_label=str(data.get("root_cause_label", "")),
        is_faulted=bool(data.get("is_faulted", True)),
        fault_start_time=_optional_int(data.get("fault_start_time")),
        severity=_optional_float(data.get("severity")),
        gold_evidence_spans=[str(item) for item in data.get("gold_evidence_spans", [])],
        metadata=dict(data.get("metadata", {})),
    )


def episode_from_dict(data: dict[str, Any]) -> ExternalEpisode:
    episode = ExternalEpisode(
        episode_id=str(data["episode_id"]),
        source_dataset=str(data["source_dataset"]),
        domain=str(data["domain"]),
        split=str(data.get("split", "external")),
        duration=int(data.get("duration", 0)),
        topology=[tuple(edge) for edge in data.get("topology", [])],
        observations=dict(data.get("observations", {})),
        candidate_actions=[action_from_dict(item) for item in data.get("candidate_actions", [])],
        ground_truth=ground_truth_from_dict(data["ground_truth"]),
        supported_metrics=[str(item) for item in data.get("supported_metrics", DEFAULT_SUPPORTED_METRICS)],
        metadata=dict(data.get("metadata", {})),
    )
    episode.validate()
    return episode


def prediction_from_dict(data: dict[str, Any]) -> ExternalPrediction:
    return ExternalPrediction(
        episode_id=str(data["episode_id"]),
        alarm_time=_optional_int(data.get("alarm_time")),
        alarm_confidence=float(data.get("alarm_confidence", 0.0)),
        root_cause_topk=[str(item) for item in data.get("root_cause_topk", [])],
        evidence_spans=[str(item) for item in data.get("evidence_spans", [])],
        action_ids=[str(item) for item in data.get("action_ids", [])],
        action_confidence=float(data.get("action_confidence", 0.0)),
        postmortem=str(data.get("postmortem", "")),
        token_count=int(data.get("token_count", 0)),
        tool_calls=int(data.get("tool_calls", 0)),
        wall_time_s=float(data.get("wall_time_s", 0.0)),
        parse_success=bool(data.get("parse_success", True)),
    )


def load_episodes_jsonl(path: str | Path) -> list[ExternalEpisode]:
    return load_jsonl(path, episode_from_dict)


def load_predictions_jsonl(path: str | Path) -> list[ExternalPrediction]:
    return load_jsonl(path, prediction_from_dict)


def write_jsonl(path: str | Path, items: Iterable[Any]) -> None:
    _write_jsonl(path, items)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
