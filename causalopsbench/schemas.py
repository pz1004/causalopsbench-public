"""Serializable benchmark schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from causalopsbench._io import dumps_sorted_json, load_jsonl, to_jsonable, write_jsonl as _write_jsonl


@dataclass(frozen=True)
class EvidenceRecord:
    span_id: str
    source_id: str
    kind: str
    timestamp: int | None
    text: str


@dataclass(frozen=True)
class ObservationFrame:
    timestamp: int
    sensors: dict[str, float]
    evidence: list[EvidenceRecord] = field(default_factory=list)


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    target_component: str
    action_type: str
    cost: float
    safety_risk: float
    expected_faults: list[str]
    description: str


@dataclass(frozen=True)
class FaultSpec:
    component: str
    fault_type: str
    start_time: int
    severity: float
    root_cause_path: list[str]

    @property
    def label(self) -> str:
        return f"{self.component}:{self.fault_type}"


@dataclass(frozen=True)
class GroundTruth:
    fault: FaultSpec
    gold_evidence_spans: list[str]
    oracle_action_ids: list[str]
    noop_loss: float
    oracle_loss: float
    safety_critical_actions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Episode:
    episode_id: str
    domain: str
    duration: int
    topology: list[tuple[str, str]]
    manuals: list[EvidenceRecord]
    observations: list[ObservationFrame]
    actions: list[ActionSpec]
    ground_truth: GroundTruth
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Prediction:
    episode_id: str
    alarm_time: int | None
    alarm_confidence: float
    root_cause_topk: list[str]
    causal_path: list[str]
    evidence_spans: list[str]
    action_ids: list[str]
    action_confidence: float
    postmortem: str = ""
    token_count: int = 0
    tool_calls: int = 0
    wall_time_s: float = 0.0
    compute_joules: float = 0.0


@dataclass(frozen=True)
class ScoreBreakdown:
    episode_id: str
    detection: float
    root_cause: float
    intervention: float
    evidence: float
    calibration: float
    efficiency: float
    composite: float
    policy_loss: float
    safety_violations: int
    notes: list[str] = field(default_factory=list)


def _dataclass_from_dict(cls: type[Any], data: dict[str, Any]) -> Any:
    if cls is EvidenceRecord:
        return EvidenceRecord(**data)
    if cls is ObservationFrame:
        return ObservationFrame(
            timestamp=data["timestamp"],
            sensors={str(k): float(v) for k, v in data["sensors"].items()},
            evidence=[EvidenceRecord(**item) for item in data.get("evidence", [])],
        )
    if cls is ActionSpec:
        return ActionSpec(**data)
    if cls is FaultSpec:
        return FaultSpec(**data)
    if cls is GroundTruth:
        return GroundTruth(
            fault=_dataclass_from_dict(FaultSpec, data["fault"]),
            gold_evidence_spans=list(data["gold_evidence_spans"]),
            oracle_action_ids=list(data["oracle_action_ids"]),
            noop_loss=float(data["noop_loss"]),
            oracle_loss=float(data["oracle_loss"]),
            safety_critical_actions=list(data.get("safety_critical_actions", [])),
        )
    if cls is Episode:
        return Episode(
            episode_id=data["episode_id"],
            domain=data["domain"],
            duration=int(data["duration"]),
            topology=[tuple(edge) for edge in data["topology"]],
            manuals=[EvidenceRecord(**item) for item in data.get("manuals", [])],
            observations=[
                _dataclass_from_dict(ObservationFrame, item)
                for item in data.get("observations", [])
            ],
            actions=[ActionSpec(**item) for item in data.get("actions", [])],
            ground_truth=_dataclass_from_dict(GroundTruth, data["ground_truth"]),
            metadata=dict(data.get("metadata", {})),
        )
    if cls is Prediction:
        return Prediction(**data)
    if cls is ScoreBreakdown:
        return ScoreBreakdown(**data)
    raise TypeError(f"Unsupported dataclass: {cls!r}")


def to_dict(value: Any) -> Any:
    """Convert dataclasses recursively into JSON-compatible dictionaries."""
    return to_jsonable(value)


def episode_from_dict(data: dict[str, Any]) -> Episode:
    return _dataclass_from_dict(Episode, data)


def prediction_from_dict(data: dict[str, Any]) -> Prediction:
    return _dataclass_from_dict(Prediction, data)


def score_from_dict(data: dict[str, Any]) -> ScoreBreakdown:
    return _dataclass_from_dict(ScoreBreakdown, data)


def dumps_json(value: Any) -> str:
    return dumps_sorted_json(value)


def load_episodes_jsonl(path: str | Path) -> list[Episode]:
    return load_jsonl(path, episode_from_dict)


def load_predictions_jsonl(path: str | Path) -> list[Prediction]:
    return load_jsonl(path, prediction_from_dict)


def write_jsonl(path: str | Path, items: Iterable[Any]) -> None:
    _write_jsonl(path, items)
