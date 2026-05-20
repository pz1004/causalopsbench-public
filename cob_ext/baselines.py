"""External validation baselines."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Protocol

from causalopsbench.metrics import clip
from cob_ext.adapters.base import evidence_records_from_observations
from cob_ext.schemas import ExternalEpisode, ExternalPrediction


class ExternalBaseline(Protocol):
    name: str

    def predict(self, episode: ExternalEpisode) -> ExternalPrediction:
        ...


class ExternalThresholdBaseline:
    name = "threshold"

    def predict(self, episode: ExternalEpisode) -> ExternalPrediction:
        component_scores, first_alarm, best_sensor = _component_anomaly_scores(episode)
        component = _best_component(component_scores, episode)
        topk = _candidate_labels(episode, component)
        evidence = _evidence_for_component(episode, component)[:5]
        confidence = min(0.95, 0.30 + max(component_scores.values(), default=0.0) / 10.0)
        return ExternalPrediction(
            episode_id=episode.episode_id,
            alarm_time=first_alarm,
            alarm_confidence=confidence,
            root_cause_topk=topk,
            evidence_spans=evidence,
            action_ids=_actions_for_component(episode, component)[:1],
            action_confidence=0.55,
            postmortem=f"External threshold deviation on {best_sensor}.",
            token_count=300,
            tool_calls=1,
            wall_time_s=0.2,
            parse_success=True,
        )


class ExternalTopologyHeuristicBaseline:
    name = "topology_heuristic"

    def predict(self, episode: ExternalEpisode) -> ExternalPrediction:
        component_scores, first_alarm, best_sensor = _component_anomaly_scores(episode)
        diffused = _diffuse_scores(episode.topology, component_scores)
        component = _best_component(diffused, episode)
        ranked = [item[0] for item in sorted(diffused.items(), key=lambda item: item[1], reverse=True)]
        topk: list[str] = []
        for candidate in ranked or [component]:
            for label in _candidate_labels(episode, candidate):
                if label not in topk:
                    topk.append(label)
                if len(topk) >= 3:
                    break
            if len(topk) >= 3:
                break
        if not topk:
            topk = _candidate_labels(episode, component)
        evidence = _evidence_for_component(episode, component)[:5]
        confidence = min(0.97, 0.35 + max(diffused.values(), default=0.0) / 12.0)
        return ExternalPrediction(
            episode_id=episode.episode_id,
            alarm_time=first_alarm,
            alarm_confidence=confidence,
            root_cause_topk=topk[:3],
            evidence_spans=evidence,
            action_ids=_actions_for_component(episode, component)[:1],
            action_confidence=0.62,
            postmortem=f"External topology heuristic ranked {component} from {best_sensor}.",
            token_count=420,
            tool_calls=2,
            wall_time_s=0.3,
            parse_success=True,
        )


def get_baseline(name: str) -> ExternalBaseline:
    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"threshold", "zscore", "z_score"}:
        return ExternalThresholdBaseline()
    if normalized in {"topology", "topology_heuristic", "topology_rca"}:
        return ExternalTopologyHeuristicBaseline()
    raise ValueError("Unknown external baseline. Choose one of: threshold, topology_heuristic")


def _component_anomaly_scores(episode: ExternalEpisode) -> tuple[dict[str, float], int | None, str]:
    frames = list(episode.observations.get("sensors", []))
    if not frames:
        return {}, None, ""
    baseline = dict(frames[0].get("values", {}))
    component_scores: dict[str, float] = defaultdict(float)
    best_sensor = ""
    best_score = 0.0
    first_alarm = None
    for frame in frames:
        values = frame.get("values", {})
        for sensor, start_value in baseline.items():
            if sensor not in values:
                continue
            scale = max(abs(float(start_value)) * 0.08, 0.2)
            score = abs(float(values[sensor]) - float(start_value)) / scale
            component = _component_for_sensor(episode, sensor)
            component_scores[component] = max(component_scores[component], score)
            if score > best_score:
                best_sensor = sensor
                best_score = score
            if first_alarm is None and score >= 3.0:
                first_alarm = int(frame.get("timestamp", 0))
    return dict(component_scores), first_alarm, best_sensor


def _diffuse_scores(topology: list[tuple[str, str]], scores: dict[str, float]) -> dict[str, float]:
    diffused = defaultdict(float, scores)
    adjacency: dict[str, set[str]] = defaultdict(set)
    for left, right in topology:
        adjacency[left].add(right)
        adjacency[right].add(left)
    for component, score in scores.items():
        queue: deque[tuple[str, int]] = deque([(component, 0)])
        seen = {component}
        while queue:
            node, distance = queue.popleft()
            if distance:
                diffused[node] += score * (0.55 ** distance)
            if distance >= 2:
                continue
            for neighbor in adjacency.get(node, set()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, distance + 1))
    return dict(diffused)


def _component_for_sensor(episode: ExternalEpisode, sensor: str) -> str:
    mapping = episode.metadata.get("sensor_components")
    if isinstance(mapping, dict) and sensor in mapping:
        return str(mapping[sensor])
    components = {node for edge in episode.topology for node in edge}
    components.update(action.target_component for action in episode.candidate_actions)
    normalized = sensor.lower().replace("_", "-").replace(".", "-")
    for component in sorted(components, key=len, reverse=True):
        tokens = component.lower().replace("_", "-").split("-")
        if any(token and token in normalized for token in tokens):
            return component
    if "." in sensor:
        return sensor.split(".", 1)[0]
    return next(iter(sorted(components)), "")


def _best_component(scores: dict[str, float], episode: ExternalEpisode) -> str:
    if scores:
        return max(scores.items(), key=lambda item: item[1])[0]
    components = {node for edge in episode.topology for node in edge}
    components.update(action.target_component for action in episode.candidate_actions)
    return next(iter(sorted(components)), "")


def _candidate_labels(episode: ExternalEpisode, component: str) -> list[str]:
    labels = [
        str(label)
        for label in episode.metadata.get("candidate_root_causes", [])
        if str(label).split(":", 1)[0] == component
    ]
    if not labels and component:
        for label in episode.metadata.get("candidate_root_causes", []):
            if component in str(label):
                labels.append(str(label))
    if not labels and component:
        labels.append(f"{component}:unknown")
    for label in episode.metadata.get("candidate_root_causes", []):
        if str(label) not in labels:
            labels.append(str(label))
        if len(labels) >= 3:
            break
    return labels[:3]


def _evidence_for_component(episode: ExternalEpisode, component: str) -> list[str]:
    records = evidence_records_from_observations(episode.observations)
    component_records = [record.span_id for record in records if record.component == component]
    if component_records:
        return component_records
    return [record.span_id for record in records]


def _actions_for_component(episode: ExternalEpisode, component: str) -> list[str]:
    return [
        action.action_id
        for action in episode.candidate_actions
        if action.target_component == component
    ]
