"""Baseline policies for CausalOpsBench."""

from __future__ import annotations

from collections import deque
import random
from typing import Protocol

from causalopsbench.domains import get_domain
from causalopsbench.metrics import graph_distance
from causalopsbench.schemas import Episode, Prediction


class Baseline(Protocol):
    name: str

    def predict(self, episode: Episode) -> Prediction:
        ...


class NoOpBaseline:
    name = "noop"

    def predict(self, episode: Episode) -> Prediction:
        return Prediction(
            episode_id=episode.episode_id,
            alarm_time=None,
            alarm_confidence=0.0,
            root_cause_topk=[],
            causal_path=[],
            evidence_spans=[],
            action_ids=[],
            action_confidence=0.0,
            postmortem="No intervention selected.",
            token_count=64,
            tool_calls=0,
            wall_time_s=0.1,
        )


class OracleBaseline:
    name = "oracle"

    def predict(self, episode: Episode) -> Prediction:
        truth = episode.ground_truth
        return Prediction(
            episode_id=episode.episode_id,
            alarm_time=truth.fault.start_time,
            alarm_confidence=0.99,
            root_cause_topk=[truth.fault.label],
            causal_path=list(truth.fault.root_cause_path),
            evidence_spans=list(truth.gold_evidence_spans),
            action_ids=list(truth.oracle_action_ids),
            action_confidence=0.99,
            postmortem="Oracle policy used hidden ground truth.",
            token_count=512,
            tool_calls=4,
            wall_time_s=1.0,
        )


class RandomBaseline:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def predict(self, episode: Episode) -> Prediction:
        labels = _public_candidate_labels(episode)
        self.rng.shuffle(labels)
        action = self.rng.choice(episode.actions).action_id if episode.actions else ""
        evidence_pool = [
            ev.span_id
            for obs in episode.observations
            for ev in obs.evidence
        ] + [ev.span_id for ev in episode.manuals]
        self.rng.shuffle(evidence_pool)
        return Prediction(
            episode_id=episode.episode_id,
            alarm_time=self.rng.randint(0, episode.duration - 1),
            alarm_confidence=self.rng.random(),
            root_cause_topk=labels[:3],
            causal_path=[],
            evidence_spans=evidence_pool[:2],
            action_ids=[action],
            action_confidence=self.rng.random(),
            postmortem="Random baseline.",
            token_count=800,
            tool_calls=6,
            wall_time_s=2.0,
        )


class ThresholdBaseline:
    """Simple classical baseline using first large sensor deviation."""

    name = "threshold"

    def predict(self, episode: Episode) -> Prediction:
        baselines = episode.observations[0].sensors
        first_alarm = None
        best_sensor = ""
        best_z = 0.0
        for frame in episode.observations:
            for sensor, baseline in baselines.items():
                scale = max(abs(baseline) * 0.08, 0.2)
                z_score = abs(frame.sensors[sensor] - baseline) / scale
                if z_score > best_z:
                    best_z = z_score
                    best_sensor = sensor
                if first_alarm is None and z_score >= 3.0:
                    first_alarm = frame.timestamp
                    best_sensor = sensor
                    best_z = z_score
                    break
            if first_alarm is not None:
                break

        component = infer_component_from_sensor(episode, best_sensor)
        labels = _candidate_labels(episode, component)
        evidence = _evidence_until(episode, first_alarm if first_alarm is not None else episode.duration)
        action_id = _action_for_component(episode, component)
        return Prediction(
            episode_id=episode.episode_id,
            alarm_time=first_alarm,
            alarm_confidence=min(0.95, 0.35 + best_z / 10.0),
            root_cause_topk=labels,
            causal_path=[component] if component else [],
            evidence_spans=evidence[:4],
            action_ids=[action_id] if action_id else [],
            action_confidence=0.62,
            postmortem=f"Threshold deviation on {best_sensor}.",
            token_count=400,
            tool_calls=2,
            wall_time_s=0.5,
        )


class TopologyRCABaseline:
    """Topology-aware classical RCA baseline inspired by graph diffusion methods."""

    name = "topology_rca"

    def predict(self, episode: Episode) -> Prediction:
        component_scores, sensor_scores, first_alarm = _component_anomaly_scores(episode)
        diffused = _diffuse_component_scores(episode, component_scores)
        ranked_components = [
            component
            for component, _ in sorted(diffused.items(), key=lambda item: item[1], reverse=True)
        ]
        component = ranked_components[0] if ranked_components else ""
        labels: list[str] = []
        for ranked in ranked_components:
            for label in _candidate_labels(episode, ranked):
                if label not in labels:
                    labels.append(label)
                if len(labels) >= 3:
                    break
            if len(labels) >= 3:
                break

        if not labels:
            labels = _public_candidate_labels(episode)[:3]

        evidence = _evidence_until(episode, first_alarm if first_alarm is not None else episode.duration)
        action_id = _nearest_safe_action(episode, component)
        path = _path_to_largest_symptom(episode, component, component_scores)
        confidence = min(0.96, 0.42 + max(diffused.values(), default=0.0) / 16.0)
        return Prediction(
            episode_id=episode.episode_id,
            alarm_time=first_alarm,
            alarm_confidence=confidence,
            root_cause_topk=labels[:3],
            causal_path=path,
            evidence_spans=evidence[:5],
            action_ids=[action_id] if action_id else [],
            action_confidence=min(0.92, 0.50 + max(sensor_scores.values(), default=0.0) / 18.0),
            postmortem=f"Topology RCA ranked {component} from diffused anomaly scores.",
            token_count=520,
            tool_calls=3,
            wall_time_s=0.7,
        )


def get_baseline(name: str, seed: int = 0) -> Baseline:
    normalized = name.strip().lower()
    if normalized == "noop":
        return NoOpBaseline()
    if normalized == "oracle":
        return OracleBaseline()
    if normalized == "random":
        return RandomBaseline(seed=seed)
    if normalized == "threshold":
        return ThresholdBaseline()
    if normalized == "topology_rca":
        return TopologyRCABaseline()
    raise ValueError("Unknown baseline. Choose one of: noop, oracle, random, threshold, topology_rca")


def infer_component_from_sensor(episode: Episode, sensor: str) -> str:
    tokens = sensor.replace("_", "-").replace(".", "-").split("-")
    components = {component for edge in episode.topology for component in edge}
    components.update(action.target_component for action in episode.actions)
    for component in sorted(components, key=len, reverse=True):
        component_tokens = component.split("-")
        if any(token in tokens for token in component_tokens):
            return component
    return episode.actions[0].target_component if episode.actions else ""


def _candidate_labels(episode: Episode, component: str) -> list[str]:
    labels = [
        label
        for label in _public_candidate_labels(episode)
        if _component_from_label(label) == component
    ]
    for action in episode.actions:
        for label in action.expected_faults:
            if label.endswith(":override"):
                continue
            if label not in labels:
                labels.append(label)
        generic = f"{action.target_component}:generic"
        if action.target_component == component and generic not in labels:
            labels.append(generic)
        if len(labels) >= 3:
            break
    return labels[:3]


def _public_candidate_labels(episode: Episode) -> list[str]:
    labels: list[str] = []
    for action in episode.actions:
        for label in action.expected_faults:
            if label.endswith(":override"):
                continue
            if label not in labels:
                labels.append(label)
        generic = f"{action.target_component}:generic"
        if generic not in labels:
            labels.append(generic)
    return labels


def _component_from_label(label: str) -> str:
    return label.split(":", 1)[0]


def _component_anomaly_scores(episode: Episode) -> tuple[dict[str, float], dict[str, float], int | None]:
    template = get_domain(episode.domain)
    baselines = episode.observations[0].sensors
    component_scores = {component: 0.0 for component in template.components}
    sensor_scores: dict[str, float] = {}
    first_alarm = None
    for frame in episode.observations:
        for sensor, baseline in baselines.items():
            scale = max(abs(baseline) * 0.08, 0.2)
            z_score = abs(frame.sensors[sensor] - baseline) / scale
            sensor_scores[sensor] = max(sensor_scores.get(sensor, 0.0), z_score)
            component = template.sensor_components.get(sensor)
            if component:
                component_scores[component] = max(component_scores.get(component, 0.0), z_score)
            if first_alarm is None and z_score >= 3.0:
                first_alarm = frame.timestamp
    return component_scores, sensor_scores, first_alarm


def _diffuse_component_scores(episode: Episode, component_scores: dict[str, float]) -> dict[str, float]:
    components = set(component_scores)
    for left, right in episode.topology:
        components.add(left)
        components.add(right)
    diffused = {component: component_scores.get(component, 0.0) for component in components}
    for candidate in components:
        for symptom, score in component_scores.items():
            if candidate == symptom or score <= 0:
                continue
            directed_distance = _directed_distance(episode.topology, candidate, symptom)
            if directed_distance is not None:
                diffused[candidate] += score * (0.48 ** directed_distance)
    return diffused


def _directed_distance(edges: list[tuple[str, str]], start: str, target: str) -> int | None:
    if start == target:
        return 0
    adjacency: dict[str, set[str]] = {}
    for left, right in edges:
        adjacency.setdefault(left, set()).add(right)
    frontier: deque[tuple[str, int]] = deque([(start, 0)])
    seen = {start}
    while frontier:
        node, distance = frontier.popleft()
        for neighbor in adjacency.get(node, set()):
            if neighbor == target:
                return distance + 1
            if neighbor not in seen:
                seen.add(neighbor)
                frontier.append((neighbor, distance + 1))
    return None


def _nearest_safe_action(episode: Episode, component: str) -> str:
    safe_actions = [action for action in episode.actions if action.action_type == "safe_mitigation"]
    if not safe_actions:
        return ""
    return min(
        safe_actions,
        key=lambda action: graph_distance(episode.topology, component, action.target_component),
    ).action_id


def _path_to_largest_symptom(
    episode: Episode,
    component: str,
    component_scores: dict[str, float],
) -> list[str]:
    if not component:
        return []
    target = max(component_scores.items(), key=lambda item: item[1])[0] if component_scores else component
    if component == target:
        return [component]
    path = _directed_shortest_path(episode.topology, component, target)
    return path or [component]


def _directed_shortest_path(edges: list[tuple[str, str]], start: str, target: str) -> list[str] | None:
    adjacency: dict[str, set[str]] = {}
    for left, right in edges:
        adjacency.setdefault(left, set()).add(right)
    frontier: deque[tuple[str, list[str]]] = deque([(start, [start])])
    seen = {start}
    while frontier:
        node, path = frontier.popleft()
        for neighbor in adjacency.get(node, set()):
            if neighbor == target:
                return path + [neighbor]
            if neighbor not in seen:
                seen.add(neighbor)
                frontier.append((neighbor, path + [neighbor]))
    return None


def _action_for_component(episode: Episode, component: str) -> str:
    safe_actions = [action for action in episode.actions if action.action_type == "safe_mitigation"]
    for action in safe_actions:
        if action.target_component == component:
            return action.action_id
    return safe_actions[0].action_id if safe_actions else ""


def _evidence_until(episode: Episode, timestamp: int) -> list[str]:
    spans = [
        evidence.span_id
        for frame in episode.observations
        if frame.timestamp <= timestamp + 5
        for evidence in frame.evidence
    ]
    spans.extend(manual.span_id for manual in episode.manuals)
    return spans
