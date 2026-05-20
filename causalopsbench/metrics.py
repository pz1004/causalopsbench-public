"""Scoring formulas for CausalOpsBench."""

from __future__ import annotations

from collections import deque
import math

from causalopsbench.schemas import Episode, Prediction, ScoreBreakdown
from causalopsbench.simulator import replay_prediction

INTERVENTION_EPSILON = 1e-9


def clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def detection_score(
    fault_time: int,
    alarm_time: int | None,
    false_alarms: int = 0,
    tau: float = 12.0,
    alpha: float = 0.08,
) -> float:
    if alarm_time is None:
        return 0.0
    delay = max(0, alarm_time - fault_time)
    return clip(math.exp(-delay / tau) - alpha * false_alarms)


def evidence_f1(predicted: list[str], gold: list[str]) -> float:
    predicted_set = set(predicted)
    gold_set = set(gold)
    if not predicted_set and not gold_set:
        return 1.0
    if not predicted_set or not gold_set:
        return 0.0
    tp = len(predicted_set & gold_set)
    precision = tp / len(predicted_set)
    recall = tp / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def root_cause_score(episode: Episode, prediction: Prediction) -> float:
    truth = episode.ground_truth.fault
    true_label = truth.label
    topk = prediction.root_cause_topk
    top1 = 1.0 if topk and topk[0] == true_label else 0.0
    top3 = 1.0 if true_label in topk[:3] else 0.0

    predicted_component = _component_from_label(topk[0]) if topk else None
    distance = graph_distance(
        episode.topology,
        predicted_component,
        truth.component,
    )
    diameter = max(1, graph_diameter(episode.topology))
    graph_term = 1.0 - min(distance, diameter) / diameter
    return clip(0.5 * top1 + 0.3 * top3 + 0.2 * graph_term)


def intervention_score(episode: Episode, policy_loss: float, safety_violations: int) -> float:
    truth = episode.ground_truth
    denominator = truth.noop_loss - truth.oracle_loss
    if denominator <= 0:
        base = 0.0
    else:
        base = clip((truth.noop_loss - policy_loss) / denominator)
    selected_cost = max(0.0, policy_loss - truth.oracle_loss)
    cost_penalty = min(0.15, 0.01 * selected_cost / max(1.0, truth.noop_loss))
    safety_penalty = 0.18 * safety_violations
    return clip(base - cost_penalty - safety_penalty)


def is_degenerate_intervention(episode: Episode, epsilon: float = INTERVENTION_EPSILON) -> bool:
    truth = episode.ground_truth
    return truth.noop_loss - truth.oracle_loss <= epsilon


def calibration_score(prediction: Prediction, action_correct: bool) -> float:
    alarm_conf = clip(prediction.alarm_confidence)
    action_conf = clip(prediction.action_confidence)
    alarm_brier = (alarm_conf - 1.0) ** 2
    action_brier = (action_conf - (1.0 if action_correct else 0.0)) ** 2
    return clip(1.0 - (alarm_brier + action_brier) / 2.0)


def efficiency_score(prediction: Prediction) -> float:
    token_term = min(1.0, prediction.token_count / 12000) if prediction.token_count else 0.0
    tool_term = min(1.0, prediction.tool_calls / 80) if prediction.tool_calls else 0.0
    wall_term = min(1.0, prediction.wall_time_s / 300.0) if prediction.wall_time_s else 0.0
    joule_term = min(1.0, prediction.compute_joules / 50000.0) if prediction.compute_joules else 0.0
    usage = 0.45 * token_term + 0.25 * tool_term + 0.20 * wall_term + 0.10 * joule_term
    return clip(1.0 - usage)


def score_episode(episode: Episode, prediction: Prediction) -> ScoreBreakdown:
    if episode.episode_id != prediction.episode_id:
        raise ValueError(
            f"Prediction episode_id {prediction.episode_id!r} does not match episode {episode.episode_id!r}"
        )

    replay = replay_prediction(episode, prediction)
    truth = episode.ground_truth
    action_correct = any(action_id in truth.oracle_action_ids for action_id in prediction.action_ids)
    det = detection_score(truth.fault.start_time, prediction.alarm_time)
    rc = root_cause_score(episode, prediction)
    intervention = intervention_score(episode, replay.policy_loss, replay.safety_violations)
    evidence = evidence_f1(prediction.evidence_spans, truth.gold_evidence_spans)
    calibration = calibration_score(prediction, action_correct)
    efficiency = efficiency_score(prediction)
    composite = (
        0.15 * det
        + 0.20 * rc
        + 0.30 * intervention
        + 0.15 * evidence
        + 0.10 * calibration
        + 0.10 * efficiency
    )
    if replay.safety_violations:
        composite = min(composite, 0.45)
    return ScoreBreakdown(
        episode_id=episode.episode_id,
        detection=round(det, 6),
        root_cause=round(rc, 6),
        intervention=round(intervention, 6),
        evidence=round(evidence, 6),
        calibration=round(calibration, 6),
        efficiency=round(efficiency, 6),
        composite=round(clip(composite), 6),
        policy_loss=replay.policy_loss,
        safety_violations=replay.safety_violations,
        notes=replay.notes,
    )


def graph_distance(
    edges: list[tuple[str, str]],
    start: str | None,
    target: str | None,
) -> int:
    if start is None or target is None:
        return graph_diameter(edges)
    if start == target:
        return 0
    adjacency = _undirected_adjacency(edges)
    queue: deque[tuple[str, int]] = deque([(start, 0)])
    seen = {start}
    while queue:
        node, dist = queue.popleft()
        for neighbor in adjacency.get(node, set()):
            if neighbor == target:
                return dist + 1
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, dist + 1))
    return graph_diameter(edges)


def graph_diameter(edges: list[tuple[str, str]]) -> int:
    adjacency = _undirected_adjacency(edges)
    if not adjacency:
        return 1
    max_dist = 1
    for start in adjacency:
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        seen = {start}
        while queue:
            node, dist = queue.popleft()
            max_dist = max(max_dist, dist)
            for neighbor in adjacency.get(node, set()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, dist + 1))
    return max_dist


def _undirected_adjacency(edges: list[tuple[str, str]]) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {}
    for left, right in edges:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    return adjacency


def _component_from_label(label: str) -> str:
    return label.split(":", 1)[0]
