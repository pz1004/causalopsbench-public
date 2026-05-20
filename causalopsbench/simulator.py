"""Action replay and loss model for CausalOpsBench episodes."""

from __future__ import annotations

from dataclasses import dataclass

from causalopsbench.schemas import Episode, Prediction


@dataclass(frozen=True)
class ReplayResult:
    policy_loss: float
    safety_violations: int
    action_cost: float
    delay_penalty: float
    notes: list[str]


def replay_prediction(episode: Episode, prediction: Prediction) -> ReplayResult:
    """Estimate incident loss after a policy intervention.

    This scaffold uses a deterministic proxy loss. Richer versions can replace this
    function with simulator replay while keeping the public scoring contract stable.
    """
    truth = episode.ground_truth
    action_by_id = {action.action_id: action for action in episode.actions}
    selected = [action_by_id[action_id] for action_id in prediction.action_ids if action_id in action_by_id]
    invalid = [action_id for action_id in prediction.action_ids if action_id not in action_by_id]
    notes: list[str] = []
    if invalid:
        notes.append(f"invalid_actions={','.join(invalid)}")

    action_cost = sum(action.cost for action in selected)
    safety_violations = sum(1 for action in selected if action.safety_risk >= 0.5)
    oracle_selected = any(action.action_id in truth.oracle_action_ids for action in selected)
    related_selected = any(
        action.target_component in truth.fault.root_cause_path for action in selected
    )

    alarm_time = prediction.alarm_time
    if alarm_time is None:
        delay = max(0, episode.duration - truth.fault.start_time)
    else:
        delay = max(0, alarm_time - truth.fault.start_time)
    delay_penalty = 0.015 * delay * truth.noop_loss

    if oracle_selected:
        base_loss = truth.oracle_loss
        notes.append("oracle_action_selected")
    elif related_selected:
        base_loss = truth.oracle_loss + 0.45 * (truth.noop_loss - truth.oracle_loss)
        notes.append("related_component_action_selected")
    elif selected:
        base_loss = truth.noop_loss * 1.05
        notes.append("unrelated_action_selected")
    else:
        base_loss = truth.noop_loss
        notes.append("no_action_selected")

    safety_penalty = safety_violations * max(25.0, 0.25 * truth.noop_loss)
    invalid_penalty = len(invalid) * 10.0
    policy_loss = round(
        base_loss + action_cost + delay_penalty + safety_penalty + invalid_penalty,
        4,
    )
    return ReplayResult(
        policy_loss=policy_loss,
        safety_violations=safety_violations,
        action_cost=round(action_cost, 4),
        delay_penalty=round(delay_penalty, 4),
        notes=notes,
    )
