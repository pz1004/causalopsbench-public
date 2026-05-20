"""Benchmark evaluation helpers."""

from __future__ import annotations

from statistics import mean
from typing import Iterable

from causalopsbench.baselines import Baseline, get_baseline
from causalopsbench.metrics import INTERVENTION_EPSILON, is_degenerate_intervention, score_episode
from causalopsbench.schemas import Episode, Prediction, ScoreBreakdown


def evaluate_predictions(
    episodes: Iterable[Episode],
    predictions: Iterable[Prediction],
) -> tuple[list[ScoreBreakdown], dict[str, float]]:
    episode_list = list(episodes)
    prediction_by_id = {prediction.episode_id: prediction for prediction in predictions}
    missing = [episode.episode_id for episode in episode_list if episode.episode_id not in prediction_by_id]
    if missing:
        raise ValueError(f"Missing predictions for {len(missing)} episodes: {missing[:3]}")
    scoreable_episodes = [
        episode for episode in episode_list if not is_degenerate_intervention(episode)
    ]
    scores = [
        score_episode(episode, prediction_by_id[episode.episode_id])
        for episode in scoreable_episodes
    ]
    summary = summarize_scores(scores)
    summary["n_total"] = float(len(episode_list))
    summary["n_scored"] = float(len(scoreable_episodes))
    summary["n_deg"] = float(len(episode_list) - len(scoreable_episodes))
    summary["intervention_epsilon"] = INTERVENTION_EPSILON
    return scores, summary


def evaluate_baseline(
    episodes: Iterable[Episode],
    baseline: str | Baseline,
    seed: int = 0,
) -> tuple[list[ScoreBreakdown], dict[str, float]]:
    baseline_obj = get_baseline(baseline, seed=seed) if isinstance(baseline, str) else baseline
    episode_list = list(episodes)
    predictions = [baseline_obj.predict(episode) for episode in episode_list]
    return evaluate_predictions(episode_list, predictions)


def summarize_scores(scores: list[ScoreBreakdown]) -> dict[str, float]:
    if not scores:
        return {
            "count": 0.0,
            "composite": 0.0,
            "detection": 0.0,
            "root_cause": 0.0,
            "intervention": 0.0,
            "evidence": 0.0,
            "calibration": 0.0,
            "efficiency": 0.0,
            "safety_violations": 0.0,
        }
    fields = [
        "composite",
        "detection",
        "root_cause",
        "intervention",
        "evidence",
        "calibration",
        "efficiency",
    ]
    summary = {"count": float(len(scores))}
    for field in fields:
        summary[field] = round(mean(getattr(score, field) for score in scores), 6)
    summary["safety_violations"] = float(sum(score.safety_violations for score in scores))
    return summary
