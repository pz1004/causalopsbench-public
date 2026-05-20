"""Score external trace predictions without native replay metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from causalopsbench._io import write_csv_rows, write_json_document
from causalopsbench.metrics import detection_score, evidence_f1, clip
from cob_ext.schemas import (
    ExternalEpisode,
    ExternalPrediction,
    ExternalScoreBreakdown,
    UNSUPPORTED_REPLAY_METRICS,
    load_episodes_jsonl,
    load_predictions_jsonl,
    to_dict,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score external trace portability predictions.")
    parser.add_argument("--episodes", nargs="+", required=True)
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)

    episodes = _load_episode_paths(args.episodes)
    prediction_dir = Path(args.predictions_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for prediction_path in sorted(prediction_dir.glob("*.jsonl")):
        if prediction_path.name.endswith("_raw.jsonl"):
            continue
        predictions = load_predictions_jsonl(prediction_path)
        scores, summary = evaluate_predictions(episodes, predictions, top_k=args.top_k)
        payload = {
            "prediction_file": str(prediction_path),
            "summary": summary,
            "scores": [to_dict(score) for score in scores],
        }
        score_path = output_dir / f"{prediction_path.stem}.json"
        write_json_document(score_path, payload)
        row = {"system": prediction_path.stem}
        row.update(summary)
        rows.append(row)
    _write_summary(output_dir / "summary.csv", rows)
    print(json.dumps({"output_dir": str(output_dir), "systems": len(rows)}, sort_keys=True))
    return 0


def evaluate_predictions(
    episodes: Iterable[ExternalEpisode],
    predictions: Iterable[ExternalPrediction],
    top_k: int = 3,
) -> tuple[list[ExternalScoreBreakdown], dict[str, float]]:
    episode_list = list(episodes)
    for episode in episode_list:
        _ensure_no_replay_metrics(episode)
    prediction_by_id = {prediction.episode_id: prediction for prediction in predictions}
    missing = [episode.episode_id for episode in episode_list if episode.episode_id not in prediction_by_id]
    if missing:
        raise ValueError(f"Missing predictions for {len(missing)} external episodes: {missing[:3]}")
    scores = [
        score_episode(episode, prediction_by_id[episode.episode_id], top_k=top_k)
        for episode in episode_list
    ]
    return scores, summarize_scores(scores)


def score_episode(
    episode: ExternalEpisode,
    prediction: ExternalPrediction,
    top_k: int = 3,
) -> ExternalScoreBreakdown:
    if episode.episode_id != prediction.episode_id:
        raise ValueError(
            f"Prediction episode_id {prediction.episode_id!r} does not match {episode.episode_id!r}"
        )
    _ensure_no_replay_metrics(episode)
    truth = episode.ground_truth
    if truth.is_faulted:
        det = detection_score(truth.fault_start_time or 0, prediction.alarm_time)
    else:
        det = 1.0 if prediction.alarm_time is None or prediction.alarm_confidence < 0.5 else 0.0
    top1 = _root_match(episode, prediction.root_cause_topk[:1])
    topk = _root_match(episode, prediction.root_cause_topk[:top_k])
    evidence = evidence_f1(prediction.evidence_spans, truth.gold_evidence_spans)
    calibration = _calibration(prediction, truth.is_faulted)
    parsing = 1.0 if prediction.parse_success else 0.0
    runtime = _runtime_efficiency(prediction)
    eps = (
        0.25 * det
        + 0.35 * topk
        + 0.20 * evidence
        + 0.10 * calibration
        + 0.05 * parsing
        + 0.05 * runtime
    )
    return ExternalScoreBreakdown(
        episode_id=episode.episode_id,
        detection=round(det, 6),
        root_cause_top1=round(top1, 6),
        root_cause_topk=round(topk, 6),
        evidence_f1=round(evidence, 6),
        calibration=round(calibration, 6),
        parsing_success=round(parsing, 6),
        runtime_efficiency=round(runtime, 6),
        external_portability_score=round(clip(eps), 6),
        notes=[],
    )


def summarize_scores(scores: list[ExternalScoreBreakdown]) -> dict[str, float]:
    if not scores:
        return {
            "count": 0.0,
            "external_portability_score": 0.0,
            "detection": 0.0,
            "root_cause_top1": 0.0,
            "root_cause_topk": 0.0,
            "evidence_f1": 0.0,
            "calibration": 0.0,
            "parsing_success": 0.0,
            "runtime_efficiency": 0.0,
        }
    fields = [
        "external_portability_score",
        "detection",
        "root_cause_top1",
        "root_cause_topk",
        "evidence_f1",
        "calibration",
        "parsing_success",
        "runtime_efficiency",
    ]
    summary = {"count": float(len(scores))}
    for field in fields:
        summary[field] = round(mean(getattr(score, field) for score in scores), 6)
    return summary


def _root_match(episode: ExternalEpisode, labels: list[str]) -> float:
    truth = episode.ground_truth
    if not truth.is_faulted:
        return 1.0 if not labels or any(_is_non_fault_label(label) for label in labels) else 0.0
    normalized = {_normalize_label(label) for label in labels}
    true_label = _normalize_label(truth.root_cause_label)
    if not true_label or not normalized:
        return 0.0
    return 1.0 if true_label in normalized else 0.0


def _normalize_label(label: str) -> str:
    return str(label).strip()


def _is_non_fault_label(label: str) -> bool:
    normalized = _normalize_label(label).lower()
    return normalized in {"none", "normal", "none:normal", "normal:none"} or normalized.startswith("none:")


def _calibration(prediction: ExternalPrediction, is_faulted: bool) -> float:
    target = 1.0 if is_faulted else 0.0
    return clip(1.0 - (clip(prediction.alarm_confidence) - target) ** 2)


def _runtime_efficiency(prediction: ExternalPrediction) -> float:
    token_term = min(1.0, prediction.token_count / 12000) if prediction.token_count else 0.0
    tool_term = min(1.0, prediction.tool_calls / 80) if prediction.tool_calls else 0.0
    wall_term = min(1.0, prediction.wall_time_s / 300.0) if prediction.wall_time_s else 0.0
    usage = 0.55 * token_term + 0.25 * tool_term + 0.20 * wall_term
    return clip(1.0 - usage)


def _ensure_no_replay_metrics(episode: ExternalEpisode) -> None:
    unsupported = set(episode.supported_metrics) & UNSUPPORTED_REPLAY_METRICS
    if unsupported:
        raise ValueError(
            f"{episode.episode_id}: unsupported external replay metrics requested: "
            + ", ".join(sorted(unsupported))
        )


def _load_episode_paths(paths: list[str]) -> list[ExternalEpisode]:
    episodes: list[ExternalEpisode] = []
    for path in paths:
        episodes.extend(load_episodes_jsonl(path))
    return episodes


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv_rows(path, rows)


if __name__ == "__main__":
    raise SystemExit(main())
