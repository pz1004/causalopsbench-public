"""Bootstrap confidence intervals for external validation summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from statistics import mean
from typing import Any

from causalopsbench._io import write_csv_rows, write_json_document
from cob_ext.scoring.score_external import score_episode, summarize_scores
from cob_ext.schemas import load_episodes_jsonl, load_predictions_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap external validation score summaries.")
    parser.add_argument("--episodes", nargs="+", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument(
        "--cluster-field",
        default="source_csv",
        help="Episode metadata field used for clustered resampling; absent values fall back to episode_id.",
    )
    args = parser.parse_args(argv)

    episodes = []
    for path in args.episodes:
        episodes.extend(load_episodes_jsonl(path))
    predictions = load_predictions_jsonl(args.predictions)
    prediction_by_id = {prediction.episode_id: prediction for prediction in predictions}
    scores = [score_episode(episode, prediction_by_id[episode.episode_id]) for episode in episodes]
    clusters = [
        str(episode.metadata.get(args.cluster_field) or episode.episode_id)
        for episode in episodes
    ]
    intervals = bootstrap_intervals(scores, clusters, args.bootstrap, args.seed)
    output = Path(args.output)
    write_json_document(output, intervals)
    print(json.dumps({"output": str(output), "metrics": len(intervals)}, sort_keys=True))
    return 0


def bootstrap_intervals(
    scores: list[Any],
    clusters: list[str] | None = None,
    samples: int = 1000,
    seed: int = 20260519,
) -> dict[str, dict[str, float]]:
    if not scores:
        return {}
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
    clusters = clusters or [str(index) for index, _ in enumerate(scores)]
    cluster_map: dict[str, list[Any]] = {}
    for cluster, score in zip(clusters, scores):
        cluster_map.setdefault(cluster, []).append(score)
    keys = sorted(cluster_map)
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {field: [] for field in fields}
    for _ in range(samples):
        sampled_scores = []
        for _ in keys:
            sampled_scores.extend(cluster_map[rng.choice(keys)])
        summary = summarize_scores(sampled_scores)
        for field in fields:
            draws[field].append(summary[field])
    observed = summarize_scores(scores)
    return {
        field: {
            "mean": observed[field],
            "ci_low": round(_quantile(values, 0.025), 6),
            "ci_high": round(_quantile(values, 0.975), 6),
        }
        for field, values in draws.items()
    }


def write_interval_csv(path: str | Path, intervals: dict[str, dict[str, float]]) -> None:
    rows = [{"metric": metric, **values} for metric, values in intervals.items()]
    write_csv_rows(
        path,
        rows,
        fieldnames=["metric", "mean", "ci_low", "ci_high"],
        write_header_when_empty=True,
    )


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


if __name__ == "__main__":
    raise SystemExit(main())
