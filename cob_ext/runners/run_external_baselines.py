"""Run dependency-free baselines on external validation episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from causalopsbench._io import write_csv_rows, write_json_document
from cob_ext.baselines import get_baseline
from cob_ext.scoring.score_external import evaluate_predictions
from cob_ext.schemas import load_episodes_jsonl, to_dict, write_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run external validation baselines.")
    parser.add_argument("--episodes", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baselines", nargs="+", default=["threshold", "topology_heuristic"])
    args = parser.parse_args(argv)

    episodes = []
    for path in args.episodes:
        episodes.extend(load_episodes_jsonl(path))
    output_dir = Path(args.output_dir)
    prediction_dir = output_dir / "predictions"
    score_dir = output_dir / "scores"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for name in args.baselines:
        baseline = get_baseline(name)
        predictions = [baseline.predict(episode) for episode in episodes]
        write_jsonl(prediction_dir / f"{baseline.name}.jsonl", predictions)
        scores, summary = evaluate_predictions(episodes, predictions)
        write_json_document(
            score_dir / f"{baseline.name}.json",
            {
                "baseline": baseline.name,
                "summary": summary,
                "scores": [to_dict(score) for score in scores],
            },
        )
        row = {"system": baseline.name}
        row.update(summary)
        rows.append(row)
    _write_summary(output_dir / "summary.csv", rows)
    print(json.dumps({"output_dir": str(output_dir), "episodes": len(episodes)}, sort_keys=True))
    return 0


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv_rows(path, rows)


if __name__ == "__main__":
    raise SystemExit(main())
