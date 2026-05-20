#!/usr/bin/env python3
"""Run reproducible CausalOpsBench scaffold experiments.

The runner intentionally uses only the Python standard library plus the local
package. It generates deterministic episodes, evaluates local baselines, and
writes JSON, CSV, and compact table artifacts for public release validation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causalopsbench.baselines import get_baseline
from causalopsbench._io import write_csv_rows, write_json_document
from causalopsbench.domains import domain_names
from causalopsbench.generator import EpisodeGenerator
from causalopsbench.metrics import INTERVENTION_EPSILON, is_degenerate_intervention, score_episode
from causalopsbench.schemas import ScoreBreakdown, to_dict, write_jsonl


METRICS = [
    "composite",
    "detection",
    "root_cause",
    "intervention",
    "evidence",
    "calibration",
    "efficiency",
]
DEFAULT_BASELINES = ["noop", "random", "threshold", "topology_rca", "oracle"]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run CausalOpsBench baseline experiments and write release artifacts."
    )
    parser.add_argument("--output-dir", default="results/cob_v2")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--domain", default="all", choices=["all"] + domain_names())
    parser.add_argument("--topology-variant", default="public", choices=["public", "heldout_v1"])
    parser.add_argument("--baselines", nargs="+", default=DEFAULT_BASELINES)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--write-episodes", dest="write_episodes", action="store_true", default=True)
    parser.add_argument("--no-write-episodes", dest="write_episodes", action="store_false")
    args = parser.parse_args(argv)

    if args.count <= 0:
        raise SystemExit("--count must be positive")
    if args.duration < 20:
        raise SystemExit("--duration must be at least 20")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir = output_dir / "episodes"
    scores_dir = output_dir / "scores"
    if args.write_episodes:
        episodes_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, Any]] = []
    for seed in args.seeds:
        episodes = EpisodeGenerator(
            seed=seed,
            duration=args.duration,
            topology_variant=args.topology_variant,
        ).generate(
            count=args.count,
            domain=args.domain,
        )
        if args.write_episodes:
            write_jsonl(episodes_dir / f"seed_{seed}.jsonl", episodes)

        degenerate_ids = {
            episode.episode_id for episode in episodes if is_degenerate_intervention(episode)
        }
        scoreable_episodes = [
            episode for episode in episodes if episode.episode_id not in degenerate_ids
        ]

        for baseline_name in args.baselines:
            output_json = scores_dir / f"{baseline_name}_seed_{seed}.json"
            if args.skip_existing and output_json.exists():
                payload = json.loads(output_json.read_text(encoding="utf-8"))
                records = payload["scores"]
            else:
                baseline = get_baseline(baseline_name, seed=seed)
                records = []
                for episode in scoreable_episodes:
                    prediction = baseline.predict(episode)
                    score = score_episode(episode, prediction)
                    records.append(_record(seed, baseline_name, episode.domain, score))
                payload = {
                    "baseline": baseline_name,
                    "seed": seed,
                    "count": len(records),
                    "n_total": len(episodes),
                    "n_scored": len(scoreable_episodes),
                    "n_deg": len(degenerate_ids),
                    "intervention_epsilon": INTERVENTION_EPSILON,
                    "summary": _summarize(records, ["baseline", "seed"]),
                    "scores": records,
                }
                write_json_document(output_json, payload)
            all_records.extend(records)

    summary_rows = _summarize_by(all_records, ["baseline"])
    domain_rows = _summarize_by(all_records, ["baseline", "domain"])
    seed_rows = _summarize_by(all_records, ["baseline", "seed"])

    _write_csv(output_dir / "summary.csv", summary_rows)
    _write_csv(output_dir / "summary_by_domain.csv", domain_rows)
    _write_csv(output_dir / "summary_by_seed.csv", seed_rows)
    _write_leaderboard_tex(output_dir / "leaderboard_table.tex", summary_rows)
    _write_domain_tex(output_dir / "domain_table.tex", domain_rows)
    _write_manifest(output_dir / "experiment_manifest.json", args, all_records)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "episodes": len(args.seeds) * args.count,
                "baselines": args.baselines,
                "summary_csv": str(output_dir / "summary.csv"),
            },
            sort_keys=True,
        )
    )
    return 0


def _record(seed: int, baseline: str, domain: str, score: ScoreBreakdown) -> dict[str, Any]:
    data = to_dict(score)
    data.update(
        {
            "seed": seed,
            "baseline": baseline,
            "domain": domain,
        }
    )
    return data


def _summarize_by(records: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(tuple(record[key] for key in keys), []).append(record)

    rows: list[dict[str, Any]] = []
    for group_key, group_records in sorted(groups.items()):
        row = {key: value for key, value in zip(keys, group_key)}
        row.update(_metric_summary(group_records))
        rows.append(row)
    return rows


def _summarize(records: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    rows = _summarize_by(records, keys)
    return rows[0] if rows else {}


def _metric_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "count": len(records),
        "safety_violations": sum(int(record["safety_violations"]) for record in records),
    }
    for metric in METRICS:
        values = [float(record[metric]) for record in records]
        row[f"{metric}_mean"] = round(statistics.fmean(values), 6)
        row[f"{metric}_std"] = round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv_rows(path, rows)


def _write_leaderboard_tex(path: Path, rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda row: row["composite_mean"], reverse=True)
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Baseline & Composite & Detect. & Root & Interv. & Evidence & Calib. & Eff. & Safety \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _tex_escape(str(row["baseline"])),
                    _fmt(row["composite_mean"]),
                    _fmt(row["detection_mean"]),
                    _fmt(row["root_cause_mean"]),
                    _fmt(row["intervention_mean"]),
                    _fmt(row["evidence_mean"]),
                    _fmt(row["calibration_mean"]),
                    _fmt(row["efficiency_mean"]),
                    str(int(row["safety_violations"])),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_domain_tex(path: Path, rows: list[dict[str, Any]]) -> None:
    baselines = sorted({row["baseline"] for row in rows})
    domains = sorted({row["domain"] for row in rows})
    by_key = {(row["baseline"], row["domain"]): row for row in rows}
    alignment = "l" + ("r" * len(domains))
    lines = [
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        "Baseline & " + " & ".join(_tex_escape(domain) for domain in domains) + r" \\",
        r"\midrule",
    ]
    for baseline in baselines:
        values = [
            _fmt(by_key[(baseline, domain)]["composite_mean"])
            if (baseline, domain) in by_key
            else "--"
            for domain in domains
        ]
        lines.append(_tex_escape(baseline) + " & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_manifest(path: Path, args: argparse.Namespace, records: list[dict[str, Any]]) -> None:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner": "scripts/run_experiments.py",
        "configuration": {
            "seeds": args.seeds,
            "count": args.count,
            "duration": args.duration,
            "domain": args.domain,
            "topology_variant": args.topology_variant,
            "baselines": args.baselines,
            "write_episodes": args.write_episodes,
        },
        "records": len(records),
        "N": len(records) // max(1, len(args.baselines)),
        "N_deg": _manifest_degenerate_count(args),
        "intervention_epsilon": INTERVENTION_EPSILON,
        "metrics": METRICS,
        "claim_policy": "synthetic scaffold experiments; not real-world industrial validation",
    }
    write_json_document(path, payload)


def _manifest_degenerate_count(args: argparse.Namespace) -> int:
    total = 0
    for seed in args.seeds:
        episodes = EpisodeGenerator(
            seed=seed,
            duration=args.duration,
            topology_variant=args.topology_variant,
        ).generate(
            count=args.count,
            domain=args.domain,
        )
        total += sum(1 for episode in episodes if is_degenerate_intervention(episode))
    return total


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _tex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


if __name__ == "__main__":
    raise SystemExit(main())
