"""Command-line interface for CausalOpsBench."""

from __future__ import annotations

import argparse
import json

from causalopsbench._io import write_json_document
from causalopsbench.baselines import get_baseline
from causalopsbench.domains import domain_names
from causalopsbench.evaluate import evaluate_baseline, evaluate_predictions
from causalopsbench.generator import EpisodeGenerator
from causalopsbench.schemas import (
    dumps_json,
    load_episodes_jsonl,
    load_predictions_jsonl,
    to_dict,
    write_jsonl,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="causalopsbench",
        description="Generate and evaluate CausalOpsBench episodes.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate JSONL benchmark episodes")
    generate.add_argument("--count", type=int, default=25)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--duration", type=int, default=60)
    generate.add_argument("--domain", default="all", choices=["all"] + domain_names())
    generate.add_argument("--topology-variant", default="public", choices=["public", "heldout_v1"])
    generate.add_argument("--output", required=True)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a baseline or predictions")
    evaluate.add_argument("--episodes", required=True)
    evaluate.add_argument("--baseline", choices=["noop", "oracle", "random", "threshold", "topology_rca"])
    evaluate.add_argument("--predictions")
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--output")

    show = subparsers.add_parser("show", help="Print episode summaries")
    show.add_argument("--episodes", required=True)
    show.add_argument("--limit", type=int, default=3)

    args = parser.parse_args(argv)
    if args.command == "generate":
        return _generate(args)
    if args.command == "evaluate":
        return _evaluate(args)
    if args.command == "show":
        return _show(args)
    parser.error(f"Unhandled command: {args.command}")
    return 2


def _generate(args: argparse.Namespace) -> int:
    episodes = EpisodeGenerator(
        seed=args.seed,
        duration=args.duration,
        topology_variant=args.topology_variant,
    ).generate(
        count=args.count,
        domain=args.domain,
    )
    write_jsonl(args.output, episodes)
    print(json.dumps({"output": args.output, "count": len(episodes)}, sort_keys=True))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    if bool(args.baseline) == bool(args.predictions):
        raise SystemExit("Provide exactly one of --baseline or --predictions")
    episodes = load_episodes_jsonl(args.episodes)
    if args.baseline:
        get_baseline(args.baseline, seed=args.seed)
        scores, summary = evaluate_baseline(episodes, args.baseline, seed=args.seed)
    else:
        predictions = load_predictions_jsonl(args.predictions)
        scores, summary = evaluate_predictions(episodes, predictions)

    payload = {
        "summary": summary,
        "scores": [to_dict(score) for score in scores],
    }
    if args.output:
        write_json_document(args.output, payload)
    print(json.dumps(summary, sort_keys=True))
    return 0


def _show(args: argparse.Namespace) -> int:
    episodes = load_episodes_jsonl(args.episodes)
    for episode in episodes[: args.limit]:
        truth = episode.ground_truth
        print(
            dumps_json(
                {
                    "episode_id": episode.episode_id,
                    "domain": episode.domain,
                    "duration": episode.duration,
                    "fault_start": truth.fault.start_time,
                    "root_cause_label": truth.fault.label,
                    "oracle_action_ids": truth.oracle_action_ids,
                    "gold_evidence_count": len(truth.gold_evidence_spans),
                    "actions": [action.action_id for action in episode.actions],
                }
            )
        )
    return 0
