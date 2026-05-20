"""Adapter for LBNL FDD RTU and SD-AHU public building traces."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
from typing import Any

from cob_ext.adapters.base import (
    SourceBlob,
    candidate_actions_for_components,
    collect_source_blobs,
    numeric,
    read_csv_blob,
    select_deterministic,
    slug,
    stable_id,
    timestamp,
    write_episodes_and_manifest,
)
from cob_ext.schemas import DEFAULT_SUPPORTED_METRICS, ExternalEpisode, ExternalGroundTruth

SUPPORTED_SUFFIXES = {".csv", ".ttl", ".pdf", ".txt", ".json"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert LBNL FDD RTU/SD-AHU data into ExternalEpisode JSONL.")
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--tracks", nargs="+", default=["RTU", "SD_AHU"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--window_minutes", type=int, default=240)
    parser.add_argument("--stride_minutes", type=int, default=120)
    parser.add_argument("--max_windows_per_csv", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260519)
    args = parser.parse_args(argv)

    episodes, blobs = build_episodes(
        raw_dir=args.raw_dir,
        tracks=args.tracks,
        window_minutes=args.window_minutes,
        stride_minutes=args.stride_minutes,
        max_windows_per_csv=args.max_windows_per_csv,
        seed=args.seed,
    )
    if not episodes:
        raise SystemExit("No LBNL FDD episodes were produced. Check raw_dir and track names.")
    write_episodes_and_manifest(
        args.output,
        args.manifest,
        episodes,
        dataset="lbnl_fdd_rtu_sdahu",
        raw_dir=args.raw_dir,
        source_blobs=blobs,
        config={
            "tracks": args.tracks,
            "window_minutes": args.window_minutes,
            "stride_minutes": args.stride_minutes,
            "max_windows_per_csv": args.max_windows_per_csv,
            "seed": args.seed,
        },
    )
    print(json.dumps({"output": args.output, "manifest": args.manifest, "episodes": len(episodes)}, sort_keys=True))
    return 0


def build_episodes(
    raw_dir: str | Path,
    tracks: list[str],
    window_minutes: int = 240,
    stride_minutes: int = 120,
    max_windows_per_csv: int = 5,
    seed: int = 20260519,
) -> tuple[list[ExternalEpisode], list[SourceBlob]]:
    blobs = collect_source_blobs(raw_dir, SUPPORTED_SUFFIXES)
    csv_blobs = [blob for blob in blobs if blob.suffix == ".csv" and _matches_track(blob.name, tracks)]
    if not csv_blobs:
        csv_blobs = [blob for blob in blobs if blob.suffix == ".csv"]
    topology = _topology_from_inventory(blobs)
    baseline_stats = _fault_free_stats(csv_blobs)
    episodes: list[ExternalEpisode] = []
    for blob in csv_blobs:
        episodes.extend(
            _episodes_for_csv(
                blob=blob,
                topology=topology,
                baseline_stats=baseline_stats,
                window_minutes=window_minutes,
                stride_minutes=stride_minutes,
                max_windows=max_windows_per_csv,
                seed=seed,
            )
        )
    return episodes, blobs


def _episodes_for_csv(
    *,
    blob: SourceBlob,
    topology: list[tuple[str, str]],
    baseline_stats: dict[str, float],
    window_minutes: int,
    stride_minutes: int,
    max_windows: int,
    seed: int,
) -> list[ExternalEpisode]:
    rows = read_csv_blob(blob)
    numeric_columns = _numeric_columns(rows)
    if not rows or not numeric_columns:
        return []
    track = _track_from_name(blob.name)
    fault_type = _fault_type_from_name(blob.name)
    component = _component_from_track(track, blob.name)
    is_faulted = fault_type not in {"fault_free", "normal"}
    windows = _window_bounds(rows, window_minutes, stride_minutes, max_windows)
    selected = select_deterministic(windows, max_windows, seed, blob.source_id)
    episodes: list[ExternalEpisode] = []
    for index, (start, end) in enumerate(selected):
        window_rows = rows[start:end]
        sensors = _sensor_frames(window_rows, numeric_columns)
        evidence = _proxy_evidence(
            blob=blob,
            track=track,
            component=component,
            fault_type=fault_type,
            rows=window_rows,
            numeric_columns=numeric_columns,
            baseline_stats=baseline_stats,
            window_index=index,
            is_faulted=is_faulted,
        )
        gold_spans = [item["span_id"] for item in evidence] if is_faulted else []
        components = _components_from_topology(topology) | {component}
        episodes.append(
            ExternalEpisode(
                episode_id=f"lbnl-fdd-{slug(Path(blob.name).stem)}-w{index}",
                source_dataset="lbnl_fdd_rtu_sdahu",
                domain="hvac",
                split="external",
                duration=max(0, len(window_rows) - 1),
                topology=topology or _default_hvac_topology(track, component, numeric_columns),
                observations={
                    "sensors": sensors,
                    "logs": [],
                    "traces": [],
                    "manuals": [],
                    "evidence": evidence,
                },
                candidate_actions=candidate_actions_for_components(components),
                ground_truth=ExternalGroundTruth(
                    fault_component=component if is_faulted else "none",
                    fault_type=fault_type,
                    root_cause_label=f"{component}:{fault_type}" if is_faulted else "none:normal",
                    is_faulted=is_faulted,
                    fault_start_time=0 if is_faulted else None,
                    severity=_severity_from_name(blob.name),
                    gold_evidence_spans=gold_spans,
                    metadata={
                        "label_source": "LBNL FDD filename/inventory metadata",
                        "proxy_evidence": True,
                        "source_csv": blob.source_id,
                    },
                ),
                supported_metrics=list(DEFAULT_SUPPORTED_METRICS),
                metadata={
                    "external_validation": True,
                    "dataset_role": "primary_hvac_external_trace",
                    "track": track,
                    "source_csv": blob.source_id,
                    "window_start_row": start,
                    "window_end_row": end,
                    "evidence_label_type": "proxy",
                    "candidate_root_causes": [
                        f"{candidate}:{fault_type}" for candidate in sorted(components)
                    ],
                },
            )
        )
    return episodes


def _sensor_frames(rows: list[dict[str, str]], numeric_columns: list[str]) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        time_value = row.get("timestamp") or row.get("time") or row.get("Date") or row.get("date")
        values = {
            column: parsed
            for column in numeric_columns
            if (parsed := numeric(row.get(column))) is not None
        }
        frames.append(
            {
                "timestamp": timestamp(time_value, fallback=index),
                "values": values,
            }
        )
    return frames


def _proxy_evidence(
    *,
    blob: SourceBlob,
    track: str,
    component: str,
    fault_type: str,
    rows: list[dict[str, str]],
    numeric_columns: list[str],
    baseline_stats: dict[str, float],
    window_index: int,
    is_faulted: bool,
) -> list[dict[str, Any]]:
    deviations: list[tuple[float, str, float, float]] = []
    for column in numeric_columns:
        values = [
            parsed
            for row in rows
            if (parsed := numeric(row.get(column))) is not None
        ]
        if not values:
            continue
        observed = mean(values)
        baseline = baseline_stats.get(column, values[0])
        deviations.append((abs(observed - baseline), column, observed, baseline))
    deviations.sort(reverse=True)
    records = []
    for rank, (delta, column, observed, baseline) in enumerate(deviations[:5]):
        span = f"lbnl-{stable_id(blob.source_id, window_index, column, rank)}"
        text = (
            f"LBNL {track} proxy evidence: sensor {column} mean={observed:.4g}, "
            f"fault-free reference={baseline:.4g}, deviation={delta:.4g}, "
            f"file label={fault_type}."
        )
        if not is_faulted:
            text = (
                f"LBNL {track} fault-free proxy evidence: sensor {column} stayed near "
                f"reference mean {baseline:.4g}."
            )
        records.append(
            {
                "span_id": span,
                "source_id": blob.source_id,
                "kind": "sensor_deviation",
                "timestamp": 0,
                "component": component,
                "proxy": True,
                "text": text,
                "metadata": {
                    "sensor": column,
                    "observed_mean": observed,
                    "reference_mean": baseline,
                    "absolute_deviation": delta,
                },
            }
        )
    return records


def _fault_free_stats(csv_blobs: list[SourceBlob]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for blob in csv_blobs:
        if _fault_type_from_name(blob.name) not in {"fault_free", "normal"}:
            continue
        rows = read_csv_blob(blob)
        for column in _numeric_columns(rows):
            values[column].extend(
                parsed
                for row in rows
                if (parsed := numeric(row.get(column))) is not None
            )
    return {column: mean(column_values) for column, column_values in values.items() if column_values}


def _numeric_columns(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return []
    ignored = {"timestamp", "time", "date", "datetime", "fault", "label", "mode"}
    columns = []
    for column in rows[0]:
        if column.lower() in ignored:
            continue
        seen = [numeric(row.get(column)) for row in rows[:20]]
        if any(value is not None for value in seen):
            columns.append(column)
    return columns


def _window_bounds(
    rows: list[dict[str, str]],
    window_minutes: int,
    stride_minutes: int,
    max_windows: int,
) -> list[tuple[int, int]]:
    if not rows:
        return []
    window = min(max(1, window_minutes), len(rows))
    stride = max(1, stride_minutes)
    bounds = [(start, min(start + window, len(rows))) for start in range(0, len(rows), stride)]
    bounds = [bound for bound in bounds if bound[1] > bound[0]]
    return bounds[:max_windows] or [(0, len(rows))]


def _topology_from_inventory(blobs: list[SourceBlob]) -> list[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for blob in blobs:
        if blob.suffix != ".ttl":
            continue
        for line in blob.text.splitlines():
            compact = line.strip()
            if not compact or compact.startswith("#"):
                continue
            tokens = [token.strip("<>;.,") for token in compact.split()]
            if len(tokens) >= 3 and any(rel in tokens[1].lower() for rel in ("feeds", "haspart", "equip")):
                edges.add((slug(tokens[0]), slug(tokens[2])))
    return sorted(edges)


def _default_hvac_topology(track: str, component: str, numeric_columns: list[str]) -> list[tuple[str, str]]:
    if track == "SD_AHU":
        base = [("outdoor-air", "ahu"), ("ahu", "vav-1"), ("ahu", "vav-2"), ("ahu", "vav-3")]
    else:
        base = [("outdoor-air", "rtu"), ("rtu", "supply-air"), ("supply-air", "zone")]
    sensor_edges = [(component, slug(column)) for column in numeric_columns[:6]]
    return base + sensor_edges


def _components_from_topology(topology: list[tuple[str, str]]) -> set[str]:
    return {node for edge in topology for node in edge}


def _matches_track(name: str, tracks: list[str]) -> bool:
    lowered = name.lower().replace("-", "_")
    aliases = {
        "rtu": ["rtu", "rooftop"],
        "sd_ahu": ["sd_ahu", "sdahu", "single_duct", "single-duct", "ahu"],
    }
    for track in tracks:
        normalized = track.lower().replace("-", "_")
        if any(alias in lowered for alias in aliases.get(normalized, [normalized])):
            return True
    return False


def _track_from_name(name: str) -> str:
    lowered = name.lower().replace("-", "_")
    if "sd_ahu" in lowered or "sdahu" in lowered or "single_duct" in lowered or "ahu" in lowered:
        return "SD_AHU"
    return "RTU"


def _fault_type_from_name(name: str) -> str:
    lowered = slug(Path(name).stem).replace("-", "_")
    if any(token in lowered for token in ("fault_free", "faultfree", "normal", "baseline")):
        return "fault_free"
    for token in ("damper", "valve", "fan", "sensor", "coil", "economizer", "heating", "cooling", "oa", "ra", "sa"):
        if token in lowered:
            return token
    return lowered or "unknown_fault"


def _component_from_track(track: str, name: str) -> str:
    lowered = name.lower()
    if "fan" in lowered:
        return "supply-fan"
    if "damper" in lowered or "economizer" in lowered:
        return "outdoor-air-damper"
    if "valve" in lowered or "coil" in lowered:
        return "cooling-coil"
    return "ahu" if track == "SD_AHU" else "rtu"


def _severity_from_name(name: str) -> float | None:
    lowered = Path(name).stem.lower()
    for marker in ("sev", "severity", "level"):
        if marker in lowered:
            tail = lowered.split(marker, 1)[1]
            digits = "".join(char for char in tail if char.isdigit() or char == ".")
            if digits:
                return numeric(digits)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
