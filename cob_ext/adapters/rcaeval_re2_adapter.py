"""Adapter for RCAEval RE2 microservice traces.

The public RCAEval archives have appeared in several directory layouts. This
adapter therefore uses conservative column-name heuristics rather than relying
on one exact file tree. It preserves source checksums in the manifest and emits
external episodes with diagnostic labels and evidence spans only.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any
import zipfile

from cob_ext.adapters.base import (
    SourceBlob,
    candidate_actions_for_components,
    first_present,
    numeric,
    read_csv_blob,
    read_json_records_blob,
    select_deterministic,
    slug,
    stable_id,
    timestamp,
    write_episodes_and_manifest,
)
from cob_ext.schemas import DEFAULT_SUPPORTED_METRICS, ExternalEpisode, ExternalGroundTruth

SUPPORTED_SUFFIXES = {".csv", ".json", ".jsonl", ".log", ".txt"}
MAX_LOG_ROWS_PER_CASE = 200
MAX_TRACE_ROWS_PER_CASE = 200
MAX_WIDE_METRIC_COLUMNS = 40
MAX_OPTIONAL_BLOB_BYTES = 5_000_000


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert RCAEval RE2 data into ExternalEpisode JSONL.")
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--systems", nargs="+", default=["SS", "OB", "TT"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max_evidence_spans", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260519)
    args = parser.parse_args(argv)

    episodes, blobs = build_episodes(
        raw_dir=args.raw_dir,
        systems=args.systems,
        max_evidence_spans=args.max_evidence_spans,
        seed=args.seed,
    )
    if not episodes:
        raise SystemExit(
            "No RCAEval RE2 episodes were produced. Check raw_dir, systems, and label files."
        )
    write_episodes_and_manifest(
        args.output,
        args.manifest,
        episodes,
        dataset="rcaeval_re2",
        raw_dir=args.raw_dir,
        source_blobs=blobs,
        config={
            "systems": args.systems,
            "max_evidence_spans": args.max_evidence_spans,
            "seed": args.seed,
        },
    )
    print(json.dumps({"output": args.output, "manifest": args.manifest, "episodes": len(episodes)}, sort_keys=True))
    return 0


def build_episodes(
    raw_dir: str | Path,
    systems: list[str],
    max_evidence_spans: int = 40,
    seed: int = 20260519,
) -> tuple[list[ExternalEpisode], list[SourceBlob]]:
    blobs = _collect_rcaeval_source_blobs(raw_dir)
    selected = [blob for blob in blobs if _matches_system(blob.name, systems)]
    if not selected:
        selected = blobs

    labels = _load_labels(selected)
    metrics = _load_metric_rows(selected)
    logs = _load_log_rows(selected)
    traces = _load_trace_rows(selected)
    observed_case_ids = set(metrics) | set(logs) | set(traces)
    case_ids = sorted(set(labels) & observed_case_ids)
    episodes = [
        _build_episode(
            case_id=case_id,
            labels=labels.get(case_id, {}),
            metric_rows=metrics.get(case_id, []),
            log_rows=logs.get(case_id, []),
            trace_rows=traces.get(case_id, []),
            max_evidence_spans=max_evidence_spans,
            seed=seed,
        )
        for case_id in case_ids
    ]
    return [episode for episode in episodes if episode is not None], selected


def _collect_rcaeval_source_blobs(raw_dir: str | Path) -> list[SourceBlob]:
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {root}")
    blobs: list[SourceBlob] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for info in sorted(archive.infolist(), key=lambda item: item.filename):
                    if info.is_dir() or not _want_rcaeval_member(info.filename, info.file_size):
                        continue
                    blobs.append(
                        SourceBlob(
                            source_id=f"{path.name}:{info.filename}",
                            name=info.filename,
                            suffix=Path(info.filename).suffix.lower(),
                            data=archive.read(info),
                        )
                    )
            continue
        if _want_rcaeval_member(str(path.relative_to(root)), path.stat().st_size):
            blobs.append(
                SourceBlob(
                    source_id=str(path.relative_to(root)),
                    name=str(path.relative_to(root)),
                    suffix=path.suffix.lower(),
                    data=path.read_bytes(),
                )
            )
    return blobs


def _want_rcaeval_member(name: str, size: int) -> bool:
    leaf = Path(name).name.lower()
    if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
        return False
    if leaf in {"inject_time.txt", "simple_metrics.csv", "tracets_lat.csv", "tracets_err.csv"}:
        return True
    if "label" in leaf or "ground_truth" in leaf or "root_cause" in leaf:
        return size <= MAX_OPTIONAL_BLOB_BYTES
    if leaf in {"metrics.csv", "logs.csv", "traces.csv"} or any(
        token in leaf for token in ("metrics", "logs", "traces")
    ):
        return size <= MAX_OPTIONAL_BLOB_BYTES
    return False


def _load_labels(blobs: list[SourceBlob]) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for blob in blobs:
        path_label = _label_from_blob_path(blob)
        if path_label:
            existing = labels.setdefault(path_label["case_id"], path_label)
            if path_label.get("fault_start_time", 0) and not existing.get("fault_start_time"):
                existing["fault_start_time"] = path_label["fault_start_time"]
        rows = _records_for_blob(blob)
        if not rows:
            continue
        if not (_looks_like_labels(rows[0]) or _filename_role(blob.name) == "label"):
            continue
        for row in rows:
            case_id = _case_id(row, blob)
            component = first_present(
                row,
                [
                    "root_cause_service",
                    "root_service",
                    "root_cause",
                    "service",
                    "component",
                    "fault_service",
                ],
            )
            indicator = first_present(
                row,
                [
                    "root_cause_indicator",
                    "indicator",
                    "fault_type",
                    "metric",
                    "cause",
                    "fault",
                ],
                default="unknown",
            )
            if component:
                labels[case_id] = {
                    "fault_component": component,
                    "fault_type": slug(indicator).replace("-", "_"),
                    "root_cause_label": f"{component}:{slug(indicator).replace('-', '_')}",
                    "fault_start_time": timestamp(
                        first_present(row, ["start_time", "timestamp", "time", "injection_time"]),
                        fallback=0,
                    ),
                    "severity": numeric(first_present(row, ["severity", "level"], default="")),
                    "raw": row,
                }
    return labels


def _load_metric_rows(blobs: list[SourceBlob]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    simple_cases = {
        _case_id({}, blob)
        for blob in blobs
        if Path(blob.name).name.lower() == "simple_metrics.csv"
    }
    for blob in blobs:
        if blob.suffix != ".csv":
            continue
        leaf = Path(blob.name).name.lower()
        case_id_for_blob = _case_id({}, blob)
        if leaf in {"pod-node-1.csv", "pod-node-2.csv", "logts.csv", "tracets_lat.csv", "tracets_err.csv"}:
            continue
        if leaf == "metrics.csv" and case_id_for_blob in simple_cases:
            continue
        rows = read_csv_blob(blob)
        if (
            not rows
            or _looks_like_labels(rows[0])
            or _looks_like_logs(rows[0])
            or _looks_like_traces(rows[0], blob.name)
            or not _looks_like_metrics(rows[0])
        ):
            continue
        for row in rows:
            case_id = _case_id(row, blob)
            grouped[case_id].extend(_metric_records_from_row(row, blob, case_id))
    return grouped


def _load_log_rows(blobs: list[SourceBlob]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for blob in blobs:
        if blob.suffix in {".json", ".jsonl", ".csv"}:
            rows = _records_for_blob(blob)
            if not rows or not _looks_like_logs(rows[0]):
                continue
            for index, row in enumerate(rows):
                case_id = _case_id(row, blob)
                if len(grouped[case_id]) >= MAX_LOG_ROWS_PER_CASE:
                    continue
                service = first_present(
                    row,
                    ["service", "component", "pod", "container", "container_name"],
                    default="unknown",
                )
                message = first_present(row, ["message", "msg", "log", "text", "event"], default=str(row))
                root_component, _ = _case_fault_tokens(case_id)
                if service != root_component and not _looks_error_text(message):
                    continue
                grouped[case_id].append(
                    {
                        "timestamp": timestamp(first_present(row, ["timestamp", "time", "ts"], default=str(index))),
                        "service": service,
                        "message": message,
                        "source_id": blob.source_id,
                    }
                )
        elif blob.suffix in {".log", ".txt"} and "trace" not in blob.name.lower():
            if Path(blob.name).name.lower() in {"inject_time.txt", "metrics_postprocess.log"}:
                continue
            for index, line in enumerate(blob.text.splitlines()):
                case_id = _case_id({}, blob)
                if len(grouped[case_id]) >= MAX_LOG_ROWS_PER_CASE:
                    continue
                if line.strip():
                    grouped[case_id].append(
                        {
                            "timestamp": index,
                            "service": _component_from_name(blob.name),
                            "message": line.strip(),
                            "source_id": blob.source_id,
                        }
                    )
    return grouped


def _load_trace_rows(blobs: list[SourceBlob]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for blob in blobs:
        if blob.suffix not in {".csv", ".json", ".jsonl"}:
            continue
        rows = _records_for_blob(blob)
        if not rows or not _looks_like_traces(rows[0], blob.name):
            continue
        for row in rows:
            caller = first_present(row, ["caller", "source", "parent", "src", "from", "upstream"])
            callee = first_present(row, ["callee", "target", "child", "dst", "to", "downstream"])
            if not caller or not callee:
                continue
            case_id = _case_id(row, blob)
            if len(grouped[case_id]) >= MAX_TRACE_ROWS_PER_CASE:
                continue
            grouped[case_id].append(
                {
                    "timestamp": timestamp(first_present(row, ["timestamp", "time", "ts"], default="0")),
                    "caller": caller,
                    "callee": callee,
                    "latency_ms": numeric(first_present(row, ["latency", "latency_ms", "duration", "duration_ms"], default="")),
                    "error": first_present(row, ["error", "status", "status_code"], default=""),
                    "source_id": blob.source_id,
                }
            )
    return grouped


def _build_episode(
    *,
    case_id: str,
    labels: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    log_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    max_evidence_spans: int,
    seed: int,
) -> ExternalEpisode | None:
    _normalize_case_timestamps(labels, metric_rows, log_rows, trace_rows)
    components = {
        row["service"] for row in metric_rows if row.get("service")
    } | {
        row["service"] for row in log_rows if row.get("service")
    } | {
        node for row in trace_rows for node in (row.get("caller"), row.get("callee")) if node
    }
    component = labels.get("fault_component") or (sorted(components)[0] if components else "")
    if not component:
        return None
    fault_type = labels.get("fault_type") or "unknown"
    root_label = labels.get("root_cause_label") or f"{component}:{fault_type}"

    topology = sorted({(str(row["caller"]), str(row["callee"])) for row in trace_rows})
    sensors = _sensor_frames(metric_rows)
    evidence = _evidence_pack(
        case_id=case_id,
        root_component=component,
        metric_rows=metric_rows,
        log_rows=log_rows,
        trace_rows=trace_rows,
        max_evidence_spans=max_evidence_spans,
        seed=seed,
    )
    gold_spans = [item["span_id"] for item in evidence if item.get("component") == component]
    if not gold_spans and evidence:
        gold_spans = [evidence[0]["span_id"]]
    candidate_components = sorted(components | {component})
    return ExternalEpisode(
        episode_id=f"rcaeval-re2-{slug(case_id)}",
        source_dataset="rcaeval_re2",
        domain="microservice",
        split="external",
        duration=_duration(sensors, metric_rows, log_rows, trace_rows),
        topology=topology,
        observations={
            "sensors": sensors,
            "logs": [item for item in evidence if item["kind"] == "log"],
            "traces": [item for item in evidence if item["kind"] == "trace"],
            "evidence": [item for item in evidence if item["kind"] == "metric"],
            "manuals": [],
        },
        candidate_actions=candidate_actions_for_components(candidate_components),
        ground_truth=ExternalGroundTruth(
            fault_component=component,
            fault_type=fault_type,
            root_cause_label=root_label,
            is_faulted=True,
            fault_start_time=labels.get("fault_start_time", 0),
            severity=labels.get("severity"),
            gold_evidence_spans=gold_spans,
            metadata={
                "label_source": "RCAEval RE2 annotations",
                "raw_label": labels.get("raw", {}),
            },
        ),
        supported_metrics=list(DEFAULT_SUPPORTED_METRICS),
        metadata={
            "external_validation": True,
            "dataset_role": "primary_microservice_external_trace",
            "candidate_root_causes": [
                f"{candidate}:{fault_type}" for candidate in candidate_components
            ],
        },
    )


def _sensor_frames(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, float]] = defaultdict(dict)
    for row in metric_rows:
        sensor = f"{row['service']}.{row['metric']}"
        grouped[int(row["timestamp"])][sensor] = float(row["value"])
    return [
        {"timestamp": time, "values": dict(sorted(values.items()))}
        for time, values in sorted(grouped.items())
    ]


def _evidence_pack(
    *,
    case_id: str,
    root_component: str,
    metric_rows: list[dict[str, Any]],
    log_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    max_evidence_spans: int,
    seed: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in _top_metric_deviations(metric_rows, limit=max(5, max_evidence_spans // 3)):
        span = f"re2-{stable_id(case_id, 'metric', row['service'], row['metric'], row['timestamp'])}"
        records.append(
            {
                "span_id": span,
                "source_id": row["source_id"],
                "kind": "metric",
                "timestamp": row["timestamp"],
                "component": row["service"],
                "text": (
                    f"Metric {row['metric']} on service {row['service']} changed by "
                    f"{row['delta']:.4g} from its case baseline."
                ),
            }
        )
    for row in log_rows:
        if row["service"] == root_component or _looks_error_text(row["message"]):
            span = f"re2-{stable_id(case_id, 'log', row['service'], row['timestamp'], row['message'])}"
            records.append(
                {
                    "span_id": span,
                    "source_id": row["source_id"],
                    "kind": "log",
                    "timestamp": row["timestamp"],
                    "component": row["service"],
                    "text": f"Log from {row['service']}: {row['message']}",
                }
            )
    for row in trace_rows:
        if row.get("latency_ms") is not None or _looks_error_text(str(row.get("error", ""))):
            span = f"re2-{stable_id(case_id, 'trace', row['caller'], row['callee'], row['timestamp'])}"
            records.append(
                {
                    "span_id": span,
                    "source_id": row["source_id"],
                    "kind": "trace",
                    "timestamp": row["timestamp"],
                    "component": row["callee"],
                    "text": (
                        f"Trace edge {row['caller']} -> {row['callee']} "
                        f"latency={row.get('latency_ms')} error={row.get('error', '')}."
                    ),
                }
            )
    records = select_deterministic(records, max_evidence_spans, seed, case_id)
    return sorted(records, key=lambda item: (item["timestamp"] if item["timestamp"] is not None else -1, item["span_id"]))


def _top_metric_deviations(metric_rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    baselines: dict[tuple[str, str], float] = {}
    ranked: list[dict[str, Any]] = []
    for row in sorted(metric_rows, key=lambda item: item["timestamp"]):
        key = (row["service"], row["metric"])
        baselines.setdefault(key, row["value"])
        delta = abs(row["value"] - baselines[key])
        if delta > 0:
            enriched = dict(row)
            enriched["delta"] = delta
            ranked.append(enriched)
    return sorted(ranked, key=lambda item: item["delta"], reverse=True)[:limit]


def _records_for_blob(blob: SourceBlob) -> list[dict[str, Any]]:
    if blob.suffix == ".csv":
        return read_csv_blob(blob)
    if blob.suffix in {".json", ".jsonl"}:
        return read_json_records_blob(blob)
    return []


def _case_id(row: dict[str, Any], blob: SourceBlob) -> str:
    explicit = first_present(
        row,
        ["case_id", "failure_id", "episode_id", "trace_id", "inject_id", "injection_id", "id"],
    )
    if explicit:
        return explicit
    path_case = _case_id_from_blob_path(blob)
    if path_case:
        return path_case
    path = Path(blob.name)
    if len(path.parts) > 1:
        return path.parts[-2]
    return path.stem


def _case_id_from_blob_path(blob: SourceBlob) -> str:
    parts = Path(blob.name).parts
    if len(parts) >= 4 and parts[-2].isdigit() and "_" in parts[-3]:
        return "/".join(parts[-4:-1]) if len(parts) >= 4 else "/".join(parts[-3:-1])
    if len(parts) >= 3 and parts[-2].isdigit():
        return "/".join(parts[-3:-1])
    return ""


def _label_from_blob_path(blob: SourceBlob) -> dict[str, Any] | None:
    case_id = _case_id_from_blob_path(blob)
    if not case_id:
        return None
    parts = Path(blob.name).parts
    if len(parts) < 3:
        return None
    fault_dir = parts[-3]
    if "_" not in fault_dir:
        return None
    component, fault_type = fault_dir.rsplit("_", 1)
    if not component or not fault_type:
        return None
    return {
        "case_id": case_id,
        "fault_component": component,
        "fault_type": slug(fault_type).replace("-", "_"),
        "root_cause_label": f"{component}:{slug(fault_type).replace('-', '_')}",
        "fault_start_time": _inject_time_for_blob(blob),
        "severity": None,
        "raw": {"source_path": blob.name, "path_derived": True},
    }


def _inject_time_for_blob(blob: SourceBlob) -> int:
    if Path(blob.name).name.lower() != "inject_time.txt":
        return 0
    text = blob.text.strip().splitlines()
    if not text:
        return 0
    return timestamp(text[0], fallback=0)


def _metric_records_from_row(row: dict[str, Any], blob: SourceBlob, case_id: str) -> list[dict[str, Any]]:
    row_time = timestamp(first_present(row, ["timestamp", "time", "ts"], default="0"))
    service = first_present(row, ["service", "component", "pod", "container", "container_name", "name"], default="")
    metric = first_present(row, ["metric", "kpi", "indicator", "sensor"], default="")
    value = numeric(first_present(row, ["value", metric], default=""))
    if metric and value is not None:
        return [
            {
                "timestamp": row_time,
                "service": service or _service_from_metric_name(metric),
                "metric": metric,
                "value": value,
                "source_id": blob.source_id,
            }
        ]

    records: list[dict[str, Any]] = []
    ignored = {"case_id", "failure_id", "episode_id", "trace_id", "inject_id", "injection_id", "id", "timestamp", "time", "ts"}
    for column, raw_value in row.items():
        if column.lower() in ignored:
            continue
        parsed = numeric(raw_value)
        if parsed is None:
            continue
        if not _keep_wide_metric_column(column, case_id, len(records)):
            continue
        records.append(
            {
                "timestamp": row_time,
                "service": _service_from_metric_name(column),
                "metric": column,
                "value": parsed,
                "source_id": blob.source_id,
            }
        )
    return records


def _keep_wide_metric_column(column: str, case_id: str, kept_count: int) -> bool:
    if kept_count >= MAX_WIDE_METRIC_COLUMNS:
        return False
    component, fault_type = _case_fault_tokens(case_id)
    lowered = column.lower()
    if component and component.lower() in lowered:
        return True
    if fault_type and fault_type.lower() in lowered:
        return True
    if any(token in lowered for token in ("error", "latency", "workload")):
        return True
    return kept_count < 12


def _case_fault_tokens(case_id: str) -> tuple[str, str]:
    parts = Path(case_id).parts
    if len(parts) >= 2:
        fault_dir = parts[-2]
    else:
        fault_dir = str(case_id)
    if "_" not in fault_dir:
        return "", ""
    component, fault_type = fault_dir.rsplit("_", 1)
    return component, fault_type


def _service_from_metric_name(metric: str) -> str:
    text = str(metric)
    separators = ["_container-", "_istio-", "_node-", "_"]
    for separator in separators:
        if separator in text:
            return text.split(separator, 1)[0]
    if "." in text:
        return text.split(".", 1)[0]
    if "-" in text:
        return text.split("-", 1)[0]
    return "unknown"


def _normalize_case_timestamps(
    labels: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    log_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
) -> None:
    times: list[int] = []
    for rows in (metric_rows, log_rows, trace_rows):
        times.extend(int(row["timestamp"]) for row in rows if row.get("timestamp") is not None)
    if labels.get("fault_start_time") is not None:
        times.append(int(labels["fault_start_time"]))
    if not times:
        return
    epoch_times = [time for time in times if time > 1_000_000_000]
    base = min(epoch_times) if epoch_times else min(times)
    for rows in (metric_rows, log_rows, trace_rows):
        for row in rows:
            if row.get("timestamp") is not None:
                row["timestamp"] = max(0, int(row["timestamp"]) - base)
    if labels.get("fault_start_time") is not None:
        labels["fault_start_time"] = max(0, int(labels["fault_start_time"]) - base)


def _looks_like_metrics(row: dict[str, Any]) -> bool:
    keys = {key.lower() for key in row}
    if keys & {"metric", "kpi", "indicator", "sensor", "value"}:
        return True
    return bool(_first_numeric_column(row))


def _looks_like_labels(row: dict[str, Any]) -> bool:
    keys = {key.lower() for key in row}
    root_fields = {
        "root_cause_service",
        "root_service",
        "root_cause",
        "root_cause_indicator",
        "fault_service",
        "fault_type",
    }
    return bool(keys & root_fields)


def _looks_like_logs(row: dict[str, Any]) -> bool:
    keys = {key.lower() for key in row}
    return bool(keys & {"message", "msg", "log", "text", "event"})


def _looks_like_traces(row: dict[str, Any], name: str) -> bool:
    keys = {key.lower() for key in row}
    has_edge = bool(keys & {"caller", "source", "parent", "src", "from", "upstream"}) and bool(
        keys & {"callee", "target", "child", "dst", "to", "downstream"}
    )
    return has_edge


def _filename_role(name: str) -> str:
    leaf = Path(name).name.lower()
    if any(token in leaf for token in ("label", "ground_truth", "ground-truth", "root_cause", "root-cause")):
        return "label"
    if any(token in leaf for token in ("trace", "span")):
        return "trace"
    if any(token in leaf for token in ("log", "event")):
        return "log"
    if any(token in leaf for token in ("metric", "kpi", "indicator")):
        return "metric"
    return "unknown"


def _first_numeric_column(row: dict[str, Any]) -> str:
    ignored = {"case_id", "timestamp", "time", "ts", "service", "component", "pod", "container"}
    for key, value in row.items():
        if key.lower() in ignored:
            continue
        if numeric(value) is not None:
            return key
    return ""


def _duration(
    sensors: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    log_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
) -> int:
    times = [item["timestamp"] for item in sensors]
    times.extend(row["timestamp"] for row in metric_rows)
    times.extend(row["timestamp"] for row in log_rows)
    times.extend(row["timestamp"] for row in trace_rows)
    if not times:
        return 0
    return max(times) - min(times) + 1


def _matches_system(name: str, systems: list[str]) -> bool:
    lowered = name.lower()
    aliases = {
        "ss": ["ss", "sock"],
        "ob": ["ob", "online", "boutique"],
        "tt": ["tt", "train", "ticket"],
    }
    for system in systems:
        for alias in aliases.get(system.lower(), [system.lower()]):
            if alias in lowered:
                return True
    return False


def _component_from_name(name: str) -> str:
    stem = Path(name).stem
    for token in ("log", "logs", "event", "events"):
        stem = stem.replace(token, "")
    return slug(stem).replace("-", "_") or "unknown"


def _looks_error_text(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("error", "fail", "timeout", "exception", "latency", "slow", "5xx"))


if __name__ == "__main__":
    raise SystemExit(main())
