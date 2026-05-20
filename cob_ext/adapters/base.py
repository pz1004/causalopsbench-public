"""Shared adapter utilities for external public datasets."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable
import zipfile

from cob_ext.schemas import ExternalActionSpec, ExternalEvidenceRecord, write_jsonl


@dataclass(frozen=True)
class SourceBlob:
    source_id: str
    name: str
    suffix: str
    data: bytes

    @property
    def text(self) -> str:
        return self.data.decode("utf-8", errors="replace")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


def collect_source_blobs(raw_dir: str | Path, suffixes: set[str] | None = None) -> list[SourceBlob]:
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {root}")
    blobs: list[SourceBlob] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        suffix = path.suffix.lower()
        if suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                for member in sorted(item for item in archive.namelist() if not item.endswith("/")):
                    member_suffix = Path(member).suffix.lower()
                    if suffixes is not None and member_suffix not in suffixes:
                        continue
                    data = archive.read(member)
                    blobs.append(
                        SourceBlob(
                            source_id=f"{path.name}:{member}",
                            name=member,
                            suffix=member_suffix,
                            data=data,
                        )
                    )
            continue
        if suffixes is not None and suffix not in suffixes:
            continue
        blobs.append(SourceBlob(source_id=str(path.relative_to(root)), name=path.name, suffix=suffix, data=path.read_bytes()))
    return blobs


def read_csv_blob(blob: SourceBlob) -> list[dict[str, str]]:
    text = blob.text.replace("\ufeff", "")
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    return [
        {str(key).strip(): str(value).strip() for key, value in row.items() if key is not None}
        for row in reader
    ]


def read_json_records_blob(blob: SourceBlob) -> list[dict[str, Any]]:
    text = blob.text.strip()
    if not text:
        return []
    if blob.suffix == ".jsonl":
        records = []
        for line in text.splitlines():
            if line.strip():
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    records.append(parsed)
        return records
    parsed = json.loads(text)
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        for key in ("records", "data", "items", "cases", "failures"):
            value = parsed.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [parsed]
    return []


def write_episodes_and_manifest(
    episodes_path: str | Path,
    manifest_path: str | Path,
    episodes: Iterable[Any],
    *,
    dataset: str,
    raw_dir: str | Path,
    source_blobs: list[SourceBlob],
    config: dict[str, Any],
) -> None:
    episode_list = list(episodes)
    for episode in episode_list:
        episode.validate()
    write_jsonl(episodes_path, episode_list)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "raw_dir": str(raw_dir),
        "episode_count": len(episode_list),
        "config": config,
        "source_files": [
            {
                "source_id": blob.source_id,
                "name": blob.name,
                "bytes": len(blob.data),
                "sha256": blob.sha256,
            }
            for blob in source_blobs
        ],
    }
    write_simple_yaml(manifest_path, manifest)


def write_simple_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_yaml_value(payload), encoding="utf-8")


def stable_id(*parts: Any, length: int = 16) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def slug(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "unknown"


def numeric(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return parsed


def timestamp(value: Any, fallback: int = 0) -> int:
    parsed = numeric(value)
    if parsed is not None:
        if parsed > 1e17:
            return int(parsed / 1_000_000_000)
        if parsed > 1e14:
            return int(parsed / 1_000_000)
        if parsed > 1e11:
            return int(parsed / 1_000)
        return int(parsed)
    text = str(value or "").strip()
    if not text:
        return fallback
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return int(datetime.strptime(text[:19], fmt).timestamp() // 60)
        except ValueError:
            pass
    return fallback


def first_present(row: dict[str, Any], names: Iterable[str], default: str = "") -> str:
    lowered = {key.lower().strip(): value for key, value in row.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def evidence_records_from_observations(observations: dict[str, Any]) -> list[ExternalEvidenceRecord]:
    records: list[ExternalEvidenceRecord] = []
    for key in ("manuals", "logs", "traces", "evidence"):
        for item in observations.get(key, []):
            if isinstance(item, dict) and "span_id" in item:
                records.append(
                    ExternalEvidenceRecord(
                        span_id=str(item["span_id"]),
                        source_id=str(item.get("source_id", "")),
                        kind=str(item.get("kind", key.rstrip("s") or "evidence")),
                        timestamp=_optional_int(item.get("timestamp")),
                        text=str(item.get("text", "")),
                        component=_optional_component(item.get("component")),
                        proxy=bool(item.get("proxy", False)),
                        metadata=dict(item.get("metadata", {})),
                    )
                )
    return records


def candidate_actions_for_components(components: Iterable[str]) -> list[ExternalActionSpec]:
    unique = sorted({component for component in components if component})
    return [
        ExternalActionSpec(
            action_id=f"inspect-{slug(component)}",
            target_component=component,
            action_type="investigate",
            description=f"Inspect external trace evidence for {component}.",
            cost=1.0,
            safety_risk=0.0,
        )
        for component in unique
    ]


def select_deterministic(items: list[Any], limit: int, seed: int, key: str) -> list[Any]:
    if limit <= 0 or len(items) <= limit:
        return list(items)
    rng = random.Random(f"{seed}:{key}")
    indexed = list(items)
    rng.shuffle(indexed)
    return indexed[:limit]


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _optional_component(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _yaml_value(value: Any, indent: int = 0) -> str:
    prefix = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(_yaml_value(item, indent + 2).rstrip("\n"))
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(item)}")
        return "\n".join(lines) + "\n"
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, dict):
                lines.append(f"{prefix}-")
                lines.append(_yaml_value(item, indent + 2).rstrip("\n"))
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
        return "\n".join(lines) + "\n"
    return f"{prefix}{_yaml_scalar(value)}\n"


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if not text or any(char in text for char in ":#[]{}&,*!?|>'\"%@`\\\n"):
        return json.dumps(text)
    return text
