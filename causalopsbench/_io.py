"""Private serialization helpers shared by benchmark tools."""

from __future__ import annotations

import csv
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterable


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses recursively into JSON-compatible containers."""
    if is_dataclass(value):
        return {field.name: to_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value


def dumps_sorted_json(value: Any) -> str:
    return json.dumps(to_jsonable(value), sort_keys=True)


def load_jsonl(path: str | Path, parser: Callable[[dict[str, Any]], Any]) -> list[Any]:
    return [
        parser(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: str | Path, items: Iterable[Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [dumps_sorted_json(item) for item in items]
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_json_document(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv_rows(
    path: str | Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str] | None = None,
    extrasaction: str = "raise",
    write_header_when_empty: bool = False,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if fieldnames is not None and write_header_when_empty:
            with output.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=fieldnames, extrasaction=extrasaction).writeheader()
            return
        output.write_text("", encoding="utf-8")
        return
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames or list(rows[0].keys()),
            extrasaction=extrasaction,
        )
        writer.writeheader()
        writer.writerows(rows)


def ordered_fieldnames(rows: list[dict[str, Any]], preferred: Iterable[str] = ()) -> list[str]:
    seen = {field for row in rows for field in row}
    ordered = [field for field in preferred if field in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered
