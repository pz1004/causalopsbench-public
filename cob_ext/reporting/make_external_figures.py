"""Create lightweight SVG figures for external validation summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build external validation SVG figures.")
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    rows = _read_rows(Path(args.summary_csv))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_bar_svg(output_dir / "external_portability_eps.svg", rows)
    print(json.dumps({"output": str(output_dir / "external_portability_eps.svg"), "rows": len(rows)}, sort_keys=True))
    return 0


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_bar_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    width = 760
    row_height = 30
    height = 60 + row_height * max(1, len(rows))
    labels = [str(row.get("system", "")) for row in rows]
    values = [_float(row.get("external_portability_score")) for row in rows]
    max_value = max(values + [1.0])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="28" font-family="Arial" font-size="16" font-weight="bold">External Portability Score</text>',
    ]
    for index, (label, value) in enumerate(zip(labels, values)):
        y = 52 + index * row_height
        bar_width = int(500 * value / max_value)
        lines.append(f'<text x="20" y="{y + 15}" font-family="Arial" font-size="12">{_xml(label[:38])}</text>')
        lines.append(f'<rect x="220" y="{y}" width="{bar_width}" height="18" fill="#2f6f8f"/>')
        lines.append(f'<text x="{230 + bar_width}" y="{y + 14}" font-family="Arial" font-size="12">{value:.3f}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _xml(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    raise SystemExit(main())
