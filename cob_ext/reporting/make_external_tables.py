"""Create CSV and TeX-format tables for external validation summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from causalopsbench._io import ordered_fieldnames, write_csv_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build external validation table artifacts.")
    parser.add_argument("--summary-csv", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    rows = []
    for path in args.summary_csv:
        rows.extend(_read_rows(Path(path)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "external_portability_summary.csv", rows)
    _write_latex(output_dir / "external_portability_table.tex", rows)
    print(json.dumps({"output_dir": str(output_dir), "rows": len(rows)}, sort_keys=True))
    return 0


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv_rows(path, rows, fieldnames=_fieldnames(rows), extrasaction="ignore")


def _write_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("% No external validation rows.\n", encoding="utf-8")
        return
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "System & EPS & Detect. & RCA@1 & RCA@$k$ & Evid. & Parse \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            "{} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\".format(
                _tex(row.get("system", "")),
                _float(row.get("external_portability_score")),
                _float(row.get("detection")),
                _float(row.get("root_cause_top1")),
                _float(row.get("root_cause_topk")),
                _float(row.get("evidence_f1")),
                _float(row.get("parsing_success")),
            )
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "system",
        "model_spec",
        "count",
        "external_portability_score",
        "detection",
        "root_cause_top1",
        "root_cause_topk",
        "evidence_f1",
        "calibration",
        "parsing_success",
        "runtime_efficiency",
    ]
    return ordered_fieldnames(rows, preferred)


def _tex(value: Any) -> str:
    return str(value).replace("_", "\\_").replace("&", "\\&")


if __name__ == "__main__":
    raise SystemExit(main())
