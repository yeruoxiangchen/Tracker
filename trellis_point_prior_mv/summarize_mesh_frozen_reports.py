#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_KEYS = [
    "coord_count",
    "stage2_topk_ratio_to_target",
    "sparse_iou",
    "sparse_pred_precision",
    "sparse_target_recall",
    "chamfer_l2_mean",
    "target_to_mesh_mean",
    "mesh_to_target_mean",
    "extent_ratio",
    "vertex_count",
    "face_count",
]


def load_report(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:.0f}"
        return f"{value:.5f}".rstrip("0").rstrip(".")
    return str(value)


def report_label(path: Path, args: argparse.Namespace) -> str:
    if args.use_parent_label:
        return path.parent.name
    return path.stem if path.name != "report.json" else path.parent.name


def collect_rows(path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    report = load_report(path)
    summary = report.get("summary", {}).get("by_mode", {})
    modes = args.modes or sorted(summary)
    out = []
    for mode in modes:
        if mode not in summary:
            continue
        item = summary[mode]
        row: dict[str, Any] = {"report": report_label(path, args), "mode": mode, "count": item.get("count")}
        for key in args.keys:
            row[key] = item.get(f"{key}_mean")
        out.append(row)
    return out


def print_markdown(rows: list[dict[str, Any]], keys: list[str]) -> None:
    headers = ["report", "mode", "count", *keys]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print("| " + " | ".join(fmt(row.get(h)) for h in headers) + " |")


def print_csv(rows: list[dict[str, Any]], keys: list[str]) -> None:
    headers = ["report", "mode", "count", *keys]
    print(",".join(headers))
    for row in rows:
        print(",".join(fmt(row.get(h)) for h in headers))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize frozen downstream mesh report.json files.")
    parser.add_argument("reports", nargs="+", help="One or more report.json paths.")
    parser.add_argument("--modes", default=None, help="Comma-separated mode list. Default: all modes in each report.")
    parser.add_argument("--keys", default=",".join(DEFAULT_KEYS), help="Comma-separated metric keys without _mean suffix.")
    parser.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    parser.add_argument("--use_parent_label", action="store_true", default=True)
    args = parser.parse_args()
    args.keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    args.modes = [m.strip() for m in args.modes.split(",") if m.strip()] if args.modes else None
    return args


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for report in args.reports:
        rows.extend(collect_rows(Path(report), args))
    if args.format == "csv":
        print_csv(rows, args.keys)
    else:
        print_markdown(rows, args.keys)


if __name__ == "__main__":
    main()
