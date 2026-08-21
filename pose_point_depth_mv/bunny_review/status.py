#!/usr/bin/env python3
"""Print concise status for a Bunny reconstruction review directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .common import load_method_result, load_protocol, method_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--methods",
        default="reference,pixal3d,reconviagen_stock,trained_full",
    )
    parser.add_argument(
        "--require_complete",
        action="store_true",
        help="exit 3 unless every requested method has a valid frozen result",
    )
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    output = {
        "protocol": str(protocol_path),
        "protocol_sha256": protocol["protocol_sha256"],
        "views": protocol["view_indices"],
        "single_view": protocol["single_view_index"],
        "methods": {},
        "comparisons": [],
    }
    for method_id in [item.strip() for item in args.methods.split(",") if item.strip()]:
        result_path = method_dir(protocol_path, method_id) / "result.json"
        if not result_path.is_file():
            output["methods"][method_id] = {
                "complete": False,
                "result": str(result_path),
            }
            continue
        try:
            result = load_method_result(protocol_path, method_id)
        except Exception as exc:
            output["methods"][method_id] = {
                "complete": False,
                "result": str(result_path),
                "error": repr(exc),
            }
        else:
            output["methods"][method_id] = {
                "complete": True,
                "display_name": result["display_name"],
                "mesh": result["mesh"]["path"],
                "mesh_bytes": result["mesh"]["bytes"],
                "input_view_indices": result["input_view_indices"],
                "backend_kind": result["backend"].get("kind"),
            }
    comparison_root = protocol_path.parent / "comparison"
    if comparison_root.is_dir():
        output["comparisons"] = sorted(
            str(path)
            for path in comparison_root.glob("*/report.json")
            if path.is_file()
        )
    output["all_requested_methods_complete"] = all(
        row.get("complete") is True for row in output["methods"].values()
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))
    if args.require_complete and not output["all_requested_methods_complete"]:
        sys.exit(3)


if __name__ == "__main__":
    main()
