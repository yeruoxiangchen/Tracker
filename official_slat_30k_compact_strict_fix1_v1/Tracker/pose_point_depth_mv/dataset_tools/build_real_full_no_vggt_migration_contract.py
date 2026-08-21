#!/usr/bin/env python3
"""Freeze a passed real-Full EMA checkpoint as no-VGGT migration parent."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pose_point_depth_mv.real_full_no_vggt_migration import (
    build_migration_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("ss", "slat"), required=True)
    parser.add_argument("--parent_checkpoint", required=True)
    parser.add_argument("--parent_report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min_real_objects", type=int, default=350)
    args = parser.parse_args()
    payload = build_migration_contract(
        stage=args.stage,
        parent_checkpoint=args.parent_checkpoint,
        parent_report=args.parent_report,
        min_real_objects=int(args.min_real_objects),
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and output.stat().st_size > 0:
        current = json.loads(output.read_text(encoding="utf-8"))
        for value in (current, payload):
            value.pop("created_at_utc", None)
        if current != payload:
            raise RuntimeError(f"refusing to overwrite changed contract: {output}")
    else:
        temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
