#!/usr/bin/env python3
"""Relocate paths in a frozen official SLat split without changing membership."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pose_point_depth_mv.proobjaverse_official_slat_protocol import (
    SPLIT_FORMAT,
    atomic_json,
    canonical_sha256,
    load_json,
    sha256_file,
    validate_frozen_selection,
)


def relocate(
    *,
    source_split: Path,
    selection_path: Path,
    data_root: Path,
    output: Path,
    expected_protocol_sha256: str,
    expected_name: str,
    protocol_manifest: Path | None,
) -> dict[str, Any]:
    split = load_json(source_split)
    split_body = dict(split)
    saved_manifest_identity = str(split_body.pop("manifest_sha256", ""))
    if (
        split.get("format") != SPLIT_FORMAT
        or saved_manifest_identity != canonical_sha256(split_body)
        or split.get("name") != expected_name
        or split.get("protocol_sha256") != expected_protocol_sha256
    ):
        raise RuntimeError("source split frozen identity differs")

    selection = load_json(selection_path)
    selection_binding = validate_frozen_selection(selection)
    selected_by_uid = {
        str(row["uid"]): (rank, row)
        for rank, row in enumerate(selection_binding["rows"])
    }
    rows: list[dict[str, Any]] = []
    for old in split.get("rows", []):
        uid = str(old.get("uid", ""))
        if uid not in selected_by_uid:
            raise RuntimeError(f"split UID is absent from frozen selection: {uid}")
        rank, selected = selected_by_uid[uid]
        render = (data_root / selected["render"]["path"]).resolve()
        slat = (data_root / selected["slat"]["path"]).resolve()
        if (
            str(old.get("shard")) != str(selected["shard"])
            or int(old.get("selection_rank", -1)) != int(rank)
            or int(old.get("render_size", -1)) != int(selected["render"]["size"])
            or int(old.get("slat_size", -1)) != int(selected["slat"]["size"])
        ):
            raise RuntimeError(f"split/selection row identity differs: {uid}")
        if (
            not render.is_file()
            or not slat.is_file()
            or render.stat().st_size != int(selected["render"]["size"])
            or slat.stat().st_size != int(selected["slat"]["size"])
        ):
            raise RuntimeError(f"relocated payload is missing or changed: {uid}")
        rows.append(
            {
                **old,
                "render_tar": str(render),
                "slat_npz": str(slat),
            }
        )

    if len(rows) != int(split.get("count", -1)) or len(
        {str(row["uid"]) for row in rows}
    ) != len(rows):
        raise RuntimeError("relocated split count/UID accounting differs")
    if protocol_manifest is not None:
        protocol = load_json(protocol_manifest)
        if protocol.get("protocol_sha256") != expected_protocol_sha256:
            raise RuntimeError("protocol manifest identity differs")

    result = {
        **split_body,
        "protocol": (
            str(protocol_manifest.resolve())
            if protocol_manifest is not None
            else str(split.get("protocol", ""))
        ),
        "rows": rows,
        "path_relocation": {
            "source_split": str(source_split.resolve()),
            "source_split_sha256": sha256_file(source_split),
            "selection": str(selection_path.resolve()),
            "selection_sha256": sha256_file(selection_path),
            "data_root": str(data_root.resolve()),
            "membership_and_order_changed": False,
            "protocol_sha256_changed": False,
        },
    }
    result["manifest_sha256"] = canonical_sha256(result)
    if output.exists():
        if load_json(output) != result:
            raise RuntimeError(f"refusing to overwrite changed relocation: {output}")
    else:
        atomic_json(output, result)
    return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_split", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected_protocol_sha256", required=True)
    parser.add_argument("--expected_name", required=True)
    parser.add_argument("--protocol_manifest", default="")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    result = relocate(
        source_split=Path(args.source_split).expanduser().resolve(strict=True),
        selection_path=Path(args.selection).expanduser().resolve(strict=True),
        data_root=Path(args.data_root).expanduser().resolve(strict=True),
        output=Path(args.output).expanduser().resolve(),
        expected_protocol_sha256=str(args.expected_protocol_sha256),
        expected_name=str(args.expected_name),
        protocol_manifest=(
            None
            if not args.protocol_manifest
            else Path(args.protocol_manifest).expanduser().resolve(strict=True)
        ),
    )
    print(
        {
            "passed": True,
            "name": result["name"],
            "count": result["count"],
            "protocol_sha256": result["protocol_sha256"],
            "manifest_sha256": result["manifest_sha256"],
        }
    )


if __name__ == "__main__":
    main()
