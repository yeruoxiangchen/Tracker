#!/usr/bin/env python3
"""Derive a fixed uniform-4 protocol from the frozen Omni200 protocol.

The object/category selection, meshes, 24 registered cameras, renderer and
metric contract are copied verbatim.  Only the four input-camera indices are
changed, to the fixed equal-index interval [0, 6, 12, 18].
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from pose_point_depth_mv.dataset_tools.build_reconviagen_omni200_benchmark import (
    atomic_json,
    canonical_sha256,
    sha256_file,
    utc_now,
    validate_protocol,
)


UNIFORM4_INDICES = [0, 6, 12, 18]
DERIVATION_POLICY = "fixed_equal_index_interval_4_of_24_v1"


def _without_protocol_hash(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if key != "protocol_sha256"}


def derive(base_path: Path, output_path: Path) -> dict:
    base = json.loads(base_path.read_text(encoding="utf-8"))
    validate_protocol(base)
    if int(base.get("object_count", -1)) != 200 or len(base.get("objects", [])) != 200:
        raise RuntimeError("base Omni200 object matrix differs from frozen 200-object scope")
    if int(base.get("registered_camera_count", -1)) != 24:
        raise RuntimeError("base Omni200 camera count differs from 24")
    camera_indices = [int(row["view_index"]) for row in base["all_24_cameras"]]
    if camera_indices != list(range(24)):
        raise RuntimeError("base Omni200 registered-camera identity/order differs")

    base_sha256 = sha256_file(base_path)
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        validate_protocol(existing)
        derivation = existing.get("view_selection_derivation", {})
        if (
            derivation.get("base_protocol_sha256") != base_sha256
            or derivation.get("base_protocol_identity") != base["protocol_sha256"]
            or derivation.get("policy") != DERIVATION_POLICY
            or derivation.get("selected_input_view_indices") != UNIFORM4_INDICES
        ):
            raise RuntimeError(f"existing uniform4 protocol binding differs: {output_path}")
        if any(
            row.get("selected_input_view_indices") != UNIFORM4_INDICES
            for row in existing["objects"]
        ):
            raise RuntimeError("existing uniform4 per-object view matrix differs")
        return existing

    payload = copy.deepcopy(base)
    payload.pop("protocol_sha256", None)
    payload["created_at_utc"] = utc_now()
    payload["scope"] = (
        "Derived fixed-uniform-4 view comparison on the exact frozen Omni200 "
        "object/camera protocol; object selection is unchanged"
    )
    payload["selection_policy"] = (
        f"{base.get('selection_policy')} for objects; {DERIVATION_POLICY} for views"
    )
    payload["view_selection"] = "fixed camera indices [0,6,12,18] for every object"
    payload["view_selection_derivation"] = {
        "policy": DERIVATION_POLICY,
        "selected_input_view_indices": list(UNIFORM4_INDICES),
        "base_protocol": str(base_path),
        "base_protocol_sha256": base_sha256,
        "base_protocol_identity": str(base["protocol_sha256"]),
        "object_uid_order_sha256": canonical_sha256(
            [str(row["uid"]) for row in base["objects"]]
        ),
        "changed_scientific_variable": "input view selection only",
    }
    for row in payload["objects"]:
        row["selected_input_view_indices"] = list(UNIFORM4_INDICES)

    payload["protocol_sha256"] = canonical_sha256(_without_protocol_hash(payload))
    validate_protocol(payload)
    atomic_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_protocol", required=True)
    parser.add_argument("--output_protocol", required=True)
    args = parser.parse_args()
    base = Path(args.base_protocol).expanduser().resolve(strict=True)
    output = Path(args.output_protocol).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = derive(base, output)
    print(
        json.dumps(
            {
                "passed": True,
                "objects": len(payload["objects"]),
                "selected_input_view_indices": UNIFORM4_INDICES,
                "protocol": str(output),
                "protocol_sha256": payload["protocol_sha256"],
                "base_protocol_sha256": payload["view_selection_derivation"][
                    "base_protocol_sha256"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
