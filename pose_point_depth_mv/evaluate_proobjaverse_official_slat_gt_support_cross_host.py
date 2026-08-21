#!/usr/bin/env python3
"""Run official GT-support SLat evaluation after a content-audited host move.

The underlying evaluator is intentionally left unchanged.  This adapter only
bridges the two path strings frozen in a migrated SLat checkpoint's projected
Native-SS binding.  The live report/checkpoint must still match the frozen
SHA256 values and every non-path field must remain exactly equal.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch

from pose_point_depth_mv import (
    evaluate_proobjaverse_official_slat_gt_support as _base,
)
from pose_point_depth_mv.no_vggt_ss_evidence import load_no_vggt_ss_evidence
from pose_point_depth_mv.proobjaverse_official_slat_protocol import sha256_file


APPROVED_CHECKPOINT_BINDING_PATH_FIELDS = {
    "report": "report_sha256",
    "checkpoint": "checkpoint_sha256",
}
APPROVED_RELOCATED_PATH = "<APPROVED_CONTENT_ADDRESSED_EVAL_PATH>"


def validate_checkpoint_native_ss_binding_relocation(
    saved_binding: Any,
    runtime_binding: Any,
    *,
    allow_path_relocation: bool,
) -> dict[str, Any]:
    """Strictly validate an evaluation-only, content-addressed path move."""

    error_message = "migrated SLat checkpoint upstream Native SS differs"
    if saved_binding == runtime_binding:
        return {
            "path_relocated": False,
            "relocations": {},
            "all_non_path_fields_exact": True,
        }
    if not allow_path_relocation:
        raise RuntimeError(error_message)
    if not isinstance(saved_binding, dict) or not isinstance(runtime_binding, dict):
        raise RuntimeError(error_message)

    saved_normalized = copy.deepcopy(saved_binding)
    runtime_normalized = copy.deepcopy(runtime_binding)
    relocations: dict[str, dict[str, Any]] = {}
    for field, hash_field in APPROVED_CHECKPOINT_BINDING_PATH_FIELDS.items():
        saved_value = saved_binding.get(field)
        runtime_value = runtime_binding.get(field)
        saved_hash = saved_binding.get(hash_field)
        runtime_hash = runtime_binding.get(hash_field)
        if not all(
            isinstance(value, str) and bool(value)
            for value in (saved_value, runtime_value, saved_hash, runtime_hash)
        ):
            raise RuntimeError(f"{error_message}: malformed {field} binding")
        if saved_hash != runtime_hash:
            raise RuntimeError(f"{error_message}: {hash_field} changed")

        saved_path = Path(saved_value).expanduser()
        runtime_path = Path(runtime_value).expanduser()
        if not saved_path.is_absolute() or not runtime_path.is_absolute():
            raise RuntimeError(f"{error_message}: {field} paths must be absolute")
        try:
            runtime_resolved = runtime_path.resolve(strict=True)
        except (FileNotFoundError, OSError, RuntimeError) as error:
            raise RuntimeError(
                f"{error_message}: runtime {field} is not an existing file"
            ) from error
        if not runtime_resolved.is_file():
            raise RuntimeError(f"{error_message}: runtime {field} is not a file")
        runtime_actual_hash = sha256_file(runtime_resolved)
        if runtime_actual_hash != runtime_hash:
            raise RuntimeError(
                f"{error_message}: runtime {field} content SHA256 changed"
            )

        saved_resolved: Path | None
        saved_path_unavailable_reason: str | None = None
        try:
            saved_resolved = saved_path.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            # A path rooted on the source AutoDL host is expected not to exist
            # or be traversable after the checkpoint has been copied back (for
            # example, an A72 /root/autodl-fs path is not traversable by the
            # source-server user).  It is never opened by the evaluator.  Its
            # immutable hash is still part of the frozen binding and is checked
            # against the live artifact above.
            saved_resolved = None
            saved_path_unavailable_reason = type(error).__name__
        if saved_resolved is not None:
            if not saved_resolved.is_file() or sha256_file(saved_resolved) != saved_hash:
                raise RuntimeError(
                    f"{error_message}: saved {field} content SHA256 changed"
                )

        saved_normalized[field] = APPROVED_RELOCATED_PATH
        runtime_normalized[field] = APPROVED_RELOCATED_PATH
        if saved_value != runtime_value:
            relocations[field] = {
                "saved": saved_value,
                "runtime": runtime_value,
                "runtime_resolved": str(runtime_resolved),
                "saved_path_exists": saved_resolved is not None,
                "saved_path_unavailable_reason": saved_path_unavailable_reason,
                "content_sha256": runtime_actual_hash,
            }

    if saved_normalized != runtime_normalized:
        raise RuntimeError(f"{error_message} outside approved path relocation")
    return {
        "path_relocated": bool(relocations),
        "relocations": relocations,
        "all_non_path_fields_exact": True,
    }


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("SLat checkpoint payload must be a dictionary")
    return checkpoint


def main() -> None:
    original_make_parser = _base.make_parser
    original_upstream_binding = _base._upstream_binding

    def make_parser():
        parser = original_make_parser()
        parser.add_argument(
            "--allow_checkpoint_data_path_relocation",
            action="store_true",
            help=(
                "allow only report/checkpoint path relocation in a migrated "
                "checkpoint binding after frozen SHA256 verification"
            ),
        )
        return parser

    args = make_parser().parse_args()
    checkpoint = _load_checkpoint(args.checkpoint)
    saved_binding = checkpoint.get("data_identity", {}).get("native_ss")
    summary_binding = checkpoint.get("model_summary", {}).get("upstream_native_ss")
    if not isinstance(saved_binding, dict) or saved_binding != summary_binding:
        raise RuntimeError("SLat checkpoint Native-SS identity is internally inconsistent")

    _, runtime_deployment = load_no_vggt_ss_evidence(args.native_ss_report)
    runtime_binding = original_upstream_binding(runtime_deployment)
    transition = validate_checkpoint_native_ss_binding_relocation(
        saved_binding,
        runtime_binding,
        allow_path_relocation=bool(args.allow_checkpoint_data_path_relocation),
    )
    print(
        "[official_slat_gt_support:checkpoint_binding] "
        + json.dumps(transition, sort_keys=True, ensure_ascii=False),
        flush=True,
    )

    def checkpoint_compatible_upstream_binding(value: dict[str, Any]) -> dict[str, Any]:
        current = original_upstream_binding(value)
        validate_checkpoint_native_ss_binding_relocation(
            saved_binding,
            current,
            allow_path_relocation=bool(args.allow_checkpoint_data_path_relocation),
        )
        return copy.deepcopy(saved_binding)

    # The base evaluator still performs the full cache/live-deployment equality
    # check before this projected binding is used.  Only its checkpoint-facing
    # projection is replaced, and only after the strict validation above.
    _base.make_parser = make_parser
    _base._upstream_binding = checkpoint_compatible_upstream_binding
    try:
        _base.main()
    finally:
        _base.make_parser = original_make_parser
        _base._upstream_binding = original_upstream_binding


if __name__ == "__main__":
    main()
