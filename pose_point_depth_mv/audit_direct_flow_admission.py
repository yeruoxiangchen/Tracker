#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pose_point_depth_mv.direct_flow import (
    lifting_cache_identity,
    validate_n3_checkpoint,
)


ADMISSION_VERSION = "pose_point_depth_mv.direct_flow_admission.v1"


def load_report(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report is not a JSON object: {resolved}")
    return payload


def report_bound_to_identity(
    report: dict[str, Any], identity: dict[str, Any]
) -> bool:
    report_manifest = report.get("cache_manifest")
    if not report_manifest:
        return False
    if Path(str(report_manifest)).resolve() != Path(identity["manifest"]).resolve():
        return False
    for key in (
        "manifest_sha256",
        "cache_schema_hash",
        "cache_config_hash",
        "uid_hash",
        "object_uid_hash",
    ):
        if key in report and report[key] != identity.get(key):
            return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hard admission gate for full-data direct physical Flow training."
    )
    parser.add_argument("--train_cache_manifest", required=True)
    parser.add_argument("--eval_cache_manifest", required=True)
    parser.add_argument("--train_cache_audit", required=True)
    parser.add_argument("--eval_cache_audit", required=True)
    parser.add_argument("--train_stock_audit", required=True)
    parser.add_argument("--eval_stock_audit", required=True)
    parser.add_argument("--n3_report", required=True)
    parser.add_argument("--correspondence_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min_train_samples", type=int, default=1259)
    parser.add_argument("--min_train_objects", type=int, default=897)
    parser.add_argument("--min_eval_samples", type=int, default=175)
    parser.add_argument("--min_eval_objects", type=int, default=128)
    parser.add_argument("--fail_on_error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_manifest = load_report(args.train_cache_manifest)
    eval_manifest = load_report(args.eval_cache_manifest)
    train_rows = list(train_manifest.get("samples", ()))
    eval_rows = list(eval_manifest.get("samples", ()))
    train_identity = lifting_cache_identity(
        args.train_cache_manifest,
        rows=train_rows,
    )
    eval_identity = lifting_cache_identity(
        args.eval_cache_manifest,
        rows=eval_rows,
    )
    train_objects = {
        str(row.get("object_uid", row.get("uid", ""))) for row in train_rows
    }
    eval_objects = {
        str(row.get("object_uid", row.get("uid", ""))) for row in eval_rows
    }
    train_uids = [str(row.get("uid", "")) for row in train_rows]
    eval_uids = [str(row.get("uid", "")) for row in eval_rows]
    overlap = sorted(train_objects & eval_objects)
    train_cache_audit = load_report(args.train_cache_audit)
    eval_cache_audit = load_report(args.eval_cache_audit)
    train_stock_audit = load_report(args.train_stock_audit)
    eval_stock_audit = load_report(args.eval_stock_audit)
    n3_audit = validate_n3_checkpoint(
        args.n3_report,
        args.correspondence_checkpoint,
    )
    checks = {
        "train_sample_count": len(train_rows) >= int(args.min_train_samples),
        "train_object_count": len(train_objects) >= int(args.min_train_objects),
        "eval_sample_count": len(eval_rows) >= int(args.min_eval_samples),
        "eval_object_count": len(eval_objects) >= int(args.min_eval_objects),
        "train_cache_audit_passed": train_cache_audit.get("passed") is True,
        "eval_cache_audit_passed": eval_cache_audit.get("passed") is True,
        "train_stock_condition_native": train_stock_audit.get("passed") is True,
        "eval_stock_condition_native": eval_stock_audit.get("passed") is True,
        "train_cache_audit_bound": report_bound_to_identity(
            train_cache_audit, train_identity
        ),
        "eval_cache_audit_bound": report_bound_to_identity(
            eval_cache_audit, eval_identity
        ),
        "train_stock_audit_bound": report_bound_to_identity(
            train_stock_audit, train_identity
        ),
        "eval_stock_audit_bound": report_bound_to_identity(
            eval_stock_audit, eval_identity
        ),
        "train_uid_unique": len(train_uids) == len(set(train_uids)),
        "eval_uid_unique": len(eval_uids) == len(set(eval_uids)),
        "cache_schema_match": train_identity["cache_schema_hash"]
        == eval_identity["cache_schema_hash"],
        "object_disjoint": not overlap,
        "n3_checkpoint_bound": n3_audit.get("n3_passed") is True,
    }
    report = {
        "stage": "full-data direct physical Flow admission",
        "format": ADMISSION_VERSION,
        "passed": all(checks.values()),
        "checks": checks,
        "train_identity": train_identity,
        "eval_identity": eval_identity,
        "train_sample_count": len(train_rows),
        "train_object_count": len(train_objects),
        "eval_sample_count": len(eval_rows),
        "eval_object_count": len(eval_objects),
        "object_overlap_count": len(overlap),
        "object_overlap": overlap,
        "n3_audit": n3_audit,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if args.fail_on_error and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
