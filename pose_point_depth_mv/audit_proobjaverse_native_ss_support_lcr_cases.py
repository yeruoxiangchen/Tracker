"""Select and package Native-SS support/LCR failure cases for manual audit.

The selector is deliberately tied to an already aggregated held-out report.  It
does not rerank checkpoints or change the benchmark.  The companion shell job
reruns the selected one-object slices with ``--save_meshes`` so that the exact
GT, Stock, Native, and Native-trained geometry can be inspected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


AGGREGATE_FORMAT = (
    "pose_point_depth_mv.proobjaverse_official_native_ss_slat_end_to_end_eval.v1"
)
SELECTION_FORMAT = "pose_point_depth_mv.official_ss_support_lcr_selection.v1"
AUDIT_FORMAT = "pose_point_depth_mv.official_ss_support_lcr_audit.v1"
ACTIVE_LIMIT_MARKER = "SLat decoder input exceeds safe active-point limit"


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _manifest_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = _load(path)
    rows = payload.get("samples", payload.get("rows"))
    if not isinstance(rows, list):
        raise ValueError(f"manifest has no sample rows: {path}")
    return [dict(row) for row in rows]


def _worker_records(aggregate: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ss_records: list[dict[str, Any]] = []
    mesh_records: list[dict[str, Any]] = []
    for binding in aggregate["worker_reports"]:
        path = Path(binding["path"]).expanduser().resolve(strict=True)
        if _sha256(path) != str(binding["sha256"]):
            raise RuntimeError(f"worker report hash differs: {path}")
        worker = _load(path)
        if worker.get("complete") is not True:
            raise RuntimeError(f"worker report is incomplete: {path}")
        ss_records.extend(dict(row) for row in worker["ss_records"])
        mesh_records.extend(dict(row) for row in worker["mesh_branch_records"])
    return ss_records, mesh_records


def _metric_row(transfer: dict[str, Any], uid: str) -> dict[str, Any] | None:
    for row in transfer["object_rows"]:
        if str(row["object_uid"]) == uid:
            return dict(row)
    return None


def select_cases(args: argparse.Namespace) -> None:
    aggregate_path = Path(args.aggregate_report).expanduser().resolve(strict=True)
    cache_path = Path(args.cache_manifest).expanduser().resolve(strict=True)
    aggregate = _load(aggregate_path)
    if aggregate.get("format") != AGGREGATE_FORMAT:
        raise ValueError(f"unexpected aggregate format: {aggregate.get('format')}")
    if int(aggregate.get("object_count", -1)) != 48 or int(
        aggregate.get("record_count", -1)
    ) != 144:
        raise ValueError("selector requires the exact held-out Dev48/three-seed report")
    if aggregate.get("integrity", {}).get("object_coverage_exact") is not True:
        raise RuntimeError("aggregate object coverage is not exact")

    rows = _manifest_rows(cache_path)
    index_by_uid = {
        str(row.get("object_uid", row.get("uid"))): index
        for index, row in enumerate(rows)
    }
    if len(index_by_uid) != len(rows):
        raise RuntimeError("cache manifest does not contain unique object UIDs")

    ss_records, mesh_records = _worker_records(aggregate)
    failures_by_uid: dict[str, list[dict[str, Any]]] = {}
    for row in mesh_records:
        error = row.get("error") or {}
        message = str(error.get("message", ""))
        if row.get("passed") is False and ACTIVE_LIMIT_MARKER in message:
            failures_by_uid.setdefault(str(row["object_uid"]), []).append(
                {
                    "seed": int(row["seed"]),
                    "branch": str(row["branch"]),
                    "active_point_count": int(row["slat_active_point_count"]),
                    "active_point_limit": int(row["slat_active_point_limit"]),
                    "error": dict(error),
                }
            )
    if not failures_by_uid:
        raise RuntimeError("aggregate contains no recorded active-support explosion")
    support_uids = sorted(
        failures_by_uid,
        key=lambda uid: max(
            row["active_point_count"] for row in failures_by_uid[uid]
        ),
        reverse=True,
    )
    if int(args.expected_support_objects) > 0 and len(support_uids) != int(
        args.expected_support_objects
    ):
        raise RuntimeError(
            f"expected {args.expected_support_objects} support failures, got {len(support_uids)}"
        )

    trained = aggregate["trained_slat_end_to_end_transfer"]
    lcr_rows = sorted(
        (
            dict(row)
            for row in trained["object_rows"]
            if str(row["object_uid"]) not in failures_by_uid
        ),
        key=lambda row: float(row["largest_component_ratio_delta"]),
    )
    if len(lcr_rows) < int(args.lcr_count):
        raise RuntimeError("not enough valid objects for the requested LCR audit")

    cases: list[dict[str, Any]] = []
    for rank, uid in enumerate(support_uids, start=1):
        if uid not in index_by_uid:
            raise KeyError(f"support-failure UID is absent from cache: {uid}")
        source_ss = [dict(row) for row in ss_records if str(row["object_uid"]) == uid]
        cases.append(
            {
                "label": f"00_support_explosion_{rank}_{uid[:12]}",
                "kind": "support_explosion",
                "rank": rank,
                "object_uid": uid,
                "object_index": int(index_by_uid[uid]),
                "object_end": int(index_by_uid[uid]) + 1,
                "original_ss_records": source_ss,
                "original_failures": failures_by_uid[uid],
                "aggregate_metrics": {
                    name: _metric_row(aggregate[name], uid)
                    for name in (
                        "stock_slat_mesh_transfer",
                        "trained_slat_end_to_end_transfer",
                        "trained_slat_increment_on_native_support",
                    )
                },
            }
        )
    for rank, row in enumerate(lcr_rows[: int(args.lcr_count)], start=1):
        uid = str(row["object_uid"])
        if uid not in index_by_uid:
            raise KeyError(f"LCR UID is absent from cache: {uid}")
        cases.append(
            {
                "label": f"{rank:02d}_lcr_worst_{rank}_{uid[:12]}",
                "kind": "lcr_worst",
                "rank": rank,
                "object_uid": uid,
                "object_index": int(index_by_uid[uid]),
                "object_end": int(index_by_uid[uid]) + 1,
                "original_ss_records": [
                    dict(item)
                    for item in ss_records
                    if str(item["object_uid"]) == uid
                ],
                "original_failures": [],
                "aggregate_metrics": {
                    name: _metric_row(aggregate[name], uid)
                    for name in (
                        "stock_slat_mesh_transfer",
                        "trained_slat_end_to_end_transfer",
                        "trained_slat_increment_on_native_support",
                    )
                },
            }
        )

    rerun_root = Path(args.rerun_root).expanduser().resolve()
    for case in cases:
        case["rerun_output"] = str(rerun_root / case["label"])
    selection = {
        "format": SELECTION_FORMAT,
        "passed": True,
        "formal": False,
        "aggregate_report": str(aggregate_path),
        "aggregate_report_sha256": _sha256(aggregate_path),
        "cache_manifest": str(cache_path),
        "cache_manifest_sha256": _sha256(cache_path),
        "selection_policy": (
            "all_active_support_limit_failures_plus_bottom3_lcr_from_"
            "trained_slat_end_to_end_transfer"
        ),
        "support_object_count": len(support_uids),
        "lcr_object_count": int(args.lcr_count),
        "cases": cases,
        "scope_guard": (
            "post-hoc failure diagnosis only; selected cases cannot replace the "
            "frozen Dev48 aggregate"
        ),
    }
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        existing = _load(output)
        if existing != selection:
            raise RuntimeError(f"existing selection differs: {output}")
    else:
        _write(output, selection)
    print(
        json.dumps(
            {
                "passed": True,
                "selection": str(output),
                "cases": [
                    {
                        "kind": case["kind"],
                        "rank": case["rank"],
                        "index": case["object_index"],
                        "uid": case["object_uid"],
                        "rerun_output": case["rerun_output"],
                    }
                    for case in cases
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _export_target(cache: Path, output: Path) -> None:
    with np.load(cache) as payload:
        mesh = trimesh.Trimesh(
            vertices=np.asarray(payload["vertices"]),
            faces=np.asarray(payload["faces"]),
            process=False,
        )
    mesh.export(output)


def finalize(args: argparse.Namespace) -> None:
    selection_path = Path(args.selection).expanduser().resolve(strict=True)
    selection = _load(selection_path)
    if selection.get("format") != SELECTION_FORMAT or selection.get("passed") is not True:
        raise ValueError("selection is not a passed support/LCR selection")

    cases_out: list[dict[str, Any]] = []
    audit_passed = True
    for case in selection["cases"]:
        rerun = Path(case["rerun_output"]).expanduser().resolve(strict=True)
        report_path = rerun / "report.json"
        report = _load(report_path)
        if report.get("complete") is not True or int(report.get("object_count", -1)) != 1:
            raise RuntimeError(f"targeted rerun is incomplete: {report_path}")
        if str(report["run_identity"]["object_uids"][0]) != str(case["object_uid"]):
            raise RuntimeError(f"targeted rerun UID differs: {report_path}")
        target_caches = list((rerun / "target_mesh_cache").glob("*.npz"))
        if len(target_caches) != 1:
            raise RuntimeError(f"expected one target Mesh cache: {rerun}")
        target_obj = rerun / "GT_target_mesh.obj"
        if not target_obj.is_file():
            _export_target(target_caches[0], target_obj)

        failures = []
        branch_rows = []
        for row in report["mesh_branch_records"]:
            reduced = {
                "object_uid": str(row["object_uid"]),
                "seed": int(row["seed"]),
                "branch": str(row["branch"]),
                "passed": bool(row.get("passed")),
                "slat_active_point_count": row.get("slat_active_point_count"),
                "structure": row.get("structure"),
                "target_structure": row.get("target_structure"),
                "mesh": row.get("mesh"),
                "error": row.get("error"),
            }
            branch_rows.append(reduced)
            if not reduced["passed"]:
                failures.append(reduced)
        expected_failure = case["kind"] == "support_explosion"
        case_passed = bool(
            (expected_failure and failures)
            or (not expected_failure and report.get("passed") is True and not failures)
        )
        audit_passed = audit_passed and case_passed
        cases_out.append(
            {
                **{key: case[key] for key in ("label", "kind", "rank", "object_uid", "object_index")},
                "audit_completed": case_passed,
                "model_runtime_passed": bool(report.get("passed")),
                "rerun_report": str(report_path),
                "rerun_report_sha256": _sha256(report_path),
                "gt_target_mesh": str(target_obj),
                "gt_target_mesh_sha256": _sha256(target_obj),
                "ss_records": report["ss_records"],
                "branch_records": branch_rows,
                "failure_count": len(failures),
                "aggregate_metrics": case["aggregate_metrics"],
            }
        )

    output = Path(args.output).expanduser().resolve()
    payload = {
        "format": AUDIT_FORMAT,
        "passed": audit_passed,
        "formal": False,
        "selection": str(selection_path),
        "selection_sha256": _sha256(selection_path),
        "case_count": len(cases_out),
        "cases": cases_out,
        "scope_guard": selection["scope_guard"],
    }
    _write(output, payload)
    summary = output.with_name("summary.txt")
    lines = [
        "Native-SS support explosion + bottom-3 LCR targeted audit",
        "=========================================================",
        f"audit_completed: {audit_passed}",
        f"cases: {len(cases_out)}",
    ]
    for case in cases_out:
        lines.append(
            f"{case['label']}: kind={case['kind']} index={case['object_index']} "
            f"runtime_passed={case['model_runtime_passed']} failures={case['failure_count']}"
        )
        lines.append(f"  gt: {case['gt_target_mesh']}")
        lines.append(f"  rerun: {case['rerun_report']}")
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": audit_passed,
                "report": str(output),
                "summary": str(summary),
                "cases": len(cases_out),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if not audit_passed:
        raise SystemExit(2)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--aggregate_report", required=True)
    select.add_argument("--cache_manifest", required=True)
    select.add_argument("--rerun_root", required=True)
    select.add_argument("--output", required=True)
    select.add_argument("--lcr_count", type=int, default=3)
    select.add_argument("--expected_support_objects", type=int, default=1)
    select.set_defaults(func=select_cases)
    finish = subparsers.add_parser("finalize")
    finish.add_argument("--selection", required=True)
    finish.add_argument("--output", required=True)
    finish.set_defaults(func=finalize)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
