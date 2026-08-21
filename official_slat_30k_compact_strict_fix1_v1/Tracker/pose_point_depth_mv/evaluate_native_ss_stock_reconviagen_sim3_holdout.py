#!/usr/bin/env python3
"""Eight-way-shard proper-Sim(3) geometry evaluation for frozen mesh outputs.

The evaluator deliberately consumes *mesh manifests*, rather than running a
model.  This keeps the geometry comparison auditable and lets Omni/Objaverse
use the same protocol.  Every prediction is scored twice: raw canonical and
GT-assisted proper Sim(3) (SO(3) + translation + isotropic scale only).

Usage:
  python -m pose_point_depth_mv.evaluate_native_ss_stock_reconviagen_sim3_holdout \
    prepare --gt ... --native ... --recon ... --output_dir ...
  CUDA_VISIBLE_DEVICES=0 python -m ... worker --run_dir ... --worker_id 0 --worker_count 8
  python -m ... merge --run_dir ...

The worker accepts CUDA_VISIBLE_DEVICES for deployment compatibility, but the
mesh alignment/metric kernels are CPU based (trimesh/scipy).  Eight workers
can therefore be launched on hosts with or without CUDA.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from pose_point_depth_mv.compare_pixal3d_singleview_smoke import (
    canonical_sha256,
    load_mesh,
    sha256_file,
    similarity_icp,
    surface_metrics,
)

FORMAT = "pose_point_depth_mv.native_ss_stock_reconviagen_sim3_holdout.v1"
RECORD_FORMAT = FORMAT + ".record"
METHODS = ("native_ss_stock", "reconviagen_original")
TRACKS = ("raw_canonical", "sim3_shape_only")
METRICS = ("chamfer_l1", "chamfer_l2", "fscore_0p01", "fscore_0p02", "fscore_0p05", "normal_consistency")
LOWER = {"chamfer_l1", "chamfer_l2"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def bind(path: str | Path, expected: str = "") -> dict[str, str]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    digest = sha256_file(p)
    if expected and digest != expected:
        raise RuntimeError(f"sha256 mismatch: {p}")
    return {"path": str(p), "sha256": digest}


def rows_from(payload: Any, label: str) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        rows = payload.get("objects", payload.get("records", payload.get("samples")))
    else:
        rows = payload
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{label}: expected a non-empty objects/records list")
    return rows


def row_uid(row: dict[str, Any]) -> str:
    return str(row.get("uid") or row.get("object_key") or row.get("object_uid") or row.get("object_id") or "")


def mesh_binding(row: dict[str, Any], label: str) -> dict[str, str]:
    value = row.get("mesh") or row.get("mesh_o") or row.get("mesh_path") or row.get("path")
    expected = row.get("mesh_sha256") or row.get("mesh_o_sha256") or row.get("sha256") or ""
    if isinstance(value, dict):
        expected = value.get("sha256", expected)
        value = value.get("path")
    if not value:
        raise RuntimeError(f"{label}: row has no mesh path")
    return bind(str(value), str(expected))


def mesh_reference(row: dict[str, Any], label: str) -> dict[str, str]:
    """Freeze a path/hash pair without reading the large mesh during prepare."""
    value = row.get("mesh") or row.get("mesh_o") or row.get("mesh_path") or row.get("path")
    expected = row.get("mesh_sha256") or row.get("mesh_o_sha256") or row.get("sha256") or ""
    if isinstance(value, dict):
        expected = value.get("sha256", expected)
        value = value.get("path")
    if not value:
        raise RuntimeError(f"{label}: row has no mesh path")
    path = Path(str(value)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if not expected:
        raise RuntimeError(f"{label}: missing frozen mesh sha256")
    return {"path": str(path), "sha256": str(expected)}


def make_cases(gt_path: Path, native_path: Path, recon_path: Path, max_objects: int = 0) -> list[dict[str, Any]]:
    gt_list = rows_from(load_json(gt_path), "gt")
    native_list = rows_from(load_json(native_path), "native")
    recon_list = rows_from(load_json(recon_path), "recon")
    gt_rows = {row_uid(r): r for r in gt_list}
    native_rows = {row_uid(r): r for r in native_list}
    recon_rows = {row_uid(r): r for r in recon_list}
    if len(gt_rows) != len(gt_list) or len(native_rows) != len(native_list) or len(recon_rows) != len(recon_list):
        raise RuntimeError("gt contains duplicate UIDs")
    common = sorted(set(gt_rows) & set(native_rows) & set(recon_rows))     
    if not common:
        raise RuntimeError("no common UIDs across the three manifests")
    if int(max_objects) > 0:
        common = common[: int(max_objects)]
    cases = []
    for uid in common:
        g, n, r = gt_rows[uid], native_rows[uid], recon_rows[uid]
        category = str(g.get("category") or n.get("category") or r.get("category") or "unknown")
        cases.append({
            "uid": uid,
            "object_uid": str(g.get("object_uid") or n.get("object_uid") or r.get("object_uid") or uid),
            "category": category,
            "target": mesh_reference(g, f"gt/{uid}"),
            "methods": {
                "native_ss_stock": mesh_reference(n, f"native/{uid}"),
                "reconviagen_original": mesh_reference(r, f"recon/{uid}"),
            },
        })
    return cases


def audit_sim3(alignment: dict[str, Any]) -> dict[str, Any]:
    matrix = np.asarray(alignment.get("matrix"), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise RuntimeError("invalid alignment matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8):
        raise RuntimeError("invalid homogeneous row")
    linear = matrix[:3, :3]
    singular = np.linalg.svd(linear, compute_uv=False)
    if np.any(singular <= 0):
        raise RuntimeError("non-positive similarity scale")
    ratio = float(singular.max() / singular.min())
    scale = float(np.cbrt(np.linalg.det(linear)))
    rotation = linear / scale
    det = float(np.linalg.det(rotation))
    orth = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    if ratio > 1.00001 or det <= 0 or orth > 1e-5:
        raise RuntimeError(f"alignment is not proper isotropic Sim(3): ratio={ratio}, det={det}, orth={orth}")
    return {**alignment, "proper_sim3_validated": True, "reflection": False,
            "anisotropic_scale": False, "isotropic_scale": scale,
            "anisotropy_ratio": ratio, "rotation_determinant": det,
            "rotation_orthogonality_max_abs": orth}


def summarize(values: Iterable[float], bootstrap: int, seed: int) -> dict[str, Any]:
    a = np.asarray(list(values), dtype=np.float64)
    if len(a) == 0 or not np.isfinite(a).all():
        raise RuntimeError("non-finite or empty summary")
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), size=(bootstrap, len(a)))].mean(axis=1)
    return {"count": int(len(a)), "mean": float(a.mean()), "median": float(np.median(a)),
            "bootstrap_mean_95_ci": [float(np.quantile(means, .025)), float(np.quantile(means, .975))],
            "min": float(a.min()), "max": float(a.max())}


def paired_summary(records: list[dict[str, Any]], track: str, bootstrap: int, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {"methods": {}, "native_vs_recon": {"candidate": "native_ss_stock", "baseline": "reconviagen_original", "metrics": {}}}
    for method in METHODS:
        out["methods"][method] = {m: summarize([r["methods"][method][track][m] for r in records], bootstrap, seed + i) for i, m in enumerate(METRICS)}
    for i, metric in enumerate(METRICS):
        deltas = []
        for r in records:
            c = float(r["methods"]["native_ss_stock"][track][metric]); b = float(r["methods"]["reconviagen_original"][track][metric])
            deltas.append((b - c) if metric in LOWER else (c - b))
        s = summarize(deltas, bootstrap, seed + 1000 + i)
        s.update({"candidate_win_count": int(sum(x > 0 for x in deltas)), "tie_count": int(sum(x == 0 for x in deltas)), "baseline_win_count": int(sum(x < 0 for x in deltas))})
        out["native_vs_recon"]["metrics"][metric] = s
    return out


def cmd_prepare(args: argparse.Namespace) -> None:
    out = Path(args.output_dir).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    cases = make_cases(Path(args.gt), Path(args.native), Path(args.recon), int(args.max_objects))
    config = {"format": FORMAT, "formal": bool(args.formal), "post_hoc": bool(args.post_hoc),
              "holdout_consumed": bool(args.holdout_consumed), "dataset": args.dataset,
              "gt_manifest": bind(args.gt), "native_manifest": bind(args.native), "recon_manifest": bind(args.recon),
              "worker_count": int(args.worker_count), "cases": cases,
              "alignment": {"policy": "GT-assisted proper Sim(3)", "proper_rotations": 24, "reflection": False, "anisotropic_scale": False,
                            "candidate_samples": args.candidate_samples, "alignment_samples": args.alignment_samples,
                            "candidate_iterations": args.candidate_iterations, "final_iterations": args.final_iterations},
              "evaluation": {"surface_samples": args.surface_samples, "bootstrap_samples": args.bootstrap_samples, "metric_seed": args.metric_seed}}
    config["config_sha256"] = canonical_sha256(config)
    atomic_json(out / "run_config.json", config)
    print(json.dumps({"passed": True, "object_count": len(cases), "run_config": str(out / "run_config.json")}))


def cmd_worker(args: argparse.Namespace) -> None:
    run = Path(args.run_dir).expanduser().resolve(); cfg = load_json(run / "run_config.json")
    cases = cfg["cases"]; wid, n = int(args.worker_id), int(args.worker_count)
    if n != int(cfg["worker_count"]) or not 0 <= wid < n: raise RuntimeError("worker binding differs from run config")
    selected = cases[ len(cases)*wid//n : len(cases)*(wid+1)//n ]
    out = run / f"worker_{wid:02d}"; out.mkdir(parents=True, exist_ok=True)
    for pos, case in enumerate(selected):
        rp = out / "records" / f"{pos:04d}_{hashlib.sha1(case['uid'].encode()).hexdigest()[:12]}.json"
        if rp.is_file() and args.resume: continue
        target_binding = bind(case["target"]["path"], case["target"]["sha256"])
        target = load_mesh(target_binding["path"]); rows = {}
        for mi, method in enumerate(METHODS):
            method_binding = bind(case["methods"][method]["path"], case["methods"][method]["sha256"])
            mesh = load_mesh(method_binding["path"])
            seed = int(cfg["evaluation"]["metric_seed"]) + cases.index(case) * 100003
            raw = surface_metrics(mesh, target, count=int(cfg["evaluation"]["surface_samples"]), seed=seed, thresholds=(.01,.02,.05))
            aligned, alignment = similarity_icp(mesh, target, seed=seed, candidate_samples=int(cfg["alignment"]["candidate_samples"]), final_samples=int(cfg["alignment"]["alignment_samples"]), candidate_iterations=int(cfg["alignment"]["candidate_iterations"]), final_iterations=int(cfg["alignment"]["final_iterations"]))
            shape = surface_metrics(aligned, target, count=int(cfg["evaluation"]["surface_samples"]), seed=seed, thresholds=(.01,.02,.05))
            rows[method] = {"mesh": method_binding, "raw_canonical": {k: float(raw[k]) for k in METRICS}, "alignment": audit_sim3(alignment), "sim3_shape_only": {k: float(shape[k]) for k in METRICS}}
        atomic_json(rp, {"format": RECORD_FORMAT, "config_sha256": cfg["config_sha256"], "uid": case["uid"], "object_uid": case["object_uid"], "category": case["category"], "target": target_binding, "methods": rows, "passed": True})
        print(f"[worker {wid}] {pos+1}/{len(selected)} {case['uid']}", flush=True)
    atomic_json(out / "report.json", {"format": FORMAT + ".worker", "passed": True, "worker_id": wid, "worker_count": n, "record_count": len(selected), "config_sha256": cfg["config_sha256"]})


def cmd_merge(args: argparse.Namespace) -> None:
    run = Path(args.run_dir).expanduser().resolve(); cfg = load_json(run / "run_config.json"); all_rows = []
    seen = set()
    for wid in range(int(cfg["worker_count"])):
        files = sorted((run / f"worker_{wid:02d}" / "records").glob("*.json"))
        expected = cfg["cases"][len(cfg["cases"])*wid//cfg["worker_count"] : len(cfg["cases"])*(wid+1)//cfg["worker_count"]]
        if len(files) != len(expected): raise RuntimeError(f"worker {wid}: expected {len(expected)} records, found {len(files)}")
        for p in files:
            row = load_json(p)
            if row.get("config_sha256") != cfg["config_sha256"] or row.get("passed") is not True: raise RuntimeError(f"invalid record {p}")
            if row["uid"] in seen: raise RuntimeError(f"duplicate uid {row['uid']}")
            seen.add(row["uid"]); all_rows.append(row)
    expected_uids = {c["uid"] for c in cfg["cases"]}
    if seen != expected_uids: raise RuntimeError(f"missing/extra UIDs: missing={expected_uids-seen} extra={seen-expected_uids}")
    all_rows.sort(key=lambda x: x["uid"])
    b = int(cfg["evaluation"]["bootstrap_samples"]); s = int(cfg["evaluation"]["metric_seed"])
    report = {"format": FORMAT, "created_at_utc": now(), "passed": True, "formal": cfg["formal"], "post_hoc": cfg["post_hoc"], "holdout_consumed": cfg["holdout_consumed"], "dataset": cfg["dataset"], "object_count": len(all_rows), "config": str(run / "run_config.json"), "config_sha256": cfg["config_sha256"], "summary": {t: paired_summary(all_rows, t, b, s + i*10000) for i,t in enumerate(TRACKS)}, "records": all_rows}
    atomic_json(run / "report.json", report)
    print(json.dumps({"passed": True, "report": str(run / "report.json"), "object_count": len(all_rows), "shape": report["summary"]["sim3_shape_only"]["native_vs_recon"]["metrics"]}, indent=2))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare"); q.add_argument("--gt", required=True); q.add_argument("--native", required=True); q.add_argument("--recon", required=True); q.add_argument("--output_dir", required=True); q.add_argument("--dataset", required=True); q.add_argument("--worker_count", type=int, default=8); q.add_argument("--max_objects", type=int, default=0); q.add_argument("--formal", action="store_true"); q.add_argument("--post_hoc", action="store_true"); q.add_argument("--holdout_consumed", action="store_true"); q.add_argument("--candidate_samples", type=int, default=2000); q.add_argument("--alignment_samples", type=int, default=10000); q.add_argument("--candidate_iterations", type=int, default=12); q.add_argument("--final_iterations", type=int, default=50); q.add_argument("--surface_samples", type=int, default=20000); q.add_argument("--bootstrap_samples", type=int, default=10000); q.add_argument("--metric_seed", type=int, default=20260812); q.set_defaults(func=cmd_prepare)
    q = sub.add_parser("worker"); q.add_argument("--run_dir", required=True); q.add_argument("--worker_id", type=int, required=True); q.add_argument("--worker_count", type=int, default=8); q.add_argument("--resume", action="store_true"); q.set_defaults(func=cmd_worker)
    q = sub.add_parser("merge"); q.add_argument("--run_dir", required=True); q.set_defaults(func=cmd_merge)
    return p


if __name__ == "__main__":
    parsed = parser().parse_args()
    parsed.func(parsed)
