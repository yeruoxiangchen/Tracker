#!/usr/bin/env python3
"""Export the paired Dev8 target, official-O and phone-O Meshes for inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import trimesh

from pose_point_depth_mv.evaluate_native_ss_stock_slat_mesh import write_json
from pose_point_depth_mv.evaluate_official30k_dev8_o_rotation_endpoint import (
    REPORT_FORMAT,
)
from pose_point_depth_mv.proobjaverse_official_slat_protocol import (
    canonical_sha256,
    sha256_file,
)


BUNDLE_FORMAT = "pose_point_depth_mv.official30k_dev8_o_mesh_bundle.v1"


def _load_report(path: Path, expected_arm: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("format") != REPORT_FORMAT or value.get("passed") is not True:
        raise RuntimeError(f"incomplete endpoint report: {path}")
    if value.get("run_identity", {}).get("arm") != expected_arm:
        raise RuntimeError(f"wrong arm in endpoint report: {path}")
    body = dict(value)
    saved = str(body.pop("report_sha256", ""))
    if canonical_sha256(body) != saved:
        raise RuntimeError(f"endpoint report SHA differs: {path}")
    return value


def _rows(report: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    rows = {
        (str(row["object_uid"]), int(row["seed"])): row
        for row in report["records"]
    }
    if len(rows) != len(report["records"]):
        raise RuntimeError("duplicate object/seed rows")
    return rows


def _checked_mesh(row: dict[str, Any], label: str) -> Path:
    path = Path(str(row.get("mesh", ""))).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} Mesh missing: {path}")
    if str(row.get("mesh_sha256", "")) != sha256_file(path):
        raise RuntimeError(f"{label} Mesh SHA differs: {path}")
    return path


def _export_target(source: Path, destination: Path) -> None:
    with np.load(source, allow_pickle=False) as payload:
        mesh = trimesh.Trimesh(
            vertices=np.asarray(payload["vertices"]),
            faces=np.asarray(payload["faces"]),
            process=False,
        )
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"empty target Mesh: {source}")
    mesh.export(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official_report", required=True)
    parser.add_argument("--phone_report", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    official_path = Path(args.official_report).expanduser().resolve()
    phone_path = Path(args.phone_report).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)

    official = _load_report(official_path, "official_o")
    phone = _load_report(phone_path, "phone_o")
    official_rows, phone_rows = _rows(official), _rows(phone)
    if set(official_rows) != set(phone_rows):
        raise RuntimeError("official-O and phone-O object/seed coverage differs")
    if official["run_identity"]["object_uids"] != phone["run_identity"]["object_uids"]:
        raise RuntimeError("official-O and phone-O object ordering differs")

    output.mkdir(parents=True)
    bundle_rows: list[dict[str, Any]] = []
    for ordinal, uid in enumerate(official["run_identity"]["object_uids"], start=1):
        matching = [key for key in official_rows if key[0] == uid]
        if len(matching) != 1:
            raise RuntimeError(f"expected one seed for uid={uid}, got {matching}")
        key = matching[0]
        seed = int(key[1])
        a, b = official_rows[key], phone_rows[key]
        target = Path(str(a["target_mesh"])).expanduser().resolve()
        if not target.is_file() or sha256_file(target) != str(a["target_mesh_sha256"]):
            raise RuntimeError(f"target Mesh binding differs: {target}")
        official_mesh = _checked_mesh(a, "official-O")
        phone_mesh = _checked_mesh(b, "phone-O")

        object_dir = output / f"{ordinal:02d}_{uid[:12]}_seed{seed}"
        object_dir.mkdir()
        target_out = object_dir / "01_GT_官方目标.obj"
        official_out = object_dir / "02_当前模型_官方O.obj"
        phone_out = object_dir / "03_当前模型_手机O_回到官方O.obj"
        _export_target(target, target_out)
        shutil.copy2(official_mesh, official_out)
        shutil.copy2(phone_mesh, phone_out)

        note = {
            "object_uid": uid,
            "seed": seed,
            "coordinate_contract": (
                "三个 Mesh 均位于同一个官方 O 坐标系；手机 O 分支先在手机轴规则下"
                "端到端推理，再用记录的 T_arm_O_to_official_O 变回官方 O。"
            ),
            "files": {
                "ground_truth": str(target_out),
                "official_o_prediction": str(official_out),
                "phone_o_prediction_mapped_to_official_o": str(phone_out),
            },
            "official_o_metrics": {
                "surface": a["surface"],
                "structure": a["structure"],
            },
            "phone_o_metrics": {
                "surface": b["surface"],
                "structure": b["structure"],
            },
            "phone_o_frame": b["frame"],
            "sha256": {
                "ground_truth_obj": sha256_file(target_out),
                "official_o_obj": sha256_file(official_out),
                "phone_o_obj": sha256_file(phone_out),
            },
        }
        write_json(object_dir / "说明.json", note)
        bundle_rows.append(note)

    manifest = {
        "format": BUNDLE_FORMAT,
        "passed": True,
        "formal": False,
        "object_count": len(bundle_rows),
        "official_report": str(official_path),
        "official_report_sha256": sha256_file(official_path),
        "phone_report": str(phone_path),
        "phone_report_sha256": sha256_file(phone_path),
        "objects": bundle_rows,
        "scope_guard": (
            "Dev8 seed42 visual endpoint diagnostic; these exports do not establish "
            "a formal benchmark claim"
        ),
    }
    manifest["report_sha256"] = canonical_sha256(manifest)
    write_json(output / "manifest.json", manifest)
    print(
        json.dumps(
            {"passed": True, "objects": len(bundle_rows), "output": str(output)},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
