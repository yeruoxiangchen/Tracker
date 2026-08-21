#!/usr/bin/env python3
"""Freeze candidate-budgeted Objaverse UIDs and audited Omni archive inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


SCHEMA = "tracker.mixed10k_source_inputs.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objaverse_uids_file", required=True)
    parser.add_argument(
        "--objaverse_manifest",
        action="append",
        required=True,
    )
    parser.add_argument("--objaverse_candidate_count", type=int, required=True)
    parser.add_argument("--selection_seed", type=int, default=20260727)
    parser.add_argument("--omni_remote_paths_file", required=True)
    parser.add_argument("--omni_archive_root", required=True)
    parser.add_argument(
        "--omni_exclude_remote_path",
        action="append",
        default=[],
    )
    parser.add_argument("--expected_omni_archives", type=int, required=True)
    parser.add_argument("--omni_exclusion_reason", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_uid(value: object) -> str:
    uid = str(value).strip()
    return uid[:-4] if uid.endswith(".glb") else uid


def read_unique_lines(path: Path, label: str) -> list[str]:
    rows = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != len(set(rows)):
        raise ValueError(f"{label} contains duplicate rows")
    return rows


def write_text(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def stable_candidate_key(seed: int, uid: str) -> str:
    return sha256_text(f"{seed}\0objaverse_candidate\0{uid}")


def load_available_objaverse(
    uid_file: Path,
    manifest_paths: list[Path],
) -> tuple[list[str], list[dict[str, Any]]]:
    requested_uids = [normalize_uid(row) for row in read_unique_lines(uid_file, "UID file")]
    requested_set = set(requested_uids)
    combined: dict[str, Path] = {}
    bindings = []
    for manifest_path in manifest_paths:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "samples" in payload:
            raise ValueError(f"{manifest_path}: expected UID-to-path JSON object")
        for raw_uid, raw_path in payload.items():
            uid = normalize_uid(raw_uid)
            mesh_path = Path(str(raw_path)).expanduser().resolve()
            previous = combined.get(uid)
            if previous is not None and previous != mesh_path:
                raise ValueError(f"conflicting Objaverse paths for UID {uid}")
            combined[uid] = mesh_path
        bindings.append(
            {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
                "entry_count": len(payload),
            }
        )
    available = [
        uid
        for uid in requested_uids
        if uid in combined
        and combined[uid].is_file()
        and combined[uid].stat().st_size > 0
    ]
    unexpected = sorted(set(combined) - requested_set)
    if unexpected:
        raise RuntimeError(
            f"download manifests contain UIDs outside the frozen UID universe: {unexpected[:5]}"
        )
    return available, bindings


def validate_existing(
    output_dir: Path,
    *,
    candidate_count: int,
    selection_seed: int,
    expected_omni_archives: int,
    exclusions: list[str],
) -> dict[str, Any]:
    report_path = output_dir / "input_report.json"
    if not report_path.is_file():
        raise RuntimeError(f"existing source-input directory has no report: {output_dir}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = {
        "schema": SCHEMA,
        "objaverse_candidate_count": candidate_count,
        "selection_seed": selection_seed,
        "omni_archive_count": expected_omni_archives,
        "omni_excluded_remote_paths": exclusions,
    }
    differing = [key for key, value in expected.items() if report.get(key) != value]
    if differing:
        raise RuntimeError(f"existing source-input report differs: {differing}")
    for binding_name in ("objaverse_candidate_uids", "omni_remote_paths"):
        binding = report[binding_name]
        path = Path(binding["path"])
        if not path.is_file() or sha256_file(path) != binding["sha256"]:
            raise RuntimeError(f"frozen output binding changed: {binding_name}")
    if report.get("passed") is not True:
        raise RuntimeError(f"existing source-input report did not pass: {report_path}")
    return report


def main() -> None:
    args = parse_args()
    if args.objaverse_candidate_count <= 0:
        raise ValueError("--objaverse_candidate_count must be positive")
    if args.expected_omni_archives <= 0:
        raise ValueError("--expected_omni_archives must be positive")

    uid_file = Path(args.objaverse_uids_file).expanduser().resolve()
    manifest_paths = [
        Path(path).expanduser().resolve() for path in args.objaverse_manifest
    ]
    omni_remote_file = Path(args.omni_remote_paths_file).expanduser().resolve()
    archive_root = Path(args.omni_archive_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    exclusions = sorted(set(args.omni_exclude_remote_path))
    if len(exclusions) != len(args.omni_exclude_remote_path):
        raise ValueError("duplicate --omni_exclude_remote_path values")

    if output_dir.exists():
        report = validate_existing(
            output_dir,
            candidate_count=args.objaverse_candidate_count,
            selection_seed=args.selection_seed,
            expected_omni_archives=args.expected_omni_archives,
            exclusions=exclusions,
        )
        print(json.dumps({"reused": True, **report}, indent=2, ensure_ascii=False))
        return

    available_uids, manifest_bindings = load_available_objaverse(
        uid_file,
        manifest_paths,
    )
    if len(available_uids) < args.objaverse_candidate_count:
        raise RuntimeError(
            f"only {len(available_uids)} valid Objaverse objects are available, "
            f"below requested {args.objaverse_candidate_count}"
        )
    selected_uids = sorted(
        available_uids,
        key=lambda uid: stable_candidate_key(args.selection_seed, uid),
    )[: args.objaverse_candidate_count]

    original_remote_paths = read_unique_lines(
        omni_remote_file,
        "Omni remote path file",
    )
    missing_exclusions = sorted(set(exclusions) - set(original_remote_paths))
    if missing_exclusions:
        raise RuntimeError(
            f"Omni exclusions are absent from the source list: {missing_exclusions}"
        )
    kept_remote_paths = [
        remote_path
        for remote_path in original_remote_paths
        if remote_path not in set(exclusions)
    ]
    if len(kept_remote_paths) != args.expected_omni_archives:
        raise RuntimeError(
            f"kept {len(kept_remote_paths)} Omni paths, "
            f"expected {args.expected_omni_archives}"
        )
    missing_archives = [
        str(archive_root / Path(remote_path).name)
        for remote_path in kept_remote_paths
        if not (archive_root / Path(remote_path).name).is_file()
        or (archive_root / Path(remote_path).name).stat().st_size <= 0
    ]
    if missing_archives:
        raise RuntimeError(
            f"{len(missing_archives)} kept Omni archives are missing: {missing_archives[:5]}"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging.",
            dir=output_dir.parent,
        )
    )
    try:
        candidate_path = staging / "objaverse_candidate_uids.txt"
        omni_path = staging / "omni_remote_paths.txt"
        write_text(candidate_path, selected_uids)
        write_text(omni_path, kept_remote_paths)
        report = {
            "schema": SCHEMA,
            "passed": True,
            "output_dir": str(output_dir),
            "selection_policy": "sha256(seed,source,uid)",
            "selection_seed": int(args.selection_seed),
            "objaverse_available_count": len(available_uids),
            "objaverse_candidate_count": len(selected_uids),
            "objaverse_uid_universe": {
                "path": str(uid_file),
                "sha256": sha256_file(uid_file),
            },
            "objaverse_download_manifests": manifest_bindings,
            "objaverse_candidate_uids": {
                "path": str(output_dir / candidate_path.name),
                "sha256": sha256_file(candidate_path),
                "entry_count": len(selected_uids),
            },
            "omni_original_remote_paths": {
                "path": str(omni_remote_file),
                "sha256": sha256_file(omni_remote_file),
                "entry_count": len(original_remote_paths),
            },
            "omni_remote_paths": {
                "path": str(output_dir / omni_path.name),
                "sha256": sha256_file(omni_path),
                "entry_count": len(kept_remote_paths),
            },
            "omni_archive_root": str(archive_root),
            "omni_archive_count": len(kept_remote_paths),
            "omni_excluded_remote_paths": exclusions,
            "omni_exclusion_reason": args.omni_exclusion_reason,
            "hard_guards": {
                "candidate_budget_exact": len(selected_uids)
                == args.objaverse_candidate_count,
                "all_kept_omni_archives_present": not missing_archives,
                "exclusions_were_in_original_list": not missing_exclusions,
            },
        }
        if not all(report["hard_guards"].values()):
            raise RuntimeError(f"source-input hard guard failed: {report['hard_guards']}")
        write_json(staging / "input_report.json", report)
        os.replace(staging, output_dir)
    except BaseException:
        raise

    print(json.dumps({"reused": False, **report}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
