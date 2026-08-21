#!/usr/bin/env python3
"""Download an exact, frozen paired subset of Stable-X/ProObjaverse-300K.

ReconViaGen needs both files for every object::

    renders_random_env/shard-XXXX/{uid}.tar
    lh-slats/shard-XXXX/{uid}.npz

The upstream repository contains occasional render-only objects.  Selecting
whole shards therefore does not guarantee an exact training-set size.  This
tool first freezes the render/latent UID intersection and then downloads only
the selected pairs.  Re-running the same output directory resumes that frozen
selection instead of silently choosing different objects.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError


FORMAT = "reconviagen.proobjaverse_paired_subset.v1"
REPORT_FORMAT = "reconviagen.proobjaverse_paired_download.v1"
DEFAULT_REPO = "Stable-X/ProObjaverse-300K"
# Pin the repository observed on 2026-08-13.  A moving `main` branch would make
# an ostensibly identical 5K experiment non-reproducible.
DEFAULT_REVISION = "c9175c52d4a45feb9536bec2f1168c50ecdc7765"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _utc_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _list_files(
    api: HfApi,
    *,
    repo_id: str,
    revision: str,
    folder: str,
    suffix: str,
    token: str | None,
) -> dict[str, dict[str, Any]]:
    try:
        rows = api.list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=folder,
            recursive=True,
            revision=revision,
            token=token,
        )
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            path = str(row.path)
            if not path.endswith(suffix):
                continue
            uid = Path(path).stem
            if uid in result:
                raise RuntimeError(f"duplicate UID in {folder}: {uid}")
            result[uid] = {
                "path": path,
                "size": int(getattr(row, "size", 0) or 0),
            }
        return result
    except EntryNotFoundError:
        return {}


def _load_exclusions(
    args: argparse.Namespace,
) -> tuple[set[str], list[dict[str, Any]]]:
    excluded: set[str] = set()
    bindings: list[dict[str, Any]] = []
    for raw_path in args.exclude_selection:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"excluded selection not found: {path}")
        selection = json.loads(path.read_text(encoding="utf-8"))
        body = {
            key: value
            for key, value in selection.items()
            if key not in {"created_at_utc", "selection_sha256"}
        }
        if selection.get("format") != FORMAT:
            raise RuntimeError(f"unsupported excluded selection format: {path}")
        if selection.get("selection_sha256") != _canonical_sha256(body):
            raise RuntimeError(f"excluded selection hash is invalid: {path}")
        if (
            selection.get("repo_id") != args.repo_id
            or selection.get("revision") != args.revision
        ):
            raise RuntimeError(
                f"excluded selection repository/revision differs: {path}"
            )
        selected_uids = [str(row["uid"]) for row in selection.get("selected", [])]
        if (
            len(selected_uids) != int(selection.get("selected_pair_count", -1))
            or len(selected_uids) != len(set(selected_uids))
        ):
            raise RuntimeError(f"excluded selection UID accounting is invalid: {path}")
        overlap = excluded & set(selected_uids)
        if overlap:
            raise RuntimeError(
                f"excluded selections overlap by {len(overlap)} UIDs; first={min(overlap)}"
            )
        excluded.update(selected_uids)
        bindings.append(
            {
                "path": str(path),
                "file_sha256": _sha256_file(path),
                "selection_sha256": str(selection["selection_sha256"]),
                "selected_pair_count": len(selected_uids),
                "selected_uid_sha256": _canonical_sha256(sorted(selected_uids)),
            }
        )
    return excluded, bindings


def _plan_selection(
    args: argparse.Namespace,
    *,
    excluded_uids: set[str],
    exclusion_bindings: list[dict[str, Any]],
) -> dict[str, Any]:
    api = HfApi()
    candidates: dict[str, dict[str, Any]] = {}
    shard_reports = []
    required_pool = int(args.count) + int(args.candidate_margin)
    for shard_index in range(args.shard_start, args.shard_stop + 1):
        shard = f"shard-{shard_index:04d}"
        renders = _list_files(
            api,
            repo_id=args.repo_id,
            revision=args.revision,
            folder=f"renders_random_env/{shard}",
            suffix=".tar",
            token=args.token,
        )
        slats = _list_files(
            api,
            repo_id=args.repo_id,
            revision=args.revision,
            folder=f"lh-slats/{shard}",
            suffix=".npz",
            token=args.token,
        )
        paired = sorted(set(renders) & set(slats))
        eligible = [uid for uid in paired if uid not in excluded_uids]
        for uid in eligible:
            if uid in candidates:
                raise RuntimeError(f"UID appears in more than one shard: {uid}")
            candidates[uid] = {
                "uid": uid,
                "shard": shard,
                "render": renders[uid],
                "slat": slats[uid],
            }
        shard_reports.append(
            {
                "shard": shard,
                "render_count": len(renders),
                "slat_count": len(slats),
                "paired_count": len(paired),
                "excluded_pair_count": len(paired) - len(eligible),
                "eligible_pair_count": len(eligible),
            }
        )
        print(
            f"[plan] {shard}: renders={len(renders)} slats={len(slats)} "
            f"paired={len(paired)} excluded={len(paired) - len(eligible)} "
            f"eligible={len(eligible)} cumulative={len(candidates)}",
            flush=True,
        )
        if len(candidates) >= required_pool:
            break

    if len(candidates) < args.count:
        raise RuntimeError(
            f"only {len(candidates)} paired objects found in scanned shards; "
            f"need {args.count}. Increase --shard-stop."
        )
    ordered = [candidates[uid] for uid in sorted(candidates)]
    selected = random.Random(args.seed).sample(ordered, args.count)
    selected.sort(key=lambda row: (row["shard"], row["uid"]))
    expected_bytes = sum(
        int(row["render"]["size"]) + int(row["slat"]["size"])
        for row in selected
    )
    identity = {
        "format": FORMAT,
        "repo_id": args.repo_id,
        "repo_type": "dataset",
        "revision": args.revision,
        "selection_policy": (
            "seeded sample from the paired UID intersection of sequentially scanned shards, "
            "after removing every UID bound by excluded selections"
            if exclusion_bindings
            else "seeded sample from the paired UID intersection of sequentially scanned shards"
        ),
        "seed": int(args.seed),
        "requested_count": int(args.count),
        "candidate_margin": int(args.candidate_margin),
        "shard_start": int(args.shard_start),
        "shard_stop_limit": int(args.shard_stop),
        "scanned_shards": shard_reports,
        "candidate_pair_count": len(candidates),
        "selected_pair_count": len(selected),
        "expected_download_bytes": expected_bytes,
        "selected": selected,
    }
    if exclusion_bindings:
        identity.update(
            {
                "excluded_selection_bindings": exclusion_bindings,
                "excluded_uid_count": len(excluded_uids),
                "excluded_uid_sha256": _canonical_sha256(sorted(excluded_uids)),
            }
        )
    return {
        "created_at_utc": _utc_now(),
        **identity,
        "selection_sha256": _canonical_sha256(identity),
    }


def _load_or_create_selection(
    args: argparse.Namespace,
    path: Path,
    *,
    excluded_uids: set[str],
    exclusion_bindings: list[dict[str, Any]],
) -> dict[str, Any]:
    if path.is_file():
        selection = json.loads(path.read_text(encoding="utf-8"))
        body = {
            key: value
            for key, value in selection.items()
            if key not in {"created_at_utc", "selection_sha256"}
        }
        if selection.get("format") != FORMAT:
            raise RuntimeError(f"unsupported frozen selection format: {path}")
        if selection.get("selection_sha256") != _canonical_sha256(body):
            raise RuntimeError(f"frozen selection hash is invalid: {path}")
        requested = {
            "repo_id": args.repo_id,
            "revision": args.revision,
            "seed": int(args.seed),
            "requested_count": int(args.count),
            "candidate_margin": int(args.candidate_margin),
            "shard_start": int(args.shard_start),
            "shard_stop_limit": int(args.shard_stop),
        }
        if exclusion_bindings:
            requested.update(
                {
                    "excluded_selection_bindings": exclusion_bindings,
                    "excluded_uid_count": len(excluded_uids),
                    "excluded_uid_sha256": _canonical_sha256(sorted(excluded_uids)),
                }
            )
        elif int(selection.get("excluded_uid_count", 0)) != 0:
            requested["excluded_uid_count"] = 0
        mismatches = {
            key: {"frozen": selection.get(key), "requested": value}
            for key, value in requested.items()
            if selection.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"output directory contains a different frozen selection: {mismatches}. "
                "Use a new --output-root."
            )
        print(f"[plan] reuse frozen selection: {path}", flush=True)
        return selection

    selection = _plan_selection(
        args,
        excluded_uids=excluded_uids,
        exclusion_bindings=exclusion_bindings,
    )
    _atomic_json(path, selection)
    print(f"[plan] frozen selection: {path}", flush=True)
    return selection


def _local_path(root: Path, remote_path: str) -> Path:
    relative = PurePosixPath(remote_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"unsafe repository path: {remote_path!r}")
    return root.joinpath(*relative.parts)


def _validate_local_file(root: Path, file_row: dict[str, Any]) -> bool:
    path = _local_path(root, str(file_row["path"]))
    return path.is_file() and path.stat().st_size == int(file_row["size"])


def _download_file(
    *,
    args: argparse.Namespace,
    output_root: Path,
    file_row: dict[str, Any],
) -> Path:
    expected_path = _local_path(output_root, str(file_row["path"]))
    if _validate_local_file(output_root, file_row):
        return expected_path
    last_error: Exception | None = None
    for attempt in range(1, args.retries + 1):
        try:
            downloaded = Path(
                hf_hub_download(
                    repo_id=args.repo_id,
                    filename=str(file_row["path"]),
                    repo_type="dataset",
                    revision=args.revision,
                    local_dir=str(output_root),
                    token=args.token,
                )
            )
            if downloaded.resolve() != expected_path.resolve():
                raise RuntimeError(
                    f"unexpected local path for {file_row['path']}: {downloaded}"
                )
            if not _validate_local_file(output_root, file_row):
                raise RuntimeError(
                    f"downloaded size mismatch for {file_row['path']}: "
                    f"expected={file_row['size']} actual={downloaded.stat().st_size}"
                )
            return downloaded
        except Exception as error:  # preserve the final Hub/Xet error verbatim
            last_error = error
            if attempt < args.retries:
                time.sleep(min(2 ** (attempt - 1), 16))
    raise RuntimeError(
        f"failed after {args.retries} attempts: {file_row['path']}: {last_error}"
    ) from last_error


def _download_pair(
    args: argparse.Namespace,
    output_root: Path,
    row: dict[str, Any],
) -> dict[str, Any]:
    render = _download_file(args=args, output_root=output_root, file_row=row["render"])
    slat = _download_file(args=args, output_root=output_root, file_row=row["slat"])
    return {
        "uid": row["uid"],
        "render": str(render),
        "slat": str(slat),
        "bytes": int(row["render"]["size"]) + int(row["slat"]["size"]),
    }


def _progress_report(
    *,
    path: Path,
    selection: dict[str, Any],
    completed: int,
    failures: list[dict[str, str]],
    complete: bool,
) -> None:
    report = {
        "format": REPORT_FORMAT,
        "updated_at_utc": _utc_now(),
        "complete": bool(complete),
        "selection": str(path.parent / "selection.json"),
        "selection_sha256": selection["selection_sha256"],
        "selected_pair_count": int(selection["selected_pair_count"]),
        "completed_pair_count": int(completed),
        "failed_pair_count": len(failures),
        "failures": failures,
    }
    body = dict(report)
    report["report_sha256"] = _canonical_sha256(body)
    _atomic_json(path, report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download an exact paired subset of Stable-X/ProObjaverse-300K."
    )
    parser.add_argument(
        "--output-root",
        default="/data/zjr/ProObjaverse-300K-ReconViaGen-5K",
    )
    parser.add_argument(
        "--state-root",
        help=(
            "Directory for selection.json and download_report.json. Defaults to "
            "output-root; use a separate state root when extending a shared data root."
        ),
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--shard-start", type=int, default=1)
    parser.add_argument("--shard-stop", type=int, default=295)
    parser.add_argument(
        "--candidate-margin",
        type=int,
        default=500,
        help="Scan until at least count+margin paired UIDs exist before seeded selection.",
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--min-free-extra-gib", type=float, default=5.0)
    parser.add_argument("--token", help="Optional Hugging Face token; HF_TOKEN also works.")
    parser.add_argument(
        "--exclude-selection",
        action="append",
        default=[],
        help=(
            "Frozen selection.json whose UIDs must be excluded. May be repeated; "
            "repository, revision, hashes and mutual disjointness are verified."
        ),
    )
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0 or args.candidate_margin < 0:
        raise ValueError("count must be positive and candidate-margin must be non-negative")
    if args.shard_start < 0 or args.shard_stop < args.shard_start:
        raise ValueError("invalid shard range")
    if args.max_workers <= 0 or args.max_workers > 32:
        raise ValueError("max-workers must be in [1,32]")
    if args.retries <= 0:
        raise ValueError("retries must be positive")

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    state_root = (
        output_root
        if args.state_root is None
        else Path(args.state_root).expanduser().resolve()
    )
    state_root.mkdir(parents=True, exist_ok=True)
    selection_path = state_root / "selection.json"
    report_path = state_root / "download_report.json"
    excluded_uids, exclusion_bindings = _load_exclusions(args)
    selection = _load_or_create_selection(
        args,
        selection_path,
        excluded_uids=excluded_uids,
        exclusion_bindings=exclusion_bindings,
    )
    expected_gib = float(selection["expected_download_bytes"]) / (1024**3)
    print(
        f"[plan] selected={selection['selected_pair_count']} "
        f"expected={expected_gib:.2f} GiB revision={selection['revision']}",
        flush=True,
    )
    if args.plan_only:
        return

    remaining_bytes = 0
    already_complete = 0
    for row in selection["selected"]:
        render_ok = _validate_local_file(output_root, row["render"])
        slat_ok = _validate_local_file(output_root, row["slat"])
        if render_ok and slat_ok:
            already_complete += 1
        else:
            if not render_ok:
                remaining_bytes += int(row["render"]["size"])
            if not slat_ok:
                remaining_bytes += int(row["slat"]["size"])
    free_bytes = shutil.disk_usage(output_root).free
    required_bytes = remaining_bytes + int(args.min_free_extra_gib * 1024**3)
    print(
        f"[preflight] already_complete={already_complete}/{args.count} "
        f"remaining={remaining_bytes / (1024**3):.2f} GiB "
        f"free={free_bytes / (1024**3):.2f} GiB",
        flush=True,
    )
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"insufficient disk space under {output_root}: free={free_bytes / (1024**3):.2f} GiB "
            f"required={required_bytes / (1024**3):.2f} GiB"
        )

    failures: list[dict[str, str]] = []
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(_download_pair, args, output_root, row): row
            for row in selection["selected"]
        }
        for future in concurrent.futures.as_completed(futures):
            row = futures[future]
            try:
                future.result()
                completed += 1
            except Exception as error:
                failures.append({"uid": str(row["uid"]), "error": repr(error)})
                print(f"[download:error] uid={row['uid']} {error}", file=sys.stderr, flush=True)
            finished = completed + len(failures)
            if finished % 100 == 0 or finished == len(futures):
                print(
                    f"[download] processed={finished}/{len(futures)} "
                    f"complete={completed} failed={len(failures)}",
                    flush=True,
                )
                _progress_report(
                    path=report_path,
                    selection=selection,
                    completed=completed,
                    failures=failures,
                    complete=False,
                )

    # Re-scan the selected files, so a resumed run reports all pairs rather
    # than only futures that happened to succeed during this invocation.
    verified = sum(
        _validate_local_file(output_root, row["render"])
        and _validate_local_file(output_root, row["slat"])
        for row in selection["selected"]
    )
    complete = verified == int(selection["selected_pair_count"]) and not failures
    _progress_report(
        path=report_path,
        selection=selection,
        completed=verified,
        failures=failures,
        complete=complete,
    )
    if not complete:
        raise RuntimeError(
            f"paired subset incomplete: verified={verified}/{selection['selected_pair_count']} "
            f"failures={len(failures)}; rerun the same command to resume"
        )
    print(f"[complete] paired objects={verified}", flush=True)
    print(f"dataset root: {output_root}", flush=True)
    print(f"state root: {state_root}", flush=True)
    print(f"selection: {selection_path}", flush=True)
    print(f"report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
