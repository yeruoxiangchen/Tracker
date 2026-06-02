#!/usr/bin/env python3
"""Download Objaverse assets into a /data-backed cache.

The upstream objaverse package derives its cache directory from HOME at import
time.  Keep the import inside main() after HOME is redirected so large assets do
not land in /home/zjr by accident.
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing
import os
from pathlib import Path
import time
from typing import Iterable


def _read_uids(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] in "[{":
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("uids", [])
        return [str(uid).strip() for uid in data if str(uid).strip()]
    return [line.strip() for line in text.splitlines() if line.strip()]


def _write_lines(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def _download_one_worker(task: tuple[str, str, int, int]) -> tuple[bool, str, str]:
    uid, object_path, total_downloads, start_file_count = task
    try:
        import objaverse  # noqa: WPS433

        out_uid, local_path = objaverse._download_object(
            uid,
            object_path,
            total_downloads,
            start_file_count,
        )
        return True, out_uid, local_path
    except BaseException as exc:  # noqa: BLE001 - keep worker errors pickle-safe.
        return False, uid, f"{type(exc).__name__}: {exc}"


def _download_missing_objects(
    objaverse_module,
    selected_uids: list[str],
    download_processes: int,
    batch_size: int,
    retries: int,
    retry_sleep: float,
) -> dict[str, str]:
    object_paths = objaverse_module._load_object_paths()
    versioned_path = Path(objaverse_module._VERSIONED_PATH)
    uid_to_path: dict[str, str] = {}
    missing: list[tuple[str, str]] = []

    for raw_uid in selected_uids:
        uid = raw_uid[:-4] if raw_uid.endswith(".glb") else raw_uid
        object_path = object_paths.get(uid)
        if object_path is None:
            print(f"[Objaverse] warning: UID not found, skip: {uid}", flush=True)
            continue
        local_path = versioned_path / object_path
        if local_path.exists():
            uid_to_path[uid] = str(local_path)
        else:
            missing.append((uid, object_path))

    if not missing:
        return uid_to_path

    total_missing = len(missing)
    print(
        "[Objaverse] missing objects to download: "
        f"{total_missing}; batch_size={batch_size}; retries={retries}",
        flush=True,
    )

    start_file_count = len(glob.glob(str(versioned_path / "glbs" / "*" / "*.glb")))
    batch_size = max(batch_size, 1)
    processes = max(download_processes, 1)

    for batch_start in range(0, total_missing, batch_size):
        batch = missing[batch_start : batch_start + batch_size]
        pending = list(batch)
        for attempt in range(1, retries + 2):
            print(
                "[Objaverse] batch "
                f"{batch_start // batch_size + 1}/"
                f"{(total_missing + batch_size - 1) // batch_size}, "
                f"attempt {attempt}, pending {len(pending)}",
                flush=True,
            )
            tasks = [
                (uid, object_path, total_missing, start_file_count)
                for uid, object_path in pending
            ]
            results: list[tuple[bool, str, str]] = []
            if processes == 1:
                for task in tasks:
                    results.append(_download_one_worker(task))
            else:
                with multiprocessing.Pool(processes=processes) as pool:
                    for result in pool.imap_unordered(_download_one_worker, tasks):
                        results.append(result)

            failed_uids: set[str] = set()
            for ok, uid, payload in results:
                if ok:
                    uid_to_path[uid] = payload
                else:
                    failed_uids.add(uid)
                    print(f"[Objaverse] failed: {uid}: {payload}", flush=True)

            if not failed_uids:
                break
            if attempt > retries:
                print(
                    "[Objaverse] giving up after retries for "
                    f"{len(failed_uids)} objects in this batch",
                    flush=True,
                )
                break

            failed_lookup = {uid: object_path for uid, object_path in pending}
            pending = [(uid, failed_lookup[uid]) for uid in sorted(failed_uids)]
            time.sleep(retry_sleep)

    return uid_to_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Objaverse UIDs/assets with the cache rooted under /data."
    )
    parser.add_argument(
        "--output-root",
        default="/data/Objaverse",
        help="Root directory for Objaverse cache and manifests.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only fetch/write the UID list. Do not download object assets.",
    )
    parser.add_argument(
        "--uids-file",
        default="",
        help="Optional UID list to download. Supports one UID per line or JSON list.",
    )
    parser.add_argument(
        "--save-uids",
        default="",
        help="Where to save the full UID list. Defaults to <output-root>/uids.txt.",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Where to save the downloaded UID->path mapping.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index into the selected UID list.",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=0,
        help="Maximum number of objects to download. 0 means all selected UIDs.",
    )
    parser.add_argument(
        "--download-processes",
        type=int,
        default=4,
        help="Number of parallel download processes passed to objaverse.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Download batch size. Smaller batches recover better from network errors.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Retry count per failed object batch.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=5.0,
        help="Seconds to sleep before retrying failed downloads.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # objaverse computes its cache path from HOME during import.
    os.environ["HOME"] = str(output_root)

    import objaverse  # noqa: WPS433

    full_uids = objaverse.load_uids()
    save_uids = Path(args.save_uids) if args.save_uids else output_root / "uids.txt"
    _write_lines(save_uids, full_uids)
    print(f"[Objaverse] total UIDs: {len(full_uids)}")
    print(f"[Objaverse] wrote UID list: {save_uids}")
    print(f"[Objaverse] cache root: {getattr(objaverse, '_VERSIONED_PATH', 'unknown')}")

    if args.metadata_only:
        return

    if args.uids_file:
        selected_uids = _read_uids(Path(args.uids_file))
    else:
        selected_uids = list(full_uids)

    start = max(args.start_index, 0)
    selected_uids = selected_uids[start:]
    if args.max_objects > 0:
        selected_uids = selected_uids[: args.max_objects]

    if not selected_uids:
        print("[Objaverse] no UIDs selected; nothing to download.")
        return

    print(
        "[Objaverse] downloading "
        f"{len(selected_uids)} objects with {args.download_processes} processes"
    )
    uid_to_path = _download_missing_objects(
        objaverse,
        selected_uids,
        args.download_processes,
        args.batch_size,
        args.retries,
        args.retry_sleep,
    )

    manifest = (
        Path(args.manifest)
        if args.manifest
        else output_root / f"manifest_{start}_{start + len(selected_uids)}.json"
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(uid_to_path, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[Objaverse] downloaded: {len(uid_to_path)}")
    print(f"[Objaverse] wrote manifest: {manifest}")


if __name__ == "__main__":
    main()
