#!/usr/bin/env python3
"""Resume an OmniObject3D file list without letting one bad file block the rest.

The OpenXLab folder downloader stops at the first failed file.  This wrapper
invokes the existing single-path downloader in a fresh subprocess for each
missing archive, records progress atomically, and continues after failures.
Existing non-empty files are preserved and skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


OMNI_REPO = "omniobject3d/OmniObject3D-New"
REPO_SAVE_NAME = OMNI_REPO.replace("/", "___")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a frozen OmniObject3D path list one file at a time."
    )
    parser.add_argument("--remote-paths-file", required=True)
    parser.add_argument("--target-path", required=True)
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--dataset-repo", default=OMNI_REPO)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=300.0)
    parser.add_argument("--per-file-sleep", type=float, default=1.0)
    parser.add_argument(
        "--single-file-downloader",
        default=str(Path(__file__).with_name("download_omniobject3d.py")),
    )
    return parser.parse_args()


def load_remote_paths(path: Path) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        remote_path = raw_line.strip()
        if not remote_path or remote_path.startswith("#"):
            continue
        pure = PurePosixPath(remote_path)
        if (
            not remote_path.startswith("/raw/raw_scans/")
            or pure.name in ("", ".", "..")
            or ".." in pure.parts
        ):
            raise ValueError(
                f"invalid Omni raw-scan path at {path}:{line_number}: {remote_path!r}"
            )
        if remote_path in seen:
            raise ValueError(f"duplicate remote path at {path}:{line_number}: {remote_path}")
        seen.add(remote_path)
        rows.append(remote_path)
    if not rows:
        raise ValueError(f"no remote paths found in {path}")
    return rows


def local_path_for(target_path: Path, dataset_repo: str, remote_path: str) -> Path:
    save_name = dataset_repo.replace("/", "___")
    return target_path / save_name / remote_path.lstrip("/")


def is_complete_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def write_state(
    state_file: Path,
    *,
    remote_paths_file: Path,
    target_path: Path,
    dataset_repo: str,
    current_round: int,
    attempts: dict[str, int],
    return_codes: dict[str, int],
    remote_paths: list[str],
) -> dict[str, Any]:
    complete: list[dict[str, Any]] = []
    missing: list[str] = []
    for remote_path in remote_paths:
        local_path = local_path_for(target_path, dataset_repo, remote_path)
        if is_complete_file(local_path):
            complete.append(
                {
                    "remote_path": remote_path,
                    "local_path": str(local_path),
                    "bytes": local_path.stat().st_size,
                }
            )
        else:
            missing.append(remote_path)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "updated_at_utc": utc_now(),
        "dataset_repo": dataset_repo,
        "remote_paths_file": str(remote_paths_file),
        "target_path": str(target_path),
        "round": current_round,
        "expected_count": len(remote_paths),
        "complete_count": len(complete),
        "missing_count": len(missing),
        "passed": not missing,
        "missing_paths": missing,
        "attempts": attempts,
        "last_return_codes": return_codes,
        "complete_files": complete,
    }
    state_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_file.with_name(f".{state_file.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, state_file)
    return payload


def main() -> None:
    args = parse_args()
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")
    if args.retry_sleep < 0 or args.per_file_sleep < 0:
        raise ValueError("sleep intervals must be non-negative")

    remote_paths_file = Path(args.remote_paths_file).expanduser().resolve()
    target_path = Path(args.target_path).expanduser().resolve()
    state_file = Path(args.state_file).expanduser().resolve()
    downloader = Path(args.single_file_downloader).expanduser().resolve()
    if not remote_paths_file.is_file():
        raise FileNotFoundError(remote_paths_file)
    if not downloader.is_file():
        raise FileNotFoundError(downloader)
    target_path.mkdir(parents=True, exist_ok=True)

    remote_paths = load_remote_paths(remote_paths_file)
    attempts: dict[str, int] = {}
    return_codes: dict[str, int] = {}
    initial = write_state(
        state_file,
        remote_paths_file=remote_paths_file,
        target_path=target_path,
        dataset_repo=args.dataset_repo,
        current_round=0,
        attempts=attempts,
        return_codes=return_codes,
        remote_paths=remote_paths,
    )
    print(
        "[omni_filewise] "
        f"expected={initial['expected_count']} complete={initial['complete_count']} "
        f"missing={initial['missing_count']} state={state_file}",
        flush=True,
    )

    for round_index in range(1, args.rounds + 1):
        pending = [
            remote_path
            for remote_path in remote_paths
            if not is_complete_file(
                local_path_for(target_path, args.dataset_repo, remote_path)
            )
        ]
        if not pending:
            break
        print(
            f"[omni_filewise] round={round_index}/{args.rounds} "
            f"pending={len(pending)} start={utc_now()}",
            flush=True,
        )

        for pending_index, remote_path in enumerate(pending, start=1):
            attempts[remote_path] = attempts.get(remote_path, 0) + 1
            print(
                f"[omni_filewise] round={round_index} "
                f"file={pending_index}/{len(pending)} attempt={attempts[remote_path]} "
                f"path={remote_path}",
                flush=True,
            )
            command = [
                sys.executable,
                "-u",
                str(downloader),
                "download",
                "--dataset-repo",
                args.dataset_repo,
                "--source-path",
                remote_path,
                "--target-path",
                str(target_path),
            ]
            completed = subprocess.run(command, check=False)
            return_codes[remote_path] = int(completed.returncode)
            local_path = local_path_for(target_path, args.dataset_repo, remote_path)
            succeeded = completed.returncode == 0 and is_complete_file(local_path)
            print(
                f"[omni_filewise] path={remote_path} rc={completed.returncode} "
                f"present={is_complete_file(local_path)} verdict="
                f"{'complete' if succeeded else 'failed_continue'}",
                flush=True,
            )
            write_state(
                state_file,
                remote_paths_file=remote_paths_file,
                target_path=target_path,
                dataset_repo=args.dataset_repo,
                current_round=round_index,
                attempts=attempts,
                return_codes=return_codes,
                remote_paths=remote_paths,
            )
            if args.per_file_sleep and pending_index < len(pending):
                time.sleep(args.per_file_sleep)

        state = write_state(
            state_file,
            remote_paths_file=remote_paths_file,
            target_path=target_path,
            dataset_repo=args.dataset_repo,
            current_round=round_index,
            attempts=attempts,
            return_codes=return_codes,
            remote_paths=remote_paths,
        )
        print(
            f"[omni_filewise] round={round_index} complete={state['complete_count']} "
            f"missing={state['missing_count']} end={utc_now()}",
            flush=True,
        )
        if state["passed"]:
            break
        if round_index < args.rounds and args.retry_sleep:
            print(
                f"[omni_filewise] retry_sleep={args.retry_sleep:.0f}s",
                flush=True,
            )
            time.sleep(args.retry_sleep)

    final = write_state(
        state_file,
        remote_paths_file=remote_paths_file,
        target_path=target_path,
        dataset_repo=args.dataset_repo,
        current_round=min(args.rounds, max(attempts.values(), default=0)),
        attempts=attempts,
        return_codes=return_codes,
        remote_paths=remote_paths,
    )
    print(
        "[omni_filewise] final "
        f"complete={final['complete_count']}/{final['expected_count']} "
        f"missing={final['missing_count']} passed={final['passed']}",
        flush=True,
    )
    if not final["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
