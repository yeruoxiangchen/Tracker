#!/usr/bin/env python3
"""Report live per-worker and per-shard progress for strict mixed rendering."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

try:
    from .monitor_mixed_render_smoke import (
        gpu_state,
        latest_progress,
        read_json,
        service_state,
    )
except ImportError:
    from monitor_mixed_render_smoke import (  # type: ignore
        gpu_state,
        latest_progress,
        read_json,
        service_state,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render_root", type=Path, required=True)
    parser.add_argument("--log_root", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--expected_shards", type=int, required=True)
    parser.add_argument("--expected_tasks_per_shard", type=int, required=True)
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument("--unit", action="append", default=[])
    parser.add_argument("--gpu_index", action="append", type=int, default=[])
    return parser.parse_args()


def manifest_counts(manifest: dict[str, Any] | None) -> tuple[int | None, int | None]:
    if manifest is None:
        return None, None
    samples = manifest.get("samples")
    failures = manifest.get("failures")
    accepted = len(samples) if isinstance(samples, list) else None
    failed = len(failures) if isinstance(failures, list) else None
    return accepted, failed


def shard_state(
    render_root: Path,
    log_root: Path,
    source: str,
    shard_index: int,
    expected_tasks: int,
) -> dict[str, Any]:
    shard_name = f"shard_{shard_index:03d}"
    shard_dir = render_root / source / shard_name
    log_path = log_root / source / f"{shard_name}.log"
    marker = (shard_dir / "_WORKER_COMPLETE.json").is_file()
    manifest = read_json(shard_dir / "manifest.json")
    accepted, failed = manifest_counts(manifest)
    progress = latest_progress(log_path)

    if accepted is not None and failed is not None:
        done = accepted + failed
    elif progress is not None and isinstance(progress.get("done"), int):
        done = int(progress["done"])
        accepted = progress.get("accepted")
        failed = progress.get("failed")
    else:
        done = 0

    total = expected_tasks
    if progress is not None and isinstance(progress.get("total"), int):
        total = int(progress["total"])
    total = max(total, done)

    if marker:
        status = "完成"
    elif manifest is not None:
        status = "manifest已生成、等待或未通过marker硬门"
    elif progress is not None and done > 0:
        status = "运行中"
    elif progress is not None:
        status = "首个sequence处理中"
    elif log_path.is_file():
        status = "初始化或首个sequence处理中"
    else:
        status = "等待"

    log_age_seconds = None
    try:
        log_age_seconds = max(
            0,
            int(dt.datetime.now().timestamp() - log_path.stat().st_mtime),
        )
    except OSError:
        pass

    return {
        "index": shard_index,
        "name": shard_name,
        "status": status,
        "done": done,
        "total": total,
        "accepted": accepted,
        "failed": failed,
        "elapsed": progress.get("elapsed") if progress else None,
        "eta": progress.get("eta") if progress else None,
        "last": progress.get("last") if progress else None,
        "marker": marker,
        "has_log": log_path.is_file(),
        "log_age_seconds": log_age_seconds,
    }


def current_worker_shard(
    states: list[dict[str, Any]],
    worker_index: int,
    num_workers: int,
) -> dict[str, Any] | None:
    assigned = [
        item for item in states if int(item["index"]) % num_workers == worker_index
    ]
    for item in assigned:
        if not item["marker"] and item["has_log"]:
            return item
    for item in assigned:
        if not item["marker"]:
            return item
    return assigned[-1] if assigned else None


def format_count(value: int | None) -> str:
    return "?" if value is None else str(value)


def main() -> None:
    args = parse_args()
    if args.expected_shards <= 0:
        raise ValueError("--expected_shards must be positive")
    if args.expected_tasks_per_shard <= 0:
        raise ValueError("--expected_tasks_per_shard must be positive")
    if args.num_workers <= 0:
        raise ValueError("--num_workers must be positive")
    if args.unit and len(args.unit) != args.num_workers:
        raise ValueError("provide exactly one --unit per worker")
    if args.gpu_index and len(args.gpu_index) != args.num_workers:
        raise ValueError("provide exactly one --gpu_index per worker")

    states = [
        shard_state(
            args.render_root,
            args.log_root,
            args.source,
            shard_index,
            args.expected_tasks_per_shard,
        )
        for shard_index in range(args.expected_shards)
    ]

    done = sum(int(item["done"]) for item in states)
    expected = args.expected_shards * args.expected_tasks_per_shard
    markers = sum(bool(item["marker"]) for item in states)
    known_accepted = sum(
        int(item["accepted"]) for item in states if item["accepted"] is not None
    )
    known_failed = sum(
        int(item["failed"]) for item in states if item["failed"] is not None
    )
    percent = 100.0 * done / expected

    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"时间: {now}")
    print(
        f"{args.source}总进度: {done}/{expected} ({percent:.2f}%) | "
        f"完成shard: {markers}/{args.expected_shards} | "
        f"当前已知accepted={known_accepted} failed={known_failed}"
    )
    print(
        "说明: 进度按已完成sequence计数；首个sequence完成前显示0，"
        "不等于worker停止。"
    )
    print()

    unit_states: list[str] = []
    for worker_index in range(args.num_workers):
        unit = args.unit[worker_index] if args.unit else None
        unit_value = service_state(unit) if unit is not None else "未配置"
        unit_states.append(unit_value)
        item = current_worker_shard(states, worker_index, args.num_workers)
        assigned = list(range(worker_index, args.expected_shards, args.num_workers))
        gpu = args.gpu_index[worker_index] if args.gpu_index else None
        prefix = (
            f"worker{worker_index}"
            f"(GPU {gpu if gpu is not None else '?'}, service={unit_value}, "
            f"assigned={assigned})"
        )
        if item is None:
            print(f"{prefix}: 无分片")
            continue
        details = [
            f'{item["name"]}: {item["status"]}',
            f'{item["done"]}/{item["total"]}',
            f'accepted={format_count(item["accepted"])}',
            f'failed={format_count(item["failed"])}',
        ]
        if item["elapsed"] is not None:
            details.append(f'elapsed={item["elapsed"]}')
        if item["eta"] is not None:
            details.append(f'shard ETA={item["eta"]}')
        if item["last"] is not None:
            details.append(f'last={item["last"]}')
        if item["log_age_seconds"] is not None:
            details.append(f'日志更新={item["log_age_seconds"]}秒前')
        print(f"{prefix}: " + " | ".join(details))

    print()
    for worker_index, gpu_index in enumerate(args.gpu_index):
        value = gpu_state(gpu_index)
        if value is not None:
            print(f"worker{worker_index} GPU(index, MiB used/total, util%): {value}")

    print()
    if markers == args.expected_shards:
        print("完成: 全部shard marker已生成，可以执行新的D8汇总硬门。")
    elif args.unit and all(
        value in {"failed", "inactive", "maintenance", "unknown"}
        for value in unit_states
    ):
        print("警告: worker服务均不活跃，但任务未完成；检查对应shard日志。")
    elif args.unit and any(value.startswith("不可查询") for value in unit_states):
        print("服务状态不可查询；当前进度仍可由持续更新的shard日志确认。")
    else:
        print("运行中: 不要重复启动相同root的worker。")


if __name__ == "__main__":
    main()
