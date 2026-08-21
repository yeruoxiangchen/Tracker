#!/usr/bin/env python3
"""Report live progress for a mixed-source multiview render smoke run."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PROGRESS_RE = re.compile(
    r"Building:\s*(?P<percent>\d+)%.*?\|\s*"
    r"(?P<done>\d+)/(?P<total>\d+)\s*\[(?P<timing>[^\]]+)\]"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render_root", type=Path, required=True)
    parser.add_argument("--log_root", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--sources", default="objaverse,omni")
    parser.add_argument("--expected_tasks_per_source", type=int, default=32)
    parser.add_argument("--gpu_index", type=int)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def latest_progress(log_path: Path) -> dict[str, Any] | None:
    try:
        text = log_path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return None

    progress_lines = [
        ANSI_ESCAPE.sub("", line)
        for line in text.replace("\r", "\n").splitlines()
        if "Building:" in line
    ]
    if not progress_lines:
        return None

    line = progress_lines[-1]
    match = PROGRESS_RE.search(line)
    if match is None:
        return {"line": line.strip()}

    timing = match.group("timing")
    elapsed = None
    eta = None
    timing_match = re.match(r"(?P<elapsed>[^<,]+)<(?P<eta>[^,]+),", timing)
    if timing_match is not None:
        elapsed = timing_match.group("elapsed").strip()
        eta = timing_match.group("eta").strip()

    def integer_field(name: str) -> int | None:
        field_match = re.search(rf"{name}=(\d+)", line)
        return int(field_match.group(1)) if field_match is not None else None

    last_match = re.search(r"last=([^,\]]+)", line)
    return {
        "done": int(match.group("done")),
        "total": int(match.group("total")),
        "percent": int(match.group("percent")),
        "accepted": integer_field("accepted"),
        "failed": integer_field("failed"),
        "elapsed": elapsed,
        "eta": eta,
        "last": last_match.group(1).strip() if last_match is not None else None,
        "line": line.strip(),
    }


def service_state(unit: str) -> str:
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", unit],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "不可查询"
    value = result.stdout.strip()
    known_states = {
        "active",
        "activating",
        "deactivating",
        "failed",
        "inactive",
        "maintenance",
        "reloading",
        "unknown",
    }
    return value if value in known_states else "不可查询/not-found"


def source_state(
    render_root: Path,
    log_root: Path,
    source: str,
    expected_tasks: int,
) -> dict[str, Any]:
    shard = render_root / source / "shard_000"
    log_path = log_root / source / "shard_000.log"
    marker_path = shard / "_WORKER_COMPLETE.json"
    manifest_path = shard / "manifest.json"
    manifest = read_json(manifest_path)
    progress = latest_progress(log_path)

    if manifest is not None:
        samples = manifest.get("samples")
        failures = manifest.get("failures")
        accepted = len(samples) if isinstance(samples, list) else None
        failed = len(failures) if isinstance(failures, list) else None
    else:
        accepted = None
        failed = None

    if marker_path.is_file():
        status = "完成"
    elif progress is not None and progress.get("done", 0) >= progress.get("total", 1):
        status = "已跑完、等待/未通过收尾"
    elif progress is not None:
        status = "运行中或已中断"
    elif shard.exists() or log_path.exists():
        status = "已创建、尚无任务进度"
    else:
        status = "未启动"

    if accepted is not None and failed is not None:
        done = accepted + failed
        total = max(expected_tasks, done)
    elif progress is not None and isinstance(progress.get("done"), int):
        done = int(progress["done"])
        total = int(progress.get("total") or expected_tasks)
        accepted = progress.get("accepted")
        failed = progress.get("failed")
    else:
        done = 0
        total = expected_tasks

    log_age_seconds = None
    try:
        log_age_seconds = max(0, int(dt.datetime.now().timestamp() - log_path.stat().st_mtime))
    except OSError:
        pass

    return {
        "source": source,
        "status": status,
        "done": done,
        "total": total,
        "accepted": accepted,
        "failed": failed,
        "elapsed": progress.get("elapsed") if progress else None,
        "eta": progress.get("eta") if progress else None,
        "last": progress.get("last") if progress else None,
        "log_age_seconds": log_age_seconds,
        "marker": marker_path.is_file(),
        "has_any_artifact": shard.exists() or log_path.exists(),
    }


def gpu_state(gpu_index: int) -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu_index),
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def main() -> None:
    args = parse_args()
    sources = [item.strip() for item in args.sources.split(",") if item.strip()]
    states = [
        source_state(
            args.render_root,
            args.log_root,
            source,
            args.expected_tasks_per_source,
        )
        for source in sources
    ]
    unit_state = service_state(args.unit)
    if unit_state == "active":
        active_source_seen = False
        for item in states:
            if item["marker"]:
                continue
            if not active_source_seen and item["has_any_artifact"]:
                item["status"] = "运行中"
                active_source_seen = True
            elif not active_source_seen:
                item["status"] = "初始化中"
                active_source_seen = True
            else:
                item["status"] = "等待前一来源"
    elif unit_state in {"failed", "inactive"}:
        for item in states:
            if item["has_any_artifact"] and not item["marker"]:
                item["status"] = "已中断或收尾失败"

    total_done = sum(item["done"] for item in states)
    total_tasks = sum(item["total"] for item in states)
    percent = 100.0 * total_done / total_tasks if total_tasks else 0.0

    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"时间: {now}")
    print(f"服务: {unit_state} ({args.unit})")
    print(f"总进度: {total_done}/{total_tasks} ({percent:.2f}%)")
    print()

    for item in states:
        accepted = "?" if item["accepted"] is None else str(item["accepted"])
        failed = "?" if item["failed"] is None else str(item["failed"])
        details = [
            f'{item["source"]}: {item["status"]}',
            f'{item["done"]}/{item["total"]}',
            f"accepted={accepted}",
            f"failed={failed}",
        ]
        if item["elapsed"] is not None:
            details.append(f'elapsed={item["elapsed"]}')
        if item["eta"] is not None:
            details.append(f'当前来源ETA={item["eta"]}')
        if item["last"] is not None:
            details.append(f'last={item["last"]}')
        if item["log_age_seconds"] is not None:
            details.append(f'日志更新={item["log_age_seconds"]}秒前')
        print(" | ".join(details))

    print()
    if not any(item["has_any_artifact"] for item in states):
        print("提示: D5 尚未启动或未创建 v4 输出；请先退出 watch 并执行 D5。")
    elif unit_state != "active" and not all(item["marker"] for item in states):
        print("警告: 服务当前不活跃且任务未全部完成；请检查 D5 unit 日志。")
    elif all(item["marker"] for item in states):
        print("完成: 两个来源均已有 _WORKER_COMPLETE.json，可以执行 D7。")

    if args.gpu_index is not None:
        gpu = gpu_state(args.gpu_index)
        if gpu is not None:
            print(f"渲染GPU(index, MiB used/total, util%): {gpu}")


if __name__ == "__main__":
    main()
