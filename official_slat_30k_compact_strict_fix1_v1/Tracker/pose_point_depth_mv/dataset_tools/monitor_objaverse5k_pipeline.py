#!/usr/bin/env python3
"""Print a concise live status table for the split-root Objaverse5K pipeline."""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
from pathlib import Path

from .monitor_mixed_render_smoke import latest_progress


DEFAULT_LIMITED_RENDER = Path(
    "/data/zjr/objaverse5k_single_subject_render512_20260810_v1"
)
DEFAULT_WIDE_RENDER = Path(
    "/data/zjr/objaverse5k_single_subject_render512_wideorbit_20260811_v2"
)
DEFAULT_WIDE_LOG = Path(
    "/data/zjr/objaverse5k_single_subject_render512_wideorbit_logs_20260811_v2"
)
DEFAULT_WORK = Path(
    "/data/zjr/objaverse5k_direct_dino_cache_shards_20260811_v1"
)
CPU_UNIT = "tracker-objaverse5k-cpu-cache-shards-v1.service"
DINO_UNIT = "tracker-objaverse5k-direct-dino-shards-v1.service"
DINO_PARTITION_UNIT = "tracker-objaverse5k-direct-dino-p{partition}-v1.service"
WIDE_UNIT = "objaverse5k-wide-render-w{worker}-20260811-v2.service"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limited_render", type=Path, default=DEFAULT_LIMITED_RENDER)
    parser.add_argument("--wide_render", type=Path, default=DEFAULT_WIDE_RENDER)
    parser.add_argument("--wide_log", type=Path, default=DEFAULT_WIDE_LOG)
    parser.add_argument("--work_root", type=Path, default=DEFAULT_WORK)
    return parser.parse_args()


def yes_no(value: bool) -> str:
    return "Y" if value else "-"


def format_age(seconds: int | None) -> str:
    if seconds is None:
        return "?"
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m{seconds % 60:02d}s"


def log_age(path: Path) -> int | None:
    try:
        return max(0, int(dt.datetime.now().timestamp() - path.stat().st_mtime))
    except OSError:
        return None


def marker(root: Path, shard: int, name: str) -> bool:
    return (root / "shards" / f"shard_{shard:03d}" / name).is_file()


def dino_marker(root: Path, shard: int) -> bool:
    shard_root = root / "shards" / f"shard_{shard:03d}"
    return any(shard_root.glob("**/_DINO_ONLY_LIFTING_COMPLETE.json"))


def service_states(units: list[str]) -> dict[str, str]:
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", *units],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {unit: "unavailable" for unit in units}
    values = result.stdout.splitlines()
    return {
        unit: values[index].strip() if index < len(values) else "not-found"
        for index, unit in enumerate(units)
    }


def gpu_states() -> list[str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return [line for line in rows if int(line.split(",", maxsplit=1)[0]) in range(5)]


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for shard in range(16):
        limited = shard < 8
        render_root = args.limited_render if limited else args.wide_render
        render_dir = render_root / "objaverse" / f"shard_{shard:03d}"
        render_ready = (render_dir / "_WORKER_COMPLETE.json").is_file()
        cpu_ready = marker(args.work_root, shard, "_CPU_SHARD_READY.json")
        dino_ready = dino_marker(args.work_root, shard)
        progress = None
        age = None
        if not limited:
            log_path = args.wide_log / "objaverse" / f"shard_{shard:03d}.log"
            progress = latest_progress(log_path)
            age = log_age(log_path)
        rows.append(
            {
                "shard": shard,
                "source": "limited" if limited else "wide",
                "render": render_ready,
                "cpu": cpu_ready,
                "dino": dino_ready,
                "progress": progress,
                "age": age,
            }
        )

    limited_ready = sum(bool(row["render"]) for row in rows[:8])
    wide_ready = sum(bool(row["render"]) for row in rows[8:])
    cpu_ready = sum(bool(row["cpu"]) for row in rows)
    dino_ready = sum(bool(row["dino"]) for row in rows)
    wide_units = [WIDE_UNIT.format(worker=worker) for worker in range(4)]
    dino_partition_units = [
        DINO_PARTITION_UNIT.format(partition=partition) for partition in range(4)
    ]
    units = [CPU_UNIT, DINO_UNIT, *wide_units, *dino_partition_units]
    states = service_states(units)
    cpu_state = states[CPU_UNIT]
    dino_state = states[DINO_UNIT]
    active_dino_partitions = [
        partition
        for partition, unit in enumerate(dino_partition_units)
        if states[unit] == "active"
    ]

    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(now)
    print(
        f"SUMMARY  render limited={limited_ready}/8 wide={wide_ready}/8  "
        f"CPU={cpu_ready}/16  DINO={dino_ready}/16"
    )
    partition_text = (
        ",".join(f"p{partition}/GPU{partition + 1}" for partition in active_dino_partitions)
        or "none"
    )
    print(
        f"WATCHERS CPU={cpu_state}  DINO-legacy={dino_state}  "
        f"DINO-partitions={partition_text}"
    )
    if dino_ready < 16 and dino_state != "active" and not active_dino_partitions:
        print("WARNING  DINO watcher is not active; CPU-ready shards will not advance.")

    print()
    print("SHARD  ROOT     RENDER  CPU  DINO  LIVE RENDER PROGRESS")
    for row in rows:
        progress = row["progress"]
        live = ""
        if not row["render"]:
            if isinstance(progress, dict) and isinstance(progress.get("done"), int):
                done = int(progress["done"])
                total = int(progress.get("total") or 0)
                accepted = progress.get("accepted")
                failed = progress.get("failed")
                eta = progress.get("eta") or "?"
                last = progress.get("last") or "?"
                live = (
                    f"{done:>3}/{total:<3} acc={accepted!s:<3} fail={failed!s:<3} "
                    f"ETA={eta:<8} last={last:<24} age={format_age(row['age'])}"
                )
            elif row["source"] == "wide":
                live = f"waiting/no progress line  age={format_age(row['age'])}"
        print(
            f"{int(row['shard']):03d}    {str(row['source']):<7}  "
            f"{yes_no(bool(row['render'])):^6}  {yes_no(bool(row['cpu'])):^3}  "
            f"{yes_no(bool(row['dino'])):^4}  {live}"
        )

    print()
    print("WORKERS (wide render)")
    for worker in range(4):
        unit = WIDE_UNIT.format(worker=worker)
        assigned = (8 + worker, 12 + worker)
        state = states[unit]
        current = next(
            (
                row
                for row in rows
                if row["shard"] in assigned and not bool(row["render"])
            ),
            None,
        )
        current_name = "complete" if current is None else f"shard_{int(current['shard']):03d}"
        print(
            f"w{worker} GPU{worker + 1}: {state:<10} assigned={assigned} "
            f"current={current_name}"
        )

    print()
    print("GPU (index, MiB used/total, util%)")
    for value in gpu_states():
        print(value)


if __name__ == "__main__":
    main()
