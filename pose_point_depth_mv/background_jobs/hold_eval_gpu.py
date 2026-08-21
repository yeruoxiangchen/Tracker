#!/usr/bin/env python3
"""Keep a small, idle CUDA allocation alive for an evaluation job.

This is an advisory reservation: it makes the selected physical GPU visibly
occupied in nvidia-smi while independent evaluation workers come and go.  It
does not provide scheduler- or driver-level exclusivity.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
from datetime import datetime, timezone
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu", required=True)
    parser.add_argument("--memory-mib", type=int, default=64)
    parser.add_argument("--label", required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.memory_mib <= 0:
        raise ValueError("--memory-mib must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the GPU reservation process")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "GPU reservation requires exactly one CUDA_VISIBLE_DEVICES entry; "
            f"visible_device_count={torch.cuda.device_count()}"
        )

    torch.cuda.set_device(0)
    # uint8 makes the requested allocation size exact and easy to audit.
    allocation = torch.empty(
        args.memory_mib * 1024 * 1024,
        dtype=torch.uint8,
        device="cuda:0",
    )
    allocation.zero_()
    torch.cuda.synchronize()

    payload = {
        "format": "pose_point_depth_mv.eval_gpu_reservation.v1",
        "ready": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "label": args.label,
        "physical_gpu": str(args.physical_gpu),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_device": 0,
        "device_name": torch.cuda.get_device_name(0),
        "requested_allocation_mib": args.memory_mib,
        "torch_allocated_bytes": torch.cuda.memory_allocated(0),
    }
    args.ready_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.ready_file.with_name(
        f"{args.ready_file.name}.tmp.{os.getpid()}"
    )
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.ready_file)
    print(json.dumps(payload, ensure_ascii=False), flush=True)

    stop = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    stop.wait()

    del allocation
    torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "stopped": True,
                "pid": os.getpid(),
                "physical_gpu": str(args.physical_gpu),
                "time_utc": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
