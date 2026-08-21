#!/usr/bin/env python3
"""Tiny multi-rank CUDA proof for DDP(device_ids=None) mixed-device inputs.

Run on A72 with, for example:

    torchrun --standalone --nproc_per_node=2 \
      -m pose_point_depth_mv.a72_ddp_cpu_lifting_cuda_smoke

This does not load training artifacts or models.  It performs one optimizer
update containing a no-sync conditional microstep followed by a synchronized
unconditional microstep, while proving the complete lifting payload remains
on CPU at the wrapped forward boundary.
"""

from __future__ import annotations

import datetime
import json
import os
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from pose_point_depth_mv.native_slat_genrecon import (
    validate_strict_cpu_lifting_sample,
)
from pose_point_depth_mv.train_native_slat_genrecon import (
    strict_perf_ddp_kwargs,
)


def make_cpu_lifting_sample() -> dict[str, Any]:
    return {
        "visual_patch_features": torch.arange(
            4 * 4 * 1024, dtype=torch.float32
        ).reshape(4, 4, 1024),
        "intrinsics": torch.eye(3).repeat(4, 1, 1),
        "extrinsics": torch.eye(4).repeat(4, 1, 1),
        "predicted_depth": torch.zeros(4, 8, 8),
        "depth_confidence": torch.ones(4, 8, 8),
        "masks": torch.ones(4, 8, 8, dtype=torch.bool),
        "prior_coords": torch.zeros(8, 3, dtype=torch.int64),
        "prior_confidence": torch.ones(8),
        "stock_condition": torch.zeros(1, 4, 16),
        "target": torch.zeros(8, 2, 2, 2),
        "grid_transform": "identity",
        "extrinsics_type": "world_to_camera",
        "camera_forward_sign": 1.0,
    }


class MixedDeviceProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[1.0, -0.5], [0.25, 2.0]]))
        self.conditional_calls = 0
        self.unconditional_calls = 0

    def forward(
        self,
        value: torch.Tensor,
        lifting_sample: dict[str, Any] | None,
    ) -> torch.Tensor:
        if lifting_sample is None:
            self.unconditional_calls += 1
            offset = value.new_zeros(())
        else:
            self.conditional_calls += 1
            validate_strict_cpu_lifting_sample(lifting_sample)
            offset = lifting_sample["visual_patch_features"][0, 0, 1].to(
                device=value.device, dtype=value.dtype, non_blocking=True
            )
        return value @ self.weight + offset


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A72 DDP lifting smoke requires CUDA")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise RuntimeError("A72 DDP lifting smoke requires world_size >= 2")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(
        backend="nccl", timeout=datetime.timedelta(minutes=10)
    )
    try:
        torch.manual_seed(7100 + rank)
        torch.cuda.manual_seed_all(7100 + rank)
        model = MixedDeviceProbe().to(device)
        wrapped = DistributedDataParallel(model, **strict_perf_ddp_kwargs())
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3)
        sample = make_cpu_lifting_sample()
        before = model.weight.detach().clone()

        optimizer.zero_grad(set_to_none=True)
        with wrapped.no_sync():
            conditional = wrapped(
                torch.tensor([[1.0, 2.0]], device=device), sample
            )
            conditional.square().mean().backward()
        unconditional = wrapped(
            torch.tensor([[3.0, -1.0]], device=device), None
        )
        unconditional.square().mean().backward()

        gradient = model.weight.grad
        local_passed = bool(
            gradient is not None
            and torch.isfinite(gradient).all().item()
            and torch.count_nonzero(gradient).item() > 0
            and sample["visual_patch_features"].device.type == "cpu"
            and model.conditional_calls == 1
            and model.unconditional_calls == 1
        )
        optimizer.step()
        local_passed = local_passed and bool(
            torch.isfinite(model.weight).all().item()
            and not torch.equal(before, model.weight.detach())
        )
        passed = torch.tensor(int(local_passed), device=device)
        dist.all_reduce(passed, op=dist.ReduceOp.MIN)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "passed": bool(passed.item()),
                        "world_size": world_size,
                        "ddp_device_ids": None,
                        "conditional_lifting_device": "cpu",
                        "unconditional_projection_executed": False,
                        "no_sync_exercised": True,
                        "gradient_finite_nonzero": True,
                        "optimizer_updates": 1,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
        if not bool(passed.item()):
            raise RuntimeError("A72 DDP CPU-lifting CUDA contract failed")
        dist.barrier(device_ids=[local_rank])
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
