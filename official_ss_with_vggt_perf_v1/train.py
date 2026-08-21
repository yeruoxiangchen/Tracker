"""Fresh official Train2000 with-VGGT Native-SS entrypoint."""

from __future__ import annotations

import sys
from typing import Any

from torch.nn.parallel import DistributedDataParallel as TorchDistributedDataParallel

from official_ss_with_vggt_perf_v1.cache import (
    WithVGGTOfficialSSDataset,
    validate_official_ss_with_vggt_cache_contract,
)
from official_ss_with_vggt_perf_v1.model import (
    VERSION,
    build_components,
    validate_checkpoint,
)
from pose_point_depth_mv import train_native_ss_genrecon as _trainer


_FROZEN_ARGS = {
    "--max_steps": "2000",
    "--seed": "42",
    "--lora_rank": "8",
    "--lora_alpha": "16",
    "--condition_channels": "1024",
    "--new_lr": "0.00005",
    "--lora_lr": "0.00001",
    "--new_weight_decay": "0.01",
    "--grad_clip": "1.0",
    "--warmup_ratio": "0.02",
    "--ema_decay": "0.9995",
    "--p_uncond": "0.1",
    "--t_logit_mean": "1.0",
    "--t_logit_std": "1.0",
    "--min_condition_views": "1",
    "--max_condition_views": "8",
    "--amp_dtype": "bf16",
}


def _ddp_preserve_cpu_sample(module, *args: Any, **kwargs: Any):
    """Keep the nested lifting sample on CPU until projection consumes it.

    The reused trainer explicitly places ``x_t``, ``t``, Stock context and
    Stock velocity on the local CUDA device.  Its remaining ``sample`` kwarg
    contains the much larger lifting payload.  A non-empty DDP ``device_ids``
    recursively migrates that payload before ``forward``; using ``None`` is
    the already validated single-process-per-GPU mixed-device DDP policy.
    """

    if args:
        raise TypeError("with-VGGT SS DDP requires keyword-only configuration")
    requested = kwargs.pop("device_ids", None)
    kwargs.pop("output_device", None)
    if requested is None or len(requested) != 1:
        raise RuntimeError(
            "with-VGGT SS expected one local device before installing the "
            f"CPU-payload DDP policy, got={requested!r}"
        )
    kwargs["device_ids"] = None
    return TorchDistributedDataParallel(module, **kwargs)


def _argument(flag: str) -> str | None:
    values: list[str] = []
    prefix = f"{flag}="
    for position, value in enumerate(sys.argv):
        if value == flag:
            if position + 1 >= len(sys.argv) or sys.argv[position + 1].startswith(
                "--"
            ):
                raise ValueError(f"{flag} requires a value")
            values.append(sys.argv[position + 1])
        elif value.startswith(prefix):
            values.append(value[len(prefix) :])
    if len(values) > 1:
        raise ValueError(f"duplicate scientific argument {flag}: {values}")
    return values[0] if values else None


def _freeze_scientific_args() -> None:
    short_smoke = "--allow_short_smoke" in sys.argv
    if short_smoke:
        sys.argv.remove("--allow_short_smoke")
    if _argument("--init_checkpoint") is not None:
        raise ValueError(
            "with-VGGT official SS is a fresh Stock-equivalent run; "
            "--init_checkpoint is forbidden"
        )
    for flag, expected in _FROZEN_ARGS.items():
        observed = _argument(flag)
        if flag == "--max_steps" and short_smoke:
            if observed is None:
                sys.argv.extend((flag, "2"))
            elif not 1 <= int(observed) <= 5:
                raise ValueError("short smoke max_steps must lie in [1,5]")
            continue
        if observed is None:
            sys.argv.extend((flag, expected))
        elif flag == "--amp_dtype" and observed != expected:
            raise ValueError(f"{flag} is frozen to {expected}, got {observed}")
        elif flag != "--amp_dtype" and float(observed) != float(expected):
            raise ValueError(f"{flag} is frozen to {expected}, got {observed}")
    if "--gradient_checkpointing" not in sys.argv:
        sys.argv.append("--gradient_checkpointing")


def main() -> None:
    _freeze_scientific_args()
    _trainer.PoseLiftingCacheDataset = WithVGGTOfficialSSDataset
    _trainer.NATIVE_SS_GENRECON_VERSION = VERSION
    _trainer.validate_genrecon_cache_contract = (
        validate_official_ss_with_vggt_cache_contract
    )
    _trainer.validate_native_ss_genrecon_checkpoint = validate_checkpoint
    _trainer.build_native_ss_genrecon_components = build_components
    _trainer.DistributedDataParallel = _ddp_preserve_cpu_sample
    _trainer.main()


if __name__ == "__main__":
    main()
