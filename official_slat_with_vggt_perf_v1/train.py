"""Compose official with-VGGT science with the verified strict-fix1 runtime."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
from typing import Any, Callable

from pose_point_depth_mv import (
    native_slat_genrecon_v2 as _v2_model,
)
from pose_point_depth_mv import (
    native_slat_genrecon_with_vggt_official as _science_model,
)
from pose_point_depth_mv import (
    train_native_slat_genrecon_with_vggt_official as _science_arm,
)

from .dataset import StrictWithVGGTNativeConditionSLatDataset
from .runtime import RUNTIME_VERSION, load_strict_modules


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _argument(name: str) -> str | None:
    values: list[str] = []
    prefix = f"{name}="
    for index, value in enumerate(sys.argv):
        if value == name:
            if index + 1 >= len(sys.argv) or sys.argv[index + 1].startswith("--"):
                raise ValueError(f"{name} requires exactly one value")
            values.append(sys.argv[index + 1])
        elif value.startswith(prefix):
            values.append(value[len(prefix) :])
    if len(values) > 1:
        raise ValueError(f"duplicate runtime argument {name}: {values}")
    return values[0] if values else None


def _default_argument(name: str, value: str) -> None:
    if _argument(name) is None:
        sys.argv.extend((name, value))


def _runtime_binding(identity: dict[str, Any]) -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent
    return {
        "version": RUNTIME_VERSION,
        "strict_fix1_trainer_sha256": identity["strict_fix1_sources"]["trainer"][
            "sha256"
        ],
        "strict_fix1_projection_sha256": identity["strict_fix1_sources"][
            "projection"
        ]["sha256"],
        "dataset_source_sha256": _sha256(package_root / "dataset.py"),
        "composition_source_sha256": _sha256(Path(__file__).resolve()),
        "ddp_device_ids": None,
        "lifting_policy": "CPU select then selected DINO/K/T nonblocking H2D",
        "stock_context_policy": "CPU select branch/views then selected-only H2D",
        "legacy_condition_file_loaded_per_sample": False,
        "native_vggt_forward_during_training": False,
        "scientific_math_changed": False,
        "checkpoint_scientific_identity_changed": False,
    }


def configure_runtime() -> tuple[Any, dict[str, Any]]:
    strict_train, strict_projection, identity = load_strict_modules()
    strict_train.NativeConditionSLatDataset = (
        StrictWithVGGTNativeConditionSLatDataset
    )
    _v2_model.project_sparse_frustum_dino = (
        strict_projection.project_sparse_frustum_dino
    )

    original_builder = _science_model.build_native_slat_official_with_vggt_components

    def build_components(**kwargs: Any):
        sampler, model, decoder, summary, defaults, normalization = original_builder(
            **kwargs
        )
        summary = {
            **summary,
            "strict_with_vggt_runtime": _runtime_binding(identity),
        }
        return sampler, model, decoder, summary, defaults, normalization

    # Reuse the frozen scientific wrapper, but direct all of its symbol patches
    # to the verified strict trainer rather than the legacy source trainer.
    _science_arm._train = strict_train
    _science_arm._base_initial_stock_audit = strict_train.initial_stock_audit
    _science_arm.WithVGGTNativeConditionSLatDataset = (
        StrictWithVGGTNativeConditionSLatDataset
    )
    _science_arm.build_native_slat_official_with_vggt_components = build_components
    return strict_train, identity


def main(
    *,
    decoder_validator: Callable[..., dict[str, Any]] | None = None,
) -> None:
    if "--skip_redundant_cache_finite_checks" in sys.argv:
        raise ValueError(
            "official with-VGGT strict v1 retains per-sample finite checks; "
            "the audited-cache profile is intentionally disabled"
        )
    for name, value in (
        ("--num_workers", "2"),
        ("--prefetch_factor", "2"),
        ("--torch_num_threads", "2"),
        ("--torch_num_interop_threads", "1"),
    ):
        _default_argument(name, value)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    strict_train, _ = configure_runtime()
    if decoder_validator is not None:
        strict_train.validate_decoder_audit = decoder_validator
    _science_arm.main()


if __name__ == "__main__":
    main()


__all__ = ["configure_runtime", "main"]
