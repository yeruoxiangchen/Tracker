"""Load the already CUDA-validated strict-fix1 implementation by exact hash.

The performance implementation remains an immutable source input.  This
package layers the official with-VGGT scientific contract on top without
editing either ``pose_point_depth_mv`` or the preserved strict-fix1 tree.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any


RUNTIME_VERSION = "official_slat_with_vggt.strict_perf_v1"
STRICT_TRAIN_SHA256 = "45978dd8a809e12245bdd9d831345f9cb7e5ea4526893e31f81cfe32c20664d5"
STRICT_PROJECTION_SHA256 = (
    "349a4df4402997d0f5858ea967faec0e4b18a60a4f4f9889c27edbb899633987"
)
SCIENTIFIC_SOURCE_SHA256 = {
    # Scientific with-VGGT delta.
    "pose_point_depth_mv/proobjaverse_official_slat_with_vggt_cache.py": (
        "c9335be738bcfbc7e0876d0c34c898bde829dd74d085e395e6224b5e7e94826a"
    ),
    "pose_point_depth_mv/native_slat_genrecon_with_vggt_official.py": (
        "fea22e3f4f9ef35ab3254acc8619e70e5f19312f7e854fe68c3bfbb0e2bafb26"
    ),
    "pose_point_depth_mv/train_native_slat_genrecon_with_vggt_official.py": (
        "db9927351c90b81b2c90fbf275d974a5f1e2c2f8a83537599482a552a4749b55"
    ),
    # Existing helpers that the dynamically loaded strict trainer/projection
    # resolve through absolute imports.  Locking them makes that composition
    # explicit and prevents a later dirty-tree edit from silently changing the
    # supposedly isolated runtime.
    "pose_point_depth_mv/native_slat_genrecon_v2.py": (
        "88a480a751adea51a2f023ea6a0f3772577400d222ceea3f71fe858f70774648"
    ),
    "pose_point_depth_mv/native_3d_condition.py": (
        "3e1717e1dc2e2d8ca2802ab47fa6ced47b503eb108091ba981886968cc43e30f"
    ),
    "pose_point_depth_mv/direct_slat_data.py": (
        "794461a80f30e0ef581474a718d6ce828ee83860b5ca95725f27305e3cc445d9"
    ),
    "pose_point_depth_mv/direct_slat_flow.py": (
        "78096b545158428529ffa1281fff38fe0ef0319fd0fb81a163061b688948fc73"
    ),
    "pose_point_depth_mv/proobjaverse_official_slat_training.py": (
        "a87c97561981e8b34735d668d1d24973975617cdbd67ce51921f1ee909e71978"
    ),
    "ar_ss_flow/local_pose_lifting_flow.py": (
        "4d25dc862f5a46430625eae52c9d9da11e76308c61f7a378d4b563d880c1aff3"
    ),
    "pose_point_depth_mv/train_native_slat_genrecon_no_vggt.py": (
        "418df0d34b126290ef0ce7cf756afed8cc390eabfe7f8bc6bdcc8a7c3ebe5a4f"
    ),
    "pose_point_depth_mv/no_vggt_ss_evidence.py": (
        "d3e50138d70b1532756a4dc0748413bd74cbfb1c89decddeb0e65108758e0a98"
    ),
    "pose_point_depth_mv/evaluate_native_ss_stock_slat_mesh.py": (
        "0d70a573bca7c89b6551305e99a253ae94f0139b790b2aba306dfbb56f4c92a9"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tracker_root() -> Path:
    return Path(__file__).resolve().parents[1]


def strict_fix1_root() -> Path:
    override = os.environ.get("WITH_VGGT_STRICT_FIX1_ROOT", "")
    if override:
        return Path(override).expanduser().resolve(strict=True)
    return (
        tracker_root() / "a72_perf_v1_fix1_testcompat1/Tracker"
    ).resolve(strict=True)


def validate_runtime_sources() -> dict[str, Any]:
    strict_root = strict_fix1_root()
    strict_sources = {
        "trainer": (
            strict_root / "pose_point_depth_mv/train_native_slat_genrecon.py",
            STRICT_TRAIN_SHA256,
        ),
        "projection": (
            strict_root / "pose_point_depth_mv/native_slat_genrecon.py",
            STRICT_PROJECTION_SHA256,
        ),
    }
    strict_identity: dict[str, dict[str, str]] = {}
    for label, (path, expected) in strict_sources.items():
        path = path.resolve(strict=True)
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"strict-fix1 {label} source changed: {actual} != {expected}: {path}"
            )
        strict_identity[label] = {
            "path": str(path),
            "sha256": actual,
        }

    scientific_identity: dict[str, dict[str, str]] = {}
    root = tracker_root()
    for relative, expected in SCIENTIFIC_SOURCE_SHA256.items():
        path = (root / relative).resolve(strict=True)
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"with-VGGT scientific source changed: {actual} != {expected}: {path}"
            )
        scientific_identity[relative] = {
            "path": str(path),
            "sha256": actual,
        }
    return {
        "version": RUNTIME_VERSION,
        "strict_fix1_sources": strict_identity,
        "scientific_sources": scientific_identity,
        "scientific_math_changed": False,
        "checkpoint_scientific_identity_changed": False,
    }


def _load_source_module(name: str, path: Path) -> ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load strict runtime source: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@functools.lru_cache(maxsize=1)
def load_strict_modules() -> tuple[ModuleType, ModuleType, dict[str, Any]]:
    identity = validate_runtime_sources()
    root = strict_fix1_root() / "pose_point_depth_mv"
    trainer = _load_source_module(
        f"{__package__}._strict_train_base",
        root / "train_native_slat_genrecon.py",
    )
    projection = _load_source_module(
        f"{__package__}._strict_projection_base",
        root / "native_slat_genrecon.py",
    )
    if trainer.strict_perf_ddp_kwargs().get("device_ids", "missing") is not None:
        raise RuntimeError("strict-fix1 DDP input policy is no longer device_ids=None")
    if not hasattr(projection, "validate_strict_cpu_lifting_sample"):
        raise RuntimeError("strict-fix1 CPU lifting contract helper is missing")
    return trainer, projection, identity


__all__ = [
    "RUNTIME_VERSION",
    "load_strict_modules",
    "strict_fix1_root",
    "tracker_root",
    "validate_runtime_sources",
]
