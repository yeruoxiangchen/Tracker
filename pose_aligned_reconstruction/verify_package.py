#!/usr/bin/env python3
"""Verify that the curated model package is self-contained from its archive."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any

from pose_aligned_reconstruction.current_30k import validate_frozen_assets


ARCHIVED_PACKAGE = "pose_point_depth_mv"
PACKAGE = "pose_aligned_reconstruction"
PACKAGE_ROOT = Path(__file__).resolve().parent
REQUIRED_ENTRYPOINTS = (
    "current_30k.py",
    "infer_current.py",
    "infer_real_proobjaverse_official_ss_slat.py",
    "materialize_proobjaverse_official_ss_targets.py",
    "prepare_proobjaverse_official_slat_compact_cache.py",
    "train_proobjaverse_official_native_ss_no_vggt.py",
    "train_proobjaverse_official_native_slat_no_vggt.py",
)
IMPORT_SMOKE_MODULES = (
    f"{PACKAGE}.train_proobjaverse_official_native_ss_no_vggt",
    f"{PACKAGE}.train_proobjaverse_official_native_slat_no_vggt",
    f"{PACKAGE}.prepare_proobjaverse_official_slat_compact_cache",
    f"{PACKAGE}.infer_current",
)
SOURCE_MANIFEST = PACKAGE_ROOT / "SOURCE_MANIFEST.sha256"


def _archived_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    failures: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""] if node.level == 0 else []
        else:
            continue
        for name in names:
            if name == ARCHIVED_PACKAGE or name.startswith(f"{ARCHIVED_PACKAGE}."):
                failures.append(f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno}:{name}")
    return failures


def verify_source_tree() -> dict[str, Any]:
    missing = [name for name in REQUIRED_ENTRYPOINTS if not (PACKAGE_ROOT / name).is_file()]
    imports: list[str] = []
    source_paths: list[str] = []
    symlinks: list[str] = []
    sources = sorted(PACKAGE_ROOT.rglob("*.py"))
    for path in sources:
        imports.extend(_archived_imports(path))
        text = path.read_text(encoding="utf-8")
        if f"/{ARCHIVED_PACKAGE}/" in text or f"Tracker/{ARCHIVED_PACKAGE}" in text:
            source_paths.append(str(path.relative_to(PACKAGE_ROOT)))
    for path in PACKAGE_ROOT.rglob("*"):
        if path.is_symlink():
            symlinks.append(str(path.relative_to(PACKAGE_ROOT)))
    if missing or imports or source_paths or symlinks:
        raise RuntimeError(
            json.dumps(
                {
                    "missing_entrypoints": missing,
                    "archived_package_imports": imports,
                    "archived_source_paths": source_paths,
                    "symlinks": symlinks,
                },
                indent=2,
            )
        )
    return {
        "python_source_count": len(sources),
        "required_entrypoints": list(REQUIRED_ENTRYPOINTS),
        "archived_package_import_count": 0,
        "archived_source_path_count": 0,
        "symlink_count": 0,
    }


def verify_source_manifest() -> dict[str, Any]:
    failures: list[str] = []
    checked = 0
    listed: set[Path] = set()
    for line in SOURCE_MANIFEST.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = (PACKAGE_ROOT / relative).resolve(strict=True)
        if PACKAGE_ROOT not in path.parents:
            raise RuntimeError(f"source manifest path escapes package: {relative}")
        listed.add(path)
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        checked += 1
        if observed != expected:
            failures.append(f"{relative}: observed={observed} expected={expected}")
    actual = {
        path.resolve()
        for path in PACKAGE_ROOT.rglob("*")
        if path.is_file()
        and path != SOURCE_MANIFEST
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }
    for path in sorted(actual - listed):
        failures.append(f"unlisted source file: {path.relative_to(PACKAGE_ROOT)}")
    for path in sorted(listed - actual):
        failures.append(f"missing source file: {path.relative_to(PACKAGE_ROOT)}")
    if failures:
        raise RuntimeError("source manifest differs:\n" + "\n".join(failures))
    return {"path": str(SOURCE_MANIFEST), "file_count": checked, "passed": True}


def verify_imports() -> dict[str, Any]:
    imported = []
    for name in IMPORT_SMOKE_MODULES:
        importlib.import_module(name)
        imported.append(name)
    leaked = sorted(
        name
        for name in sys.modules
        if name == ARCHIVED_PACKAGE or name.startswith(f"{ARCHIVED_PACKAGE}.")
    )
    if leaked:
        raise RuntimeError(f"archived package loaded during import smoke: {leaked}")
    return {"modules": imported, "archived_modules_loaded": []}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imports", action="store_true", help="import current entry points")
    parser.add_argument("--assets", action="store_true", help="validate frozen asset metadata")
    parser.add_argument(
        "--full-hash",
        action="store_true",
        help="with --assets, hash all frozen files including checkpoints",
    )
    args = parser.parse_args()
    if args.full_hash and not args.assets:
        parser.error("--full-hash requires --assets")
    result: dict[str, Any] = {
        "source_tree": verify_source_tree(),
        "source_manifest": verify_source_manifest(),
    }
    if args.imports:
        result["import_smoke"] = verify_imports()
    if args.assets:
        result["frozen_assets"] = validate_frozen_assets(full_hash=args.full_hash)
    result["passed"] = True
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
