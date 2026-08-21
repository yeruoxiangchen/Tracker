#!/usr/bin/env python3
"""Shared immutable bindings for the real Omni benchmark front end."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


def atomic_torch_save(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, destination)


def atomic_npz(path: str | Path, **arrays: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, destination)


def to_cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: to_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu_tree(item) for item in value)
    return value


def to_device_tree(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: to_device_tree(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(to_device_tree(item, device) for item in value)
    return value


def object_key(row: dict[str, Any]) -> str:
    return f"{row['category']}:{row['object_id']}"


def index_objects(rows: Iterable[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = object_key(row)
        if key in output:
            raise RuntimeError(f"duplicate {label} object={key}")
        output[key] = row
    if not output:
        raise RuntimeError(f"{label} contains no objects")
    return output


def parse_object_selection(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    selected = {str(value) for value in values}
    if len(selected) != len(values):
        raise ValueError("--object values must be unique category:object_id keys")
    if any(":" not in value for value in selected):
        raise ValueError("--object values must use category:object_id")
    return selected


def select_rows(
    rows: Iterable[dict[str, Any]], values: list[str] | None
) -> list[dict[str, Any]]:
    indexed = index_objects(rows, label="input")
    requested = parse_object_selection(values)
    if requested is None:
        return [indexed[key] for key in sorted(indexed)]
    missing = sorted(requested - set(indexed))
    if missing:
        raise KeyError(f"selected objects are absent: {missing}")
    return [indexed[key] for key in sorted(requested)]


def validate_bound_file(path: str | Path, digest: str, *, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file() or sha256_file(resolved) != str(digest):
        raise RuntimeError(f"{label} binding changed: {resolved}")
    return resolved


def resolve_torch_device(value: str | torch.device) -> torch.device:
    """Resolve bare ``cuda`` to the first logical CUDA_VISIBLE_DEVICES entry."""

    device = torch.device(value)
    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    index = 0 if device.index is None else int(device.index)
    torch.cuda.set_device(index)
    return torch.device("cuda", index)
