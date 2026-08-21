"""Shared immutable contracts for Bunny model reconstruction review."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable


PROTOCOL_FORMAT = "pose_point_depth_mv.bunny_review_protocol.v1"
METHOD_RESULT_FORMAT = "pose_point_depth_mv.bunny_method_result.v1"
REPORT_FORMAT = "pose_point_depth_mv.bunny_review_report.v1"
ADAPTER_FORMAT = "pose_point_depth_mv.bunny_command_adapter.v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def binding(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": int(resolved.stat().st_size),
    }


def validate_binding(value: dict[str, Any], *, label: str) -> Path:
    if not isinstance(value, dict):
        raise TypeError(f"{label} binding is not a dictionary")
    path = Path(str(value.get("path", ""))).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    actual = sha256_file(path)
    if actual != str(value.get("sha256", "")):
        raise RuntimeError(
            f"{label} changed after freeze: {path} {actual} != {value.get('sha256')}"
        )
    if "bytes" in value and int(value["bytes"]) != int(path.stat().st_size):
        raise RuntimeError(f"{label} size changed after freeze: {path}")
    return path


def atomic_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def atomic_text(path: str | Path, value: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, destination)


def atomic_copy(source: str | Path, destination: str | Path) -> None:
    source_path = Path(source).resolve()
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_name(
        f".{destination_path.name}.tmp-{os.getpid()}"
    )
    shutil.copy2(source_path, temporary)
    os.replace(temporary, destination_path)


def load_protocol(path: str | Path, *, verify_files: bool = True) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    if payload.get("format") != PROTOCOL_FORMAT:
        raise ValueError(f"unsupported Bunny protocol={payload.get('format')!r}")
    expected = str(payload.get("protocol_sha256", ""))
    body = dict(payload)
    body.pop("protocol_sha256", None)
    actual = canonical_sha256(body)
    if not expected or actual != expected:
        raise RuntimeError(f"Bunny protocol hash mismatch: {actual} != {expected}")
    if verify_files:
        validate_binding(payload["reference"]["mesh"], label="reference mesh")
        for index, view in enumerate(payload["views"]):
            validate_binding(view["source"], label=f"view[{index}].source")
            validate_binding(view["rgba"], label=f"view[{index}].rgba")
            validate_binding(view["mask"], label=f"view[{index}].mask")
    return payload


def method_dir(protocol_path: str | Path, method_id: str) -> Path:
    safe = str(method_id)
    if not safe or safe in {".", ".."} or "/" in safe or "\\" in safe:
        raise ValueError(f"unsafe method_id={method_id!r}")
    return Path(protocol_path).resolve().parent / "methods" / safe


def write_method_result(
    *,
    protocol_path: str | Path,
    method_id: str,
    display_name: str,
    mesh_path: str | Path,
    input_view_indices: Iterable[int],
    backend: dict[str, Any],
    auxiliary_meshes: dict[str, str | Path] | None = None,
    notes: list[str] | None = None,
) -> Path:
    protocol = load_protocol(protocol_path)
    input_indices = [int(value) for value in input_view_indices]
    primary = binding(mesh_path)
    auxiliary = {
        key: binding(value)
        for key, value in sorted((auxiliary_meshes or {}).items())
        if Path(value).is_file()
    }
    result = {
        "format": METHOD_RESULT_FORMAT,
        "complete": True,
        "protocol": binding(protocol_path),
        "protocol_sha256": protocol["protocol_sha256"],
        "method_id": str(method_id),
        "display_name": str(display_name),
        "input_view_indices": input_indices,
        "input_view_count": len(input_indices),
        "mesh": primary,
        "auxiliary_meshes": auxiliary,
        "backend": backend,
        "notes": list(notes or []),
    }
    result_path = method_dir(protocol_path, method_id) / "result.json"
    atomic_json(result_path, result)
    return result_path


def load_method_result(
    protocol_path: str | Path,
    method_id: str,
    *,
    verify_mesh: bool = True,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    result_path = method_dir(protocol_path, method_id) / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(
            f"missing method result: {result_path}; run/register {method_id} first"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("format") != METHOD_RESULT_FORMAT or result.get("complete") is not True:
        raise ValueError(f"incomplete/unsupported method result: {result_path}")
    if result.get("method_id") != method_id:
        raise RuntimeError(f"method ID mismatch in {result_path}")
    if result.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise RuntimeError(f"protocol binding mismatch in {result_path}")
    if verify_mesh:
        validate_binding(result["mesh"], label=f"{method_id}.mesh")
        for key, value in result.get("auxiliary_meshes", {}).items():
            validate_binding(value, label=f"{method_id}.auxiliary_meshes.{key}")
    return result


def code_bindings(paths: dict[str, str | Path]) -> dict[str, dict[str, Any]]:
    return {key: binding(value) for key, value in sorted(paths.items())}


def parse_int_csv(value: str) -> list[int]:
    values = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"invalid unique integer CSV={value!r}")
    return values
