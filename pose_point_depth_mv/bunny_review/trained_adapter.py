#!/usr/bin/env python3
"""Run or register a replaceable trained-model backend for Bunny review."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from .common import (
    ADAPTER_FORMAT,
    atomic_copy,
    atomic_json,
    binding,
    code_bindings,
    load_method_result,
    load_protocol,
    method_dir,
    parse_int_csv,
    write_method_result,
)


def safe_destination(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"adapter output escapes method directory: {relative}") from exc
    return path


def adapter_context(
    protocol_path: Path,
    method_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    ordered = sorted(protocol["views"], key=lambda row: int(row["view_index"]))
    rgba = [str(Path(row["rgba"]["path"]).resolve()) for row in ordered]
    masks = [str(Path(row["mask"]["path"]).resolve()) for row in ordered]
    selected = next(
        row
        for row in ordered
        if int(row["view_index"]) == int(protocol["single_view_index"])
    )
    return {
        "format": "pose_point_depth_mv.bunny_adapter_context.v1",
        "protocol": str(protocol_path),
        "protocol_sha256": protocol["protocol_sha256"],
        "method_id": method_id,
        "output_dir": str(output_dir),
        "input_dir": str(protocol_path.parent / "inputs"),
        "single_view_rgba": str(Path(selected["rgba"]["path"]).resolve()),
        "multiview_rgba": rgba,
        "multiview_masks": masks,
        "reference_mesh": str(Path(protocol["reference"]["mesh"]["path"]).resolve()),
        "camera_calibrated": False,
        "warning": (
            "Bunny thumbnails contain no intrinsics/extrinsics, sparse points, or TM2W. "
            "A model that requires those inputs must produce them explicitly and record "
            "their provenance; it must not read reference_mesh as hidden evidence."
        ),
    }


def placeholders(context_path: Path, context: dict[str, Any]) -> dict[str, str]:
    return {
        "protocol": context["protocol"],
        "protocol_sha256": context["protocol_sha256"],
        "method_id": context["method_id"],
        "output_dir": context["output_dir"],
        "input_dir": context["input_dir"],
        "context": str(context_path),
        "single_view_rgba": context["single_view_rgba"],
        "multiview_rgba_csv": ",".join(context["multiview_rgba"]),
        "reference_mesh": context["reference_mesh"],
    }


def expand(value: str, substitutions: dict[str, str]) -> str:
    try:
        return str(value).format_map(substitutions)
    except KeyError as exc:
        raise ValueError(f"unknown adapter placeholder in {value!r}: {exc}") from exc


def validate_adapter(payload: dict[str, Any]) -> None:
    if payload.get("format") != ADAPTER_FORMAT:
        raise ValueError(f"unsupported adapter format={payload.get('format')!r}")
    command = payload.get("command")
    if not isinstance(command, list) or not command or not all(
        isinstance(item, str) and item for item in command
    ):
        raise ValueError("adapter command must be a non-empty string list")
    if not str(payload.get("expected_mesh", "")):
        raise ValueError("adapter expected_mesh is required")
    for key in ("checkpoint_paths", "input_paths", "code_paths"):
        values = payload.get(key, [])
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f"adapter {key} must be a string list")


def run_command(args: argparse.Namespace) -> None:
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    output_dir = method_dir(protocol_path, args.method_id)
    result_path = output_dir / "result.json"
    if result_path.is_file():
        result = load_method_result(protocol_path, args.method_id)
        print(
            json.dumps(
                {
                    "status": "reused",
                    "method_id": args.method_id,
                    "mesh": result["mesh"]["path"],
                },
                indent=2,
            )
        )
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"partial trained-method output exists; preserve and inspect: {output_dir}"
        )
    adapter_path = args.adapter.resolve()
    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
    validate_adapter(adapter)
    if (
        bool(adapter.get("reference_mesh_declared_as_model_input", False))
        and not args.allow_reference_as_input
    ):
        raise RuntimeError(
            "adapter declares reference Mesh as an input; rerun with "
            "--allow_reference_as_input only for an explicitly labeled "
            "oracle/sensor simulation"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    context = adapter_context(protocol_path, args.method_id, output_dir)
    context_path = output_dir / "adapter_context.json"
    atomic_json(context_path, context)
    substitutions = placeholders(context_path, context)
    command = [expand(item, substitutions) for item in adapter["command"]]
    environment = os.environ.copy()
    for key, value in adapter.get("environment", {}).items():
        environment[str(key)] = expand(str(value), substitutions)
    log_path = output_dir / "adapter.log"
    print(f"[bunny_trained_adapter] command={json.dumps(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(Path(expand(adapter.get("cwd", str(Path.cwd())), substitutions))),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            return_code = process.wait()
        finally:
            process.stdout.close()
    if return_code != 0:
        raise RuntimeError(
            f"trained adapter failed rc={return_code}; partial output preserved: "
            f"{output_dir}"
        )
    mesh_path = safe_destination(
        output_dir,
        expand(str(adapter["expected_mesh"]), substitutions),
    )
    if not mesh_path.is_file():
        raise FileNotFoundError(
            f"adapter returned success but expected Mesh is missing: {mesh_path}"
        )
    checkpoint_bindings = {
        f"checkpoint_{index:02d}": binding(expand(path, substitutions))
        for index, path in enumerate(adapter.get("checkpoint_paths", []))
    }
    input_bindings = {
        f"input_{index:02d}": binding(expand(path, substitutions))
        for index, path in enumerate(adapter.get("input_paths", []))
    }
    external_code_bindings = {
        f"code_{index:02d}": binding(expand(path, substitutions))
        for index, path in enumerate(adapter.get("code_paths", []))
    }
    declared_inputs = adapter.get("input_view_indices", protocol["view_indices"])
    input_indices = [int(value) for value in declared_inputs]
    backend = {
        "kind": "replaceable_command_adapter",
        "adapter": binding(adapter_path),
        "adapter_id": str(adapter.get("adapter_id", args.method_id)),
        "command": command,
        "return_code": return_code,
        "log": binding(log_path),
        "checkpoint_bindings": checkpoint_bindings,
        "input_bindings": input_bindings,
        "external_code_bindings": external_code_bindings,
        "runner_code_bindings": code_bindings(
            {
                "trained_adapter": Path(__file__).resolve(),
                "common": Path(__file__).resolve().with_name("common.py"),
            }
        ),
        "input_contract": adapter.get("input_contract", {}),
        "reference_mesh_declared_as_model_input": bool(
            adapter.get("reference_mesh_declared_as_model_input", False)
        ),
    }
    result_path = write_method_result(
        protocol_path=protocol_path,
        method_id=args.method_id,
        display_name=args.display_name,
        mesh_path=mesh_path,
        input_view_indices=input_indices,
        backend=backend,
        notes=list(adapter.get("notes", [])),
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "method_id": args.method_id,
                "result": str(result_path),
                "mesh": str(mesh_path),
            },
            indent=2,
        )
    )


def register_mesh(args: argparse.Namespace) -> None:
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    output_dir = method_dir(protocol_path, args.method_id)
    result_path = output_dir / "result.json"
    if result_path.is_file():
        result = load_method_result(protocol_path, args.method_id)
        print(json.dumps({"status": "reused", "mesh": result["mesh"]["path"]}, indent=2))
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"partial method output exists: {output_dir}")
    source = args.mesh.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    suffix = source.suffix.lower()
    if suffix not in {".glb", ".gltf", ".obj", ".ply", ".stl"}:
        raise ValueError(f"unsupported registered Mesh suffix={suffix!r}")
    destination = output_dir / f"mesh_registered{suffix}"
    atomic_copy(source, destination)
    auxiliary: dict[str, Path] = {}
    if suffix == ".obj":
        source_mtl = source.with_suffix(".mtl")
        if source_mtl.is_file():
            copied_mtl = output_dir / source_mtl.name
            atomic_copy(source_mtl, copied_mtl)
            auxiliary["material"] = copied_mtl
    input_indices = (
        parse_int_csv(args.input_view_indices)
        if args.input_view_indices
        else [int(value) for value in protocol["view_indices"]]
    )
    checkpoints = {
        f"checkpoint_{position:02d}": binding(path)
        for position, path in enumerate(args.checkpoint)
    }
    code = {
        f"code_{position:02d}": binding(path)
        for position, path in enumerate(args.code)
    }
    result_path = write_method_result(
        protocol_path=protocol_path,
        method_id=args.method_id,
        display_name=args.display_name,
        mesh_path=destination,
        auxiliary_meshes=auxiliary,
        input_view_indices=input_indices,
        backend={
            "kind": "registered_external_mesh",
            "source_mesh": binding(source),
            "checkpoint_bindings": checkpoints,
            "external_code_bindings": code,
            "registration_runner": binding(Path(__file__).resolve()),
            "input_contract": json.loads(args.input_contract_json),
            "reference_mesh_declared_as_model_input": bool(args.reference_as_input),
        },
        notes=[
            "Mesh was produced by an external/current trained inference entrypoint "
            "and copied into this immutable review directory."
        ],
    )
    print(
        json.dumps(
            {
                "status": "registered",
                "result": str(result_path),
                "mesh": str(destination),
            },
            indent=2,
        )
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    command = subparsers.add_parser(
        "command",
        help="run a JSON argv adapter and validate its expected Mesh",
    )
    command.add_argument("--protocol", type=Path, required=True)
    command.add_argument("--adapter", type=Path, required=True)
    command.add_argument("--method_id", default="trained_full")
    command.add_argument("--display_name", default="Current trained Full")
    command.add_argument("--allow_reference_as_input", action="store_true")
    command.set_defaults(handler=run_command)

    register = subparsers.add_parser(
        "register",
        help="copy/register a Mesh produced by the current inference code",
    )
    register.add_argument("--protocol", type=Path, required=True)
    register.add_argument("--mesh", type=Path, required=True)
    register.add_argument("--method_id", default="trained_full")
    register.add_argument("--display_name", default="Current trained Full")
    register.add_argument("--input_view_indices", default="")
    register.add_argument("--checkpoint", type=Path, action="append", default=[])
    register.add_argument("--code", type=Path, action="append", default=[])
    register.add_argument("--input_contract_json", default="{}")
    register.add_argument("--reference_as_input", action="store_true")
    register.set_defaults(handler=register_mesh)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
