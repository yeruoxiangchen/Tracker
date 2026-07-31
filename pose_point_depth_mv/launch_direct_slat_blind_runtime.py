#!/usr/bin/env python3
"""Launch the blind exporter with the runtime frozen in its protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--blind_key_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_renders", action="store_true")
    return parser.parse_args()


def canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    args = parse_args()
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    body = dict(protocol)
    saved = str(body.pop("protocol_sha256", ""))
    if canonical_sha256(body) != saved:
        raise RuntimeError("protocol canonical SHA-256 mismatch")
    runtime = dict(protocol["runtime"])
    os.environ["ATTN_BACKEND"] = str(runtime["attention_backend"])
    os.environ["SPARSE_ATTN_BACKEND"] = str(runtime["sparse_attention_backend"])
    os.environ["SPCONV_ALGO"] = str(runtime["spconv_algo"])
    cublas = str(runtime.get("cublas_workspace_config", ""))
    if cublas:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas
    else:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    command = [
        sys.executable,
        "-u",
        "-m",
        "pose_point_depth_mv.export_direct_slat_blind_holdout",
        "--protocol",
        str(protocol_path),
        "--blind_key_file",
        str(Path(args.blind_key_file).resolve()),
        "--output_dir",
        str(Path(args.output_dir).resolve()),
        "--device",
        str(args.device),
    ]
    if args.resume:
        command.append("--resume")
    if args.skip_renders:
        command.append("--skip_renders")
    os.execvpe(command[0], command, os.environ.copy())


if __name__ == "__main__":
    main()
