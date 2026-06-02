#!/usr/bin/env python3
"""OpenXLab helper for OmniObject3D downloads.

Official dataset repo:
    omniobject3d/OmniObject3D-New

Use `list` first to inspect available paths, then use `download --source-path`
for a subset.  `get` downloads the whole compressed dataset and is about 1.2TB.
"""

from __future__ import annotations

import argparse
from pathlib import Path


OMNI_REPO = "omniobject3d/OmniObject3D-New"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download/list OmniObject3D via OpenXLab.")
    parser.add_argument(
        "action",
        choices=("info", "list", "download", "get"),
        help="info/list inspect the repository; download gets one path; get downloads all.",
    )
    parser.add_argument(
        "--dataset-repo",
        default=OMNI_REPO,
        help=f"OpenXLab dataset repo. Default: {OMNI_REPO}",
    )
    parser.add_argument(
        "--source-path",
        default="/raw/point_clouds/ply_files",
        help="Remote file/folder path for action=download.",
    )
    parser.add_argument(
        "--target-path",
        default="/data/OmniObject3D/raw",
        help="Local target directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_path = Path(args.target_path).expanduser().resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    if args.action == "info":
        from openxlab.dataset.handler.info_dataset_repository import info

        info(args.dataset_repo)
    elif args.action == "list":
        from rich import print as rprint
        from openxlab.dataset.handler import list_dataset_repository

        # openxlab 0.1.2 has a missing rprint import in this module.
        list_dataset_repository.rprint = rprint
        list_dataset_repository.query(args.dataset_repo)
    elif args.action == "download":
        from openxlab.dataset.handler.download_dataset_repository import download

        download(
            dataset_repo=args.dataset_repo,
            source_path=args.source_path,
            target_path=str(target_path),
        )
    elif args.action == "get":
        from openxlab.dataset.handler.get_dataset_repository import get

        get(dataset_repo=args.dataset_repo, target_path=str(target_path))
    else:
        raise ValueError(f"Unsupported action: {args.action}")


if __name__ == "__main__":
    main()
