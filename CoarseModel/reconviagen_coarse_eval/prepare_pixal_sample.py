#!/usr/bin/env python3

"""Convert one Pixal3D multiview manifest sample into a CoarseModel-style dataset."""

from __future__ import annotations

import argparse
import json

from common import DEFAULT_OUTPUT_ROOT, prepare_coarse_dataset_from_pixal_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--case_name", default=None)
    parser.add_argument("--case_prefix", default="pixalv9")
    parser.add_argument("--max_frames", type=int, default=8)
    args = parser.parse_args()

    report = prepare_coarse_dataset_from_pixal_sample(
        manifest_path=args.manifest,
        sample_index=args.sample_index,
        output_root=args.output_root,
        case_name=args.case_name,
        max_frames=args.max_frames,
        case_prefix=args.case_prefix,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

