#!/usr/bin/env python3

"""Evaluate a generated mesh against the source Objaverse GLB with normalized surface metrics."""

from __future__ import annotations

import argparse
import json

from common import evaluate_mesh_quality, evaluate_mesh_quality_against_points, load_sparse_target_points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_mesh", required=True)
    parser.add_argument("--gt_mesh", default=None)
    parser.add_argument("--gt_points_npz", default=None)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--icp_iters", type=int, default=12)
    args = parser.parse_args()

    if args.gt_points_npz:
        report = evaluate_mesh_quality_against_points(
            pred_mesh_path=args.pred_mesh,
            gt_points=load_sparse_target_points(args.gt_points_npz),
            output_json=args.output_json,
            samples=args.samples,
            seed=args.seed,
            icp_iters=args.icp_iters,
        )
    else:
        if not args.gt_mesh:
            raise ValueError("Either --gt_mesh or --gt_points_npz is required")
        report = evaluate_mesh_quality(
            pred_mesh_path=args.pred_mesh,
            gt_mesh_path=args.gt_mesh,
            output_json=args.output_json,
            samples=args.samples,
            seed=args.seed,
            icp_iters=args.icp_iters,
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
