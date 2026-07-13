#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

TRACKER_ROOT = Path(__file__).resolve().parents[1]
if str(TRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKER_ROOT))

from trellis_point_prior_mv.sparse_coord_tools import (  # noqa: E402
    coords_with_batch,
    coords_xyz,
    sparse_diagnostic_metrics,
)


def _resolve_manifest_relative(path: str, roots: list[Path]) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    for root in roots:
        resolved = root / candidate
        if resolved.exists():
            return resolved
    return roots[0] / candidate


def _load_prior_manifest_sample(manifest_path: Path, uid: str = "") -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = payload.get("samples") or []
    if not samples:
        raise ValueError(f"No samples found in prior manifest: {manifest_path}")
    if uid:
        matches = [sample for sample in samples if str(sample.get("uid", "")) == str(uid)]
        if not matches:
            raise ValueError(f"uid={uid!r} not found in prior manifest {manifest_path}")
        sample = matches[0]
    else:
        sample = samples[0]

    prior_root = Path(payload.get("prior_root") or manifest_path.parent)
    prior_npz = _resolve_manifest_relative(
        str(sample.get("prior_npz", "")),
        [prior_root, manifest_path.parent, Path(str(sample.get("dataset_root", ".")))],
    )
    if not prior_npz.exists():
        raise FileNotFoundError(f"Prior npz not found: {prior_npz}")

    prior_payload = np.load(prior_npz)
    if "prior_coords" in prior_payload:
        prior_coords = prior_payload["prior_coords"]
    elif "coords" in prior_payload:
        prior_coords = prior_payload["coords"]
    else:
        raise KeyError(f"No prior_coords/coords key in {prior_npz}")

    summary = {
        "manifest": str(manifest_path),
        "uid": str(sample.get("uid", "")),
        "prior_npz": str(prior_npz),
        "prior_coord_count": int(coords_xyz(prior_coords).shape[0]),
        "sample_dataset_root": str(sample.get("dataset_root", "")),
        "sample_sparse_subdir": str(sample.get("sparse_subdir", "")),
        "sample_frame_count": int(len(sample.get("frames") or [])),
    }
    return sample, np.asarray(prior_coords, dtype=np.int32), summary


def _load_coords(run_dir: Path) -> np.ndarray:
    path = run_dir / "coords.npz"
    if not path.exists():
        raise FileNotFoundError(f"coords.npz not found: {path}")
    payload = np.load(path)
    if "coords" not in payload:
        raise KeyError(f"No coords key in {path}")
    return np.asarray(payload["coords"], dtype=np.int32)


def _xyz_set(coords: np.ndarray) -> set[tuple[int, int, int]]:
    xyz = coords_xyz(coords)
    return {tuple(int(v) for v in row) for row in xyz.reshape(-1, 3)}


def _coords_from_set(points: set[tuple[int, int, int]]) -> np.ndarray:
    if not points:
        return coords_with_batch(np.zeros((0, 3), dtype=np.int32))
    arr = np.asarray(sorted(points), dtype=np.int32)
    return coords_with_batch(arr)


def _set_compare(base: set[tuple[int, int, int]], adapter: set[tuple[int, int, int]]) -> dict[str, float | int]:
    intersection = len(base & adapter)
    union = len(base | adapter)
    return {
        "baseline_count": int(len(base)),
        "adapter_count": int(len(adapter)),
        "kept_count": int(intersection),
        "union_count": int(union),
        "removed_count": int(len(base - adapter)),
        "added_count": int(len(adapter - base)),
        "iou": float(intersection / max(1, union)),
        "baseline_keep_ratio": float(intersection / max(1, len(base))),
        "adapter_keep_ratio": float(intersection / max(1, len(adapter))),
        "added_ratio_of_adapter": float(len(adapter - base) / max(1, len(adapter))),
        "removed_ratio_of_baseline": float(len(base - adapter) / max(1, len(base))),
    }


def _metric(metrics: dict[str, Any], prefix: str, suffix: str) -> float | None:
    value = metrics.get(f"{prefix}_{suffix}")
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(a - b)


def _direction_summary(report: dict[str, Any]) -> dict[str, float | None]:
    added = report["subset_metrics"]["added"]
    removed = report["subset_metrics"]["removed"]
    kept = report["subset_metrics"]["kept"]
    return {
        "added_minus_removed_within_prior_radius_ratio": _diff(
            _metric(added, "added", "within_prior_radius_ratio"),
            _metric(removed, "removed", "within_prior_radius_ratio"),
        ),
        "added_minus_removed_prior_distance_mean": _diff(
            _metric(added, "added", "prior_distance_mean"),
            _metric(removed, "removed", "prior_distance_mean"),
        ),
        "added_minus_removed_projection_any_mask_hit_ratio": _diff(
            _metric(added, "added", "projection_any_mask_hit_ratio"),
            _metric(removed, "removed", "projection_any_mask_hit_ratio"),
        ),
        "added_minus_removed_projection_keep_ratio": _diff(
            _metric(added, "added", "projection_keep_ratio"),
            _metric(removed, "removed", "projection_keep_ratio"),
        ),
        "added_minus_removed_visible_outside_mask_event_ratio": _diff(
            _metric(added, "added", "visible_outside_mask_event_ratio"),
            _metric(removed, "removed", "visible_outside_mask_event_ratio"),
        ),
        "adapter_minus_baseline_within_prior_radius_ratio": _diff(
            _metric(report["subset_metrics"]["adapter"], "adapter", "within_prior_radius_ratio"),
            _metric(report["subset_metrics"]["baseline"], "baseline", "within_prior_radius_ratio"),
        ),
        "adapter_minus_baseline_projection_any_mask_hit_ratio": _diff(
            _metric(report["subset_metrics"]["adapter"], "adapter", "projection_any_mask_hit_ratio"),
            _metric(report["subset_metrics"]["baseline"], "baseline", "projection_any_mask_hit_ratio"),
        ),
        "adapter_minus_baseline_visible_outside_mask_event_ratio": _diff(
            _metric(report["subset_metrics"]["adapter"], "adapter", "visible_outside_mask_event_ratio"),
            _metric(report["subset_metrics"]["baseline"], "baseline", "visible_outside_mask_event_ratio"),
        ),
        "kept_within_prior_radius_ratio": _metric(kept, "kept", "within_prior_radius_ratio"),
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    cmp_row = report["set_compare"]
    direction = report["direction_summary"]
    lines = [
        "# B4.0 Delta Prior Alignment",
        "",
        "## Inputs",
        "",
        f"- baseline_dir: `{report['baseline_dir']}`",
        f"- adapter_dir: `{report['adapter_dir']}`",
        f"- prior_manifest: `{report['prior_summary']['manifest']}`",
        f"- uid: `{report['prior_summary']['uid']}`",
        "",
        "## Set Compare",
        "",
        "```text",
    ]
    for key, value in cmp_row.items():
        lines.append(f"{key}: {value}")
    lines.extend(["```", "", "## Direction Summary", "", "```text"])
    for key, value in direction.items():
        lines.append(f"{key}: {value}")
    lines.extend(
        [
            "```",
            "",
            "## Interpretation",
            "",
            "```text",
            "Good direction requires:",
            "  added > removed on within_prior_radius_ratio / projection mask hit",
            "  added < removed on prior_distance_mean / visible_outside_mask_event_ratio",
            "  adapter - baseline does not increase outside-mask ratio",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="B4.0 baseline-vs-adapter delta prior alignment diagnostic.")
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prior_manifest", required=True)
    parser.add_argument("--prior_uid", default="")
    parser.add_argument("--prior_radius", type=float, default=4.0)
    parser.add_argument("--projection_min_support_views", type=int, default=1)
    parser.add_argument("--projection_min_support_ratio", type=float, default=0.0)
    parser.add_argument("--visual_hull_min_visible_views", type=int, default=1)
    parser.add_argument("--visual_hull_min_support_ratio", type=float, default=0.0)
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--save_subset_coords", action="store_true")
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    adapter_dir = Path(args.adapter_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_coords = _load_coords(baseline_dir)
    adapter_coords = _load_coords(adapter_dir)
    sample, prior_coords, prior_summary = _load_prior_manifest_sample(Path(args.prior_manifest), args.prior_uid)

    baseline_set = _xyz_set(baseline_coords)
    adapter_set = _xyz_set(adapter_coords)
    subsets = {
        "baseline": _coords_from_set(baseline_set),
        "adapter": _coords_from_set(adapter_set),
        "added": _coords_from_set(adapter_set - baseline_set),
        "removed": _coords_from_set(baseline_set - adapter_set),
        "kept": _coords_from_set(adapter_set & baseline_set),
        "union": _coords_from_set(adapter_set | baseline_set),
    }

    subset_metrics: dict[str, dict[str, Any]] = {}
    for name, coords in subsets.items():
        subset_metrics[name] = sparse_diagnostic_metrics(
            name,
            coords,
            prior_coords,
            sample,
            prior_radius=float(args.prior_radius),
            min_support_views=int(args.projection_min_support_views),
            min_support_ratio=float(args.projection_min_support_ratio),
            visual_hull_min_visible_views=int(args.visual_hull_min_visible_views),
            visual_hull_min_support_ratio=float(args.visual_hull_min_support_ratio),
            grid_resolution=64,
            mask_threshold=int(args.mask_threshold),
        )

    report: dict[str, Any] = {
        "args": vars(args),
        "baseline_dir": str(baseline_dir),
        "adapter_dir": str(adapter_dir),
        "prior_summary": prior_summary,
        "set_compare": _set_compare(baseline_set, adapter_set),
        "subset_metrics": subset_metrics,
    }
    report["direction_summary"] = _direction_summary(report)

    if args.save_subset_coords:
        for name, coords in subsets.items():
            np.savez_compressed(output_dir / f"{name}_coords.npz", coords=coords)

    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(output_dir / "report.md", report)
    print(f"[B4.0] wrote {output_dir / 'report.json'}", flush=True)
    print(f"[B4.0] wrote {output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
