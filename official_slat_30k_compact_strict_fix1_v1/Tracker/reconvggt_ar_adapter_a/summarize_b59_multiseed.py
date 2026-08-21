#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_report(path: Path) -> dict[str, Any]:
    report_path = path if path.name == "report.json" else path / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["_report_path"] = str(report_path)
    return report


def _write_md(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# B5.9 Multi-Seed Candidate Summary",
        "",
        f"- reports: `{report['report_count']}`",
        f"- seeds: `{report['seeds']}`",
        f"- invariant configuration: `{report['configuration_invariant_passed']}`",
        f"- all-seed passing candidates: `{report['passing_candidates']}`",
        "",
        "## Candidates",
        "",
        "```text",
    ]
    for row in report["candidates"]:
        lines.append(
            f"{row['candidate']}: all_seeds={row['all_seeds_passed']} "
            f"seed_pass={row['passing_seed_count']}/{row['expected_seed_count']} "
            f"sign_consistent={row['per_session_sign_consistent']}"
        )
        for seed_row in row["per_seed"]:
            lines.append(
                f"  seed={seed_row['seed']} all_sessions={seed_row['all_sessions_passed']} "
                f"strict={seed_row['strict_pass_session_count']}/{seed_row['session_count']}"
            )
    lines.extend(["```", "", "## Judgment", "", report["judgment"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate B5.9 candidates across sparse sampling seeds.")
    parser.add_argument("--report_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_seeds", default="42,43,44")
    args = parser.parse_args()

    expected_seeds = sorted(int(x.strip()) for x in args.expected_seeds.split(",") if x.strip())
    reports = [_load_report(Path(item)) for item in args.report_dirs]
    reports_by_seed: dict[int, dict[str, Any]] = {}
    for report in reports:
        seed = int(report["args"]["seed"])
        if seed in reports_by_seed:
            raise ValueError(f"Duplicate seed {seed}: {report['_report_path']}")
        reports_by_seed[seed] = report
    missing_seeds = sorted(set(expected_seeds) - set(reports_by_seed))
    extra_seeds = sorted(set(reports_by_seed) - set(expected_seeds))

    invariant_keys = [
        "physical_frame_scope",
        "evaluation_frame_scope",
        "candidate_gate_normalization",
        "candidate_formulas",
        "candidate_scales",
        "basis_modes",
        "positive_min_visible_views",
        "positive_min_support_ratio",
        "negative_min_visible_views",
        "negative_max_support_ratio",
        "negative_min_outside_ratio",
        "negative_prior_radius_multiplier",
        "max_views",
        "ss_steps",
        "ss_cfg_strength",
        "ss_guidance_rescale",
        "ss_rescale_t",
        "candidate_min_changed_count",
        "candidate_min_changed_ratio",
        "candidate_max_changed_ratio",
        "candidate_min_set_iou",
        "candidate_require_absolute_outside_nonincrease",
        "candidate_max_component_increase",
        "candidate_min_coord_count_ratio",
        "candidate_max_coord_count_ratio",
    ]
    invariant_values = {
        key: sorted({json.dumps(report["args"].get(key), sort_keys=True) for report in reports})
        for key in invariant_keys
    }
    invariant_passed = all(len(values) == 1 for values in invariant_values.values())

    aggregate_by_seed = {
        seed: {row["candidate"]: row for row in report.get("aggregate", [])}
        for seed, report in reports_by_seed.items()
    }
    candidate_names = sorted(
        set.intersection(*(set(rows) for rows in aggregate_by_seed.values()))
        if aggregate_by_seed
        else set()
    )
    candidate_rows = []
    for candidate in candidate_names:
        per_seed = []
        per_session_signs: dict[str, list[tuple[bool, bool, bool]]] = {}
        for seed in expected_seeds:
            row = aggregate_by_seed.get(seed, {}).get(candidate)
            if row is None:
                per_seed.append(
                    {
                        "seed": seed,
                        "all_sessions_passed": False,
                        "strict_pass_session_count": 0,
                        "session_count": 0,
                        "missing": True,
                    }
                )
                continue
            per_seed.append(
                {
                    "seed": seed,
                    "all_sessions_passed": bool(row.get("all_sessions_passed")),
                    "strict_pass_session_count": int(row.get("strict_pass_session_count") or 0),
                    "session_count": int(row.get("session_count") or 0),
                    "missing": False,
                }
            )
            for session_row in row.get("per_session", []):
                per_session_signs.setdefault(str(session_row["session"]), []).append(
                    (
                        session_row.get("mask") is not None and float(session_row["mask"]) >= 0.0,
                        session_row.get("outside") is not None and float(session_row["outside"]) <= 0.0,
                        session_row.get("prior") is not None and float(session_row["prior"]) >= 0.0,
                    )
                )
        sign_consistency = {
            session: len(signs) == len(expected_seeds) and all(all(sign) for sign in signs)
            for session, signs in per_session_signs.items()
        }
        passing_seed_count = sum(bool(row["all_sessions_passed"]) for row in per_seed)
        all_seeds_passed = bool(
            invariant_passed
            and not missing_seeds
            and len(per_seed) == len(expected_seeds)
            and passing_seed_count == len(expected_seeds)
            and sign_consistency
            and all(sign_consistency.values())
        )
        candidate_rows.append(
            {
                "candidate": candidate,
                "expected_seed_count": len(expected_seeds),
                "passing_seed_count": passing_seed_count,
                "per_session_sign_consistent": sign_consistency,
                "all_seeds_passed": all_seeds_passed,
                "per_seed": per_seed,
            }
        )
    candidate_rows.sort(
        key=lambda row: (bool(row["all_seeds_passed"]), int(row["passing_seed_count"])),
        reverse=True,
    )
    passing = [row["candidate"] for row in candidate_rows if row["all_seeds_passed"]]
    if missing_seeds or extra_seeds:
        judgment = f"INCOMPLETE: missing_seeds={missing_seeds}, extra_seeds={extra_seeds}. Do not enter B5.10."
    elif not invariant_passed:
        judgment = "INVALID COMPARISON: candidate reports differ in non-seed configuration. Do not enter B5.10."
    elif passing:
        judgment = "PASS: at least one identical candidate passes every train/validation session for every expected seed. It is eligible for the flow-gradient audit."
    else:
        judgment = "FAIL: no identical candidate passes all sessions and all seeds. Do not enter B5.10."
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "args": vars(args),
        "report_count": len(reports),
        "report_paths": [report["_report_path"] for report in reports],
        "seeds": sorted(reports_by_seed),
        "expected_seeds": expected_seeds,
        "missing_seeds": missing_seeds,
        "extra_seeds": extra_seeds,
        "configuration_invariant_passed": invariant_passed,
        "configuration_invariants": invariant_values,
        "candidates": candidate_rows,
        "passing_candidates": passing,
        "judgment": judgment,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_md(output_dir / "report.md", report)
    print(f"[B5.9 multiseed] passing={passing} wrote {output_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
