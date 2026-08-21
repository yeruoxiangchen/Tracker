#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_report(directory: str) -> dict[str, Any]:
    path = Path(directory) / "report.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    report["_report_path"] = str(path.resolve())
    return report


def parse_seeds(text: str) -> list[int]:
    seeds = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError(f"expected unique training seeds, got {text!r}")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Require every J1a.2-A trained seed to pass the same specificity gate."
    )
    parser.add_argument("--report_dirs", nargs="+", required=True)
    parser.add_argument("--expected_train_seeds", default="42,43,44")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--experiment_label", default="J1a.2-A")
    parser.add_argument("--fail_on_decision", action="store_true")
    args = parser.parse_args()

    expected = parse_seeds(args.expected_train_seeds)
    reports = [load_report(item) for item in args.report_dirs]
    by_seed: dict[int, dict[str, Any]] = {}
    for report in reports:
        seed = int(report.get("checkpoint_train_seed", -1))
        if seed in by_seed:
            raise RuntimeError(f"duplicate training seed {seed}")
        decision = report.get("decision", {})
        stock = report.get("stock_equivalence", {})
        by_seed[seed] = {
            "report": report["_report_path"],
            "passed": bool(decision.get("passed", False)),
            "stock_equivalence_passed": bool(
                stock.get("condition_exact", False)
                and stock.get("velocity_exact", False)
                and stock.get("null_present_condition_exact", False)
                and stock.get("null_present_velocity_exact", False)
            ),
            "physical_specificity_ratio": float(
                decision.get("physical_specificity_ratio", float("nan"))
            ),
            "signed_physical_specificity_ratio": float(
                decision.get("signed_physical_specificity_ratio", float("nan"))
            ),
            "alignment_probability_gap": float(
                decision.get("alignment_probability_gap", float("nan"))
            ),
            "correct_shuffled_object_win_rate": float(
                report.get("summary", {})
                .get("by_object", {})
                .get("shuffled_minus_correct", {})
                .get("positive_rate", float("nan"))
            ),
        }
    observed = sorted(by_seed)
    checks = {
        "expected_seeds_present": observed == sorted(expected),
        "all_seed_decisions_pass": all(
            by_seed.get(seed, {}).get("passed", False) for seed in expected
        ),
        "all_seed_stock_equivalence_pass": all(
            by_seed.get(seed, {}).get("stock_equivalence_passed", False)
            for seed in expected
        ),
    }
    result = {
        "format": "reconvggt.j1a2a.multiseed_summary.v1",
        "experiment_label": str(args.experiment_label),
        "expected_train_seeds": expected,
        "observed_train_seeds": observed,
        "checks": checks,
        "by_seed": {str(seed): by_seed[seed] for seed in observed},
        "passed": all(checks.values()),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        f"# {args.experiment_label} Multi-seed Gate",
        "",
        f"- decision: `{'PASS' if result['passed'] else 'FAIL'}`",
        f"- expected seeds: `{expected}`",
        f"- observed seeds: `{observed}`",
        "",
        "| seed | pass | stock exact | specificity | signed specificity | "
        "alignment gap | object win rate |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for seed in observed:
        row = by_seed[seed]
        lines.append(
            f"| {seed} | {row['passed']} | {row['stock_equivalence_passed']} | "
            f"{row['physical_specificity_ratio']:.6f} | "
            f"{row['signed_physical_specificity_ratio']:.6f} | "
            f"{row['alignment_probability_gap']:.6f} | "
            f"{row['correct_shuffled_object_win_rate']:.2%} |"
        )
    (output_dir / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if args.fail_on_decision and not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
