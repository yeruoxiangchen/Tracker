#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_seeds(text: str) -> list[int]:
    seeds = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError(f"expected unique training seeds, got {text!r}")
    return seeds


def load_report(directory: str) -> dict[str, Any]:
    path = Path(directory) / "report.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("format") != "reconvggt.sparse_anchor_ss_flow_eval.v1":
        raise RuntimeError(f"unsupported sparse-anchor report format: {path}")
    report["_report_path"] = str(path.resolve())
    return report


def metric(report: dict[str, Any], key: str, statistic: str) -> float:
    return float(report["summary"]["by_object"][key][statistic])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Require every SS16 sparse-anchor training seed to pass the same local gate."
    )
    parser.add_argument("--report_dirs", nargs="+", required=True)
    parser.add_argument("--expected_train_seeds", default="42,43,44")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    expected = parse_seeds(args.expected_train_seeds)
    reports = [load_report(item) for item in args.report_dirs]
    by_seed: dict[int, dict[str, Any]] = {}
    for report in reports:
        seed = int(report.get("checkpoint_train_seed", -1))
        if seed in by_seed:
            raise RuntimeError(f"duplicate training seed {seed}")
        stock = report["stock_equivalence"]
        decision = report["decision"]
        by_seed[seed] = {
            "report": report["_report_path"],
            "checkpoint_step": int(report.get("checkpoint_step", -1)),
            "passed": bool(decision.get("passed", False)),
            "stock_and_null_exact": bool(
                float(stock.get("disabled_max_abs_diff", float("inf"))) == 0.0
                and float(stock.get("null_max_abs_diff", float("inf"))) == 0.0
            ),
            "correct_stock_positive_mean": metric(
                report, "correct_minus_stock_positive_probability", "mean"
            ),
            "correct_stock_positive_median": metric(
                report, "correct_minus_stock_positive_probability", "median"
            ),
            "correct_stock_object_win": metric(
                report, "correct_minus_stock_positive_probability", "positive_rate"
            ),
            "correct_corrupted_positive_mean": metric(
                report, "correct_minus_corrupted_positive_probability", "mean"
            ),
            "correct_corrupted_positive_median": metric(
                report, "correct_minus_corrupted_positive_probability", "median"
            ),
            "correct_corrupted_object_win": metric(
                report, "correct_minus_corrupted_positive_probability", "positive_rate"
            ),
            "stock_correct_outside_mean": metric(
                report, "stock_minus_correct_outside_probability", "mean"
            ),
            "stock_correct_flow_mean": metric(
                report, "stock_minus_correct_flow", "mean"
            ),
            "neutral_velocity_mse": float(
                report["summary"]["correct_neutral_velocity_mse"]["mean"]
            ),
            "positive_t_correct_vs_stock": int(
                decision.get("positive_t_correct_vs_stock", 0)
            ),
            "positive_t_correct_vs_corrupted": int(
                decision.get("positive_t_correct_vs_corrupted", 0)
            ),
            "required_positive_t": int(decision.get("required_positive_t", 0)),
        }

    observed = sorted(by_seed)
    checks = {
        "expected_seeds_present": observed == sorted(expected),
        "all_seed_decisions_pass": all(
            by_seed.get(seed, {}).get("passed", False) for seed in expected
        ),
        "all_seed_stock_and_null_exact": all(
            by_seed.get(seed, {}).get("stock_and_null_exact", False)
            for seed in expected
        ),
        "all_seed_correct_beats_corrupted_mean": all(
            by_seed.get(seed, {}).get("correct_corrupted_positive_mean", float("-inf"))
            > 0.0
            for seed in expected
        ),
    }
    result = {
        "format": "reconvggt.sparse_anchor_ss_flow_multiseed.v1",
        "expected_train_seeds": expected,
        "observed_train_seeds": observed,
        "checks": checks,
        "by_seed": {str(seed): by_seed[seed] for seed in observed},
        "passed": all(checks.values()),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# SS16 Sparse-anchor Multi-seed Gate",
        "",
        f"- decision: `{'PASS' if result['passed'] else 'FAIL'}`",
        f"- expected seeds: `{expected}`",
        f"- observed seeds: `{observed}`",
        "",
        "| seed | pass | stock/null exact | correct-stock local | "
        "correct-corrupt local | correct-corrupt win | outside gain | flow gain |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed in observed:
        row = by_seed[seed]
        lines.append(
            f"| {seed} | {row['passed']} | {row['stock_and_null_exact']} | "
            f"{row['correct_stock_positive_mean']:.8f} | "
            f"{row['correct_corrupted_positive_mean']:.8f} | "
            f"{row['correct_corrupted_object_win']:.2%} | "
            f"{row['stock_correct_outside_mean']:.8f} | "
            f"{row['stock_correct_flow_mean']:.8f} |"
        )
    lines.extend(["", "## Checks", ""])
    for key, passed in checks.items():
        lines.append(f"- {key}: `{'PASS' if passed else 'FAIL'}`")
    (output_dir / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
