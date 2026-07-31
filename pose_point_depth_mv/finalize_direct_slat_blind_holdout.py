#!/usr/bin/env python3
"""Freeze ratings, unblind once, and decide the formal Direct-SLAT holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from pose_point_depth_mv.direct_slat_blind import (
    FINAL_REPORT_FORMAT,
    PROTOCOL_FORMAT,
    RATINGS_FREEZE_FORMAT,
    SEALED_REPORT_FORMAT,
    aggregate_ratings,
    aggregate_unblinded,
    atomic_json,
    blind_pair_id,
    canonical_sha256,
    pair_identity,
    read_and_validate_rater_csv,
    sha256_file,
    validate_binding_tree,
    validate_execution_compatibility_record,
)
from pose_point_depth_mv.freeze_direct_slat_ratings import (
    validate_archive,
    validate_public_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--blind_key_file", required=True)
    parser.add_argument("--blind_output_dir", required=True)
    parser.add_argument("--ratings_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("format") != PROTOCOL_FORMAT:
        raise ValueError(f"unexpected protocol format={protocol.get('format')!r}")
    body = dict(protocol)
    saved = str(body.pop("protocol_sha256", ""))
    if canonical_sha256(body) != saved:
        raise RuntimeError("protocol canonical SHA-256 mismatch")
    validate_binding_tree(protocol["bindings"], "bindings")
    validate_binding_tree(protocol["sample_bindings"], "sample_bindings")
    validate_binding_tree(protocol["runtime_bindings"], "runtime_bindings")
    validate_binding_tree(protocol["code_bindings"], "code_bindings")
    validate_execution_compatibility_record(protocol)
    if protocol.get("formal") is not True or protocol.get("mode") != "confirmatory":
        raise RuntimeError("only a confirmatory protocol can be formally unblinded")
    return protocol


def validate_completion(root: Path) -> dict[str, Any]:
    path = root / "completion_manifest.json"
    completion = json.loads(path.read_text(encoding="utf-8"))
    if (
        completion.get("complete") is not True
        or completion.get("formal") is not True
        or completion.get("mode") != "confirmatory"
        or completion.get("science_decision_emitted") is not False
    ):
        raise RuntimeError("blind output is not a completed sealed confirmatory run")
    for row in completion.get("files", []):
        artifact = root / str(row["path"])
        if (
            not artifact.is_file()
            or sha256_file(artifact) != str(row["sha256"])
        ):
            raise RuntimeError(f"sealed blind output changed: {artifact}")
    sealed_path = root / str(completion["sealed_report"])
    if sha256_file(sealed_path) != str(completion["sealed_report_sha256"]):
        raise RuntimeError("sealed metric report hash differs from completion")
    return completion


def load_frozen_ratings(
    path: Path,
    *,
    protocol: dict[str, Any],
    completion_sha256: str,
    expected_pair_ids: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    freeze = json.loads(path.read_text(encoding="utf-8"))
    body = dict(freeze)
    saved = str(body.pop("ratings_freeze_sha256", ""))
    if (
        freeze.get("format") != RATINGS_FREEZE_FORMAT
        or freeze.get("complete") is not True
        or freeze.get("blind_key_read") is not False
        or canonical_sha256(body) != saved
        or freeze.get("protocol_sha256") != protocol["protocol_sha256"]
        or freeze.get("formal_completion_sha256") != completion_sha256
        or int(freeze.get("expected_pair_count", -1)) != len(expected_pair_ids)
        or freeze.get("expected_pair_ids_sha256")
        != canonical_sha256(sorted(expected_pair_ids))
    ):
        raise RuntimeError("ratings freeze manifest is invalid or mismatched")

    public_binding = freeze.get("public_bundle", {})
    public_manifest_path = Path(str(public_binding.get("manifest_path", ""))).resolve()
    if (
        not public_manifest_path.is_file()
        or sha256_file(public_manifest_path)
        != str(public_binding.get("manifest_sha256", ""))
    ):
        raise RuntimeError("frozen public bundle manifest changed")
    public_manifest, public_pair_ids = validate_public_bundle(public_manifest_path)
    if (
        public_manifest.get("public_bundle_sha256")
        != public_binding.get("public_bundle_sha256")
        or public_manifest.get("protocol_sha256") != protocol["protocol_sha256"]
        or public_manifest.get("formal_completion_sha256") != completion_sha256
        or set(public_pair_ids) != set(expected_pair_ids)
    ):
        raise RuntimeError("frozen public bundle identity or pair coverage differs")

    archive_binding = freeze.get("public_archive", {})
    archive_manifest_path = Path(
        str(archive_binding.get("manifest_path", ""))
    ).resolve()
    if (
        not archive_manifest_path.is_file()
        or sha256_file(archive_manifest_path)
        != str(archive_binding.get("manifest_sha256", ""))
    ):
        raise RuntimeError("frozen public archive manifest changed")
    archive_manifest, archive = validate_archive(
        archive_manifest_path,
        public_manifest=public_manifest,
        public_manifest_path=public_manifest_path,
    )
    if (
        str(archive.resolve()) != str(
            Path(str(archive_binding.get("archive_path", ""))).resolve()
        )
        or archive_manifest.get("archive_sha256")
        != archive_binding.get("archive_sha256")
    ):
        raise RuntimeError("frozen public archive identity differs")

    rating_bindings = list(freeze.get("ratings", []))
    if len(rating_bindings) < int(
        protocol["blinding"]["minimum_independent_raters"]
    ):
        raise RuntimeError("too few frozen independent rater score files")
    rater_files = []
    for binding in rating_bindings:
        score_path = Path(str(binding.get("path", ""))).resolve()
        if (
            not score_path.is_file()
            or sha256_file(score_path) != str(binding.get("sha256", ""))
        ):
            raise RuntimeError(f"frozen rater score file changed: {score_path}")
        parsed = read_and_validate_rater_csv(
            score_path, expected_pair_ids=expected_pair_ids
        )
        if parsed["rater_id"] != str(binding.get("rater_id", "")):
            raise RuntimeError(f"frozen rater identity changed: {score_path}")
        rater_files.append(parsed)
    rater_ids = [str(row["rater_id"]) for row in rater_files]
    if len(rater_ids) != len(set(rater_ids)):
        raise RuntimeError("frozen rater identities are not unique")
    return freeze, rater_files


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    automatic = report["automatic"]
    ratings = report["blind_ratings"]
    lines = [
        "# Direct-SLAT end-to-end blind holdout",
        "",
        f"- Final decision: `{'PASS' if report['passed'] else 'FAIL'}`",
        f"- Automatic Mesh gates: `{'PASS' if automatic['automatic_passed'] else 'FAIL'}`",
        f"- Blind review gates: `{'PASS' if ratings['passed'] else 'FAIL'}`",
        f"- Objects: `{len(automatic['object_rows'])}`",
        f"- Valid pairs: `{automatic['valid_pair_count']}`",
        f"- Raters: `{ratings['rater_count']}`",
        "",
        "## Automatic checks",
        "",
    ]
    lines.extend(
        f"- `{name}`: `{value}`"
        for name, value in automatic["checks"].items()
    )
    lines.extend(["", "## Blind checks", ""])
    lines.extend(
        f"- `{name}`: `{value}`"
        for name, value in ratings["checks"].items()
    )
    lines.extend(
        [
            "",
            "## Primary metrics",
            "",
            "```json",
            json.dumps(
                {
                    "chamfer_l1_improvement": automatic["summary"][
                        "chamfer_l1_improvement"
                    ],
                    "fscore_0p02_delta": automatic["summary"][
                        "fscore_0p02_delta"
                    ],
                    "largest_component_ratio_delta": automatic["summary"][
                        "largest_component_ratio_delta"
                    ],
                    "boundary_edge_count_delta": automatic["summary"][
                        "boundary_edge_count_delta"
                    ],
                    "boundary_total_length_delta": automatic["summary"][
                        "boundary_total_length_delta"
                    ],
                    "nonmanifold_edge_count_delta": automatic["summary"][
                        "nonmanifold_edge_count_delta"
                    ],
                    "connected_component_count_delta": automatic["summary"][
                        "connected_component_count_delta"
                    ],
                    "watertight_rate_delta": automatic["summary"][
                        "watertight_rate_delta"
                    ],
                    "zero_boundary_rate_delta": automatic["summary"][
                        "zero_boundary_rate_delta"
                    ],
                    "nonmanifold_free_rate_delta": automatic["summary"][
                        "nonmanifold_free_rate_delta"
                    ],
                    "blind_overall_score_delta": ratings["summary"][
                        "overall_score_delta"
                    ],
                    "blind_main_structure_delta": ratings["summary"][
                        "main_structure_delta"
                    ],
                    "blind_overall_preference_delta": ratings["summary"][
                        "overall_preference_delta"
                    ],
                },
                indent=2,
            ),
            "```",
            "",
            report["claim_scope"],
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    protocol_path = Path(args.protocol).resolve()
    protocol = load_protocol(protocol_path)
    blind_root = Path(args.blind_output_dir).resolve()
    completion = validate_completion(blind_root)
    if completion.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise RuntimeError("blind output and protocol identities differ")
    sealed_path = blind_root / str(completion["sealed_report"])
    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    if (
        sealed.get("format") != SEALED_REPORT_FORMAT
        or sealed.get("complete") is not True
        or sealed.get("formal") is not True
        or sealed.get("protocol_sha256") != protocol["protocol_sha256"]
    ):
        raise RuntimeError("sealed metrics are not a completed formal report")

    expected_pair_ids = []
    for frozen in protocol["selection"]["rows"]:
        for seed in protocol["sampling"]["joint_seeds"]:
            pair_id = blind_pair_id(
                protocol["protocol_name"],
                str(frozen["uid"]),
                int(seed),
            )
            if pair_id in expected_pair_ids:
                raise RuntimeError("protocol generated a duplicate blind pair ID")
            expected_pair_ids.append(pair_id)
    record_pair_ids = {
        str(row.get("pair_id", "")) for row in sealed.get("records", [])
    }
    if record_pair_ids != set(expected_pair_ids):
        raise RuntimeError("sealed metrics do not cover exactly the frozen pairs")

    ratings_manifest_path = Path(args.ratings_manifest).resolve()
    ratings_freeze, rater_files = load_frozen_ratings(
        ratings_manifest_path,
        protocol=protocol,
        completion_sha256=sha256_file(
            blind_root / "completion_manifest.json"
        ),
        expected_pair_ids=expected_pair_ids,
    )

    # The blind key is intentionally not opened until all public and rating
    # artifacts have been validated against their frozen hashes above.
    key_path = Path(args.blind_key_file).resolve()
    blind_key = bytes.fromhex(key_path.read_text(encoding="ascii").strip())
    commitment = hashlib.sha256(blind_key).hexdigest()
    if commitment != str(
        protocol["blinding"]["blind_key_sha256_commitment"]
    ):
        raise RuntimeError("blind key does not match protocol commitment")
    mapping: dict[str, dict[str, str]] = {}
    for frozen in protocol["selection"]["rows"]:
        for seed in protocol["sampling"]["joint_seeds"]:
            pair_id, side_mapping = pair_identity(
                protocol["protocol_name"],
                str(frozen["uid"]),
                int(seed),
                blind_key,
            )
            mapping[pair_id] = side_mapping
    pair_to_object: dict[str, str] = {}
    for row in sealed["records"]:
        pair_id = str(row["pair_id"])
        object_uid = str(row["object_uid"])
        previous = pair_to_object.setdefault(pair_id, object_uid)
        if previous != object_uid:
            raise RuntimeError(f"pair={pair_id} maps to multiple objects")

    automatic = aggregate_unblinded(
        list(sealed["records"]), mapping, protocol=protocol
    )
    ratings = aggregate_ratings(
        rater_files,
        mapping=mapping,
        pair_to_object=pair_to_object,
        bootstrap_samples=int(protocol["statistics"]["bootstrap_samples"]),
        checks_config=protocol["statistics"]["rating_checks"],
    )
    passed = bool(automatic["automatic_passed"] and ratings["passed"])
    report = {
        "format": FINAL_REPORT_FORMAT,
        "complete": True,
        "formal": True,
        "passed": passed,
        "protocol": str(protocol_path),
        "protocol_sha256": protocol["protocol_sha256"],
        "blind_output_dir": str(blind_root),
        "completion_manifest_sha256": sha256_file(
            blind_root / "completion_manifest.json"
        ),
        "sealed_report_sha256": sha256_file(sealed_path),
        "ratings_manifest": str(ratings_manifest_path),
        "ratings_manifest_sha256": sha256_file(ratings_manifest_path),
        "ratings_freeze_sha256": ratings_freeze["ratings_freeze_sha256"],
        "public_bundle_sha256": ratings_freeze["public_bundle"][
            "public_bundle_sha256"
        ],
        "public_archive_sha256": ratings_freeze["public_archive"][
            "archive_sha256"
        ],
        "candidate": protocol["candidate"],
        "automatic": automatic,
        "blind_ratings": ratings,
        "unblinding_mapping": mapping,
        "claim_scope": (
            "PASS supports only the frozen full step800 end-to-end candidate "
            "against native stock on this untouched object holdout. It does not "
            "establish LoRA-only/adapter-only causality and the holdout cannot be "
            "reused for configuration selection."
        ),
        "post_unblinding_policy": (
            "No object, seed, rater, or failed pair may be removed; a FAIL is a "
            "scientific result, not a prompt to retune thresholds on this holdout."
        ),
    }

    output_dir = Path(args.output_dir).resolve()
    marker_path = protocol_path.parent / "unblinded_once.json"
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if marker_path.exists():
        raise RuntimeError(
            f"this protocol has already been unblinded: {marker_path}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(output_dir / "final_report.json", report)
    write_markdown(output_dir / "final_report.md", report)
    marker = {
        "protocol_sha256": protocol["protocol_sha256"],
        "final_report": str((output_dir / "final_report.json").resolve()),
        "final_report_sha256": sha256_file(output_dir / "final_report.json"),
        "passed": passed,
        "rater_score_bindings": ratings["rater_files"],
    }
    descriptor = os.open(
        marker_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(marker, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "passed": passed,
                "automatic_passed": automatic["automatic_passed"],
                "blind_ratings_passed": ratings["passed"],
                "report": str(output_dir / "final_report.json"),
                "runtime_exit_code": 0 if passed else 2,
            },
            indent=2,
        ),
        flush=True,
    )
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
