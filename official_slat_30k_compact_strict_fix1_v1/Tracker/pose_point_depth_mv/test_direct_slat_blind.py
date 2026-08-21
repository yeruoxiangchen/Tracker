from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from pose_point_depth_mv.direct_slat_blind import (
    RATER_COLUMNS,
    aggregate_ratings,
    aggregate_unblinded,
    assert_unseen_holdout,
    execution_compatibility_record,
    pair_identity,
    read_and_validate_rater_csv,
    repeat_floors,
    runtime_selection_rows,
    select_object_rows,
    target_family_identity,
)
from pose_point_depth_mv.direct_slat_flow import (
    SUPPORT_RUNTIME_FIELDS,
    legacy_support_runtime_identity,
    support_generator_identity,
    support_runtime_identity,
)


def support_config(target_name: str) -> dict:
    values = {
        "pretrained": "model",
        "ss_flow_checkpoint_sha256": "ss",
        "expected_ss_step": 900,
        "correspondence_checkpoint_sha256": "corr",
        "n3_report_sha256": "n3",
        "ss_seeds": [42, 43, 44],
        "ss_steps": 30,
        "cfg_strength": 7.5,
        "guidance_rescale": 0.5,
        "rescale_t": 3.0,
        "physical_scale": 1.0,
        "condition_replay_max_abs": 0.001,
        "min_frame_iou": 0.9,
        "amp_dtype": "bf16",
        "mapping_version": "mapping",
    }
    assert set(values) == set(SUPPORT_RUNTIME_FIELDS)
    values["target_source"] = {"run_config": target_name}
    return values


def protocol(object_count: int = 3) -> dict:
    return {
        "selection": {"object_count": object_count},
        "sampling": {"joint_seeds": [42, 43, 44]},
        "statistics": {
            "bootstrap_samples": 200,
            "repeat_floors": {
                "chamfer_l1_abs": 1.0e-4,
                "fscore_0p02_abs": 1.0e-4,
                "largest_component_ratio_abs": 1.0e-4,
                "boundary_edge_count_abs": 0.0,
                "boundary_total_length_abs": 0.0,
                "nonmanifold_edge_count_abs": 0.0,
                "component_count_abs": 0.0,
            },
            "checks": {
                "chamfer_object_win_rate_min": 0.55,
                "minimum_nonnegative_seed_directions": 2,
                "per_seed_chamfer_mean_min": -0.002,
                "largest_component_ratio_mean_delta_min": -0.02,
                "largest_component_ratio_object_delta_min": -0.10,
                "largest_component_ratio_pair_delta_min": -0.15,
                "boundary_edge_count_mean_increase_max": 32.0,
                "boundary_edge_count_object_increase_max": 256.0,
                "boundary_edge_count_pair_increase_max": 512.0,
                "boundary_total_length_mean_increase_max": 0.02,
                "boundary_total_length_object_increase_max": 0.25,
                "boundary_total_length_pair_increase_max": 0.50,
                "nonmanifold_edge_count_mean_increase_max": 8.0,
                "nonmanifold_edge_count_object_increase_max": 64.0,
                "nonmanifold_edge_count_pair_increase_max": 128.0,
                "component_count_mean_increase_max": 1.0,
                "component_count_object_increase_max": 10.0,
                "component_count_pair_increase_max": 20.0,
                "topology_object_worsening_rate_max": 0.10,
                "watertight_rate_delta_min": -0.05,
                "zero_boundary_rate_delta_min": -0.05,
                "nonmanifold_free_rate_delta_min": 0.0,
            },
        },
    }


def structure(*, watertight: bool = True) -> dict:
    return {
        "mesh_success": True,
        "vertices_finite": True,
        "is_winding_consistent": True,
        "is_watertight": watertight,
        "boundary_edge_count": 0 if watertight else 4,
        "boundary_total_length": 0.0 if watertight else 0.01,
        "nonmanifold_edge_count": 0,
        "component_count": 1,
        "largest_component_ratio": 0.95,
    }


def surface(branch: str) -> dict:
    if branch == "stock":
        return {
            "chamfer_l1": 0.10,
            "fscore_0p02": 0.20,
            "normal_consistency": 0.50,
        }
    return {
        "chamfer_l1": 0.09,
        "fscore_0p02": 0.22,
        "normal_consistency": 0.51,
    }


def sealed_records(*, topology_regression: bool = False):
    records = []
    mapping = {}
    pair_to_object = {}
    for object_index in range(3):
        object_uid = f"obj{object_index}"
        for seed in (42, 43, 44):
            pair_id = f"pair_{object_index}_{seed}"
            mapping[pair_id] = {"A": "stock", "B": "full"}
            pair_to_object[pair_id] = object_uid
            for side, branch in (("A", "stock"), ("B", "full")):
                records.append(
                    {
                        "pair_id": pair_id,
                        "side": side,
                        "object_uid": object_uid,
                        "seed": seed,
                        "passed": True,
                        "surface": surface(branch),
                        "structure": structure(
                            watertight=not (
                                topology_regression and branch == "full"
                            )
                        ),
                    }
                )
    return records, mapping, pair_to_object


class SupportIdentityTest(unittest.TestCase):
    def test_new_target_source_keeps_runtime_identity(self) -> None:
        train = support_config("train.json")
        holdout = support_config("holdout.json")
        self.assertNotEqual(
            support_generator_identity(train),
            support_generator_identity(holdout),
        )
        self.assertEqual(
            support_runtime_identity(train),
            support_runtime_identity(holdout),
        )
        self.assertEqual(
            legacy_support_runtime_identity(
                support_generator_identity(train)
            ),
            support_runtime_identity(holdout),
        )


class ProtocolHelperTest(unittest.TestCase):
    def test_runtime_selection_uses_bound_lifting_index(self) -> None:
        selected = {
            "object_position": 0,
            "object_uid": "object",
            "uid": "object_seq000",
            "view_count": 8,
            "cache_indices": {"42": 3, "43": 4, "44": 5},
        }
        bound = {**selected, "source_lifting_index": 7}
        protocol_value = {
            "selection": {"rows": [selected]},
            "sample_bindings": [bound],
        }
        rows = runtime_selection_rows(protocol_value)
        self.assertEqual(rows[0]["source_lifting_index"], 7)
        protocol_value["sample_bindings"][0] = {
            **bound,
            "cache_indices": {"42": 99},
        }
        with self.assertRaises(RuntimeError):
            runtime_selection_rows(protocol_value)

    def test_selection_requires_all_seeds_and_one_sequence_per_object(self) -> None:
        rows = [
            {
                "object_uid": object_uid,
                "uid": uid,
                "support_seed": seed,
                "view_count": 4,
            }
            for object_uid, uid in (("B", "B1"), ("A", "A1"))
            for seed in (42, 43, 44)
        ]
        selected = select_object_rows(rows, seeds=(42, 43, 44))
        self.assertEqual([row["object_uid"] for row in selected], ["A", "B"])
        self.assertEqual(
            set(selected[0]["cache_indices"]), {"42", "43", "44"}
        )

    def test_unseen_checks_uid_and_source_hash(self) -> None:
        holdout = {
            "samples": [
                {
                    "object_uid": "new",
                    "source_glb": "/tmp/new.glb",
                    "source_glb_sha256": "new-sha",
                }
            ]
        }
        seen = {
            "samples": [
                {
                    "object_uid": "old",
                    "source_glb": "/tmp/old.glb",
                    "source_glb_sha256": "old-sha",
                }
            ]
        }
        self.assertTrue(assert_unseen_holdout(holdout, [seen])["passed"])
        seen["samples"][0]["source_glb_sha256"] = "new-sha"
        with self.assertRaises(RuntimeError):
            assert_unseen_holdout(holdout, [seen])

    def test_target_family_ignores_dataset_specific_paths(self) -> None:
        fields = {
            "format": "pose_point_depth_mv.local_lh_slats.v2",
            "source_kind": "local",
            "encoder_config_sha256": "encoder-config",
            "encoder_weights_sha256": "encoder",
            "mesh_decoder_weights_sha256": "decoder",
            "dinov2_hubconf_sha256": "hub",
            "dinov2_checkpoint_sha256": "dino",
            "dinov2_model": "dinov2_vitl14_reg",
            "image_size": 518,
            "min_views": 8,
            "max_views": 16,
            "coordinate_source": "coords",
            "feature_fusion": "mean",
            "posterior": "mean",
        }
        left = {"config": {**fields, "required_object_count": 100}}
        right = {"config": {**fields, "required_object_count": 32}}
        self.assertEqual(
            target_family_identity(left), target_family_identity(right)
        )

    def test_hmac_mapping_is_stable_and_balanced_pair(self) -> None:
        first = pair_identity("protocol", "uid", 42, b"k" * 32)
        second = pair_identity("protocol", "uid", 42, b"k" * 32)
        self.assertEqual(first, second)
        self.assertEqual(set(first[1]), {"A", "B"})
        self.assertEqual(set(first[1].values()), {"stock", "full"})

    def test_execution_compatibility_changes_with_sampling(self) -> None:
        value = {
            "candidate": {"step": 800},
            "code_bindings": {"code.py": {"path": "/x", "sha256": "a"}},
            "runtime_bindings": {"runtime": {"path": "/y", "sha256": "b"}},
            "bindings": {
                "ss_flow_checkpoint": {"sha256": "ss"},
                "direct_slat_checkpoint": {"sha256": "slat"},
            },
            "pretrained_id": "model",
            "pretrained": "/snapshot",
            "support_runtime_identity": {"hash": "support"},
            "target_family_identity": {"hash": "target"},
            "runtime": {"amp_dtype": "bf16"},
            "sampling": {"joint_seeds": [42, 43, 44]},
            "mesh": {"surface_samples": 20000},
        }
        first = execution_compatibility_record(value)
        value["sampling"]["joint_seeds"] = [42, 43]
        second = execution_compatibility_record(value)
        self.assertNotEqual(first["sha256"], second["sha256"])


class DecisionTest(unittest.TestCase):
    def test_surface_and_safety_pass(self) -> None:
        records, mapping, _ = sealed_records()
        result = aggregate_unblinded(
            records, mapping, protocol=protocol()
        )
        self.assertTrue(result["automatic_passed"], result["checks"])

    def test_topology_regression_overrides_surface_gain(self) -> None:
        records, mapping, _ = sealed_records(topology_regression=True)
        result = aggregate_unblinded(
            records, mapping, protocol=protocol()
        )
        self.assertFalse(result["automatic_passed"])
        self.assertFalse(result["checks"]["watertight_rate_non_degrading"])
        self.assertFalse(result["checks"]["zero_boundary_rate_non_degrading"])

    def test_repeat_topology_jitter_fails(self) -> None:
        rows = [
            {
                "metric_abs_diff": {
                    "chamfer_l1_abs": 1.0e-5,
                    "fscore_0p02_abs": 2.0e-5,
                    "largest_component_ratio_abs": 0.0,
                    "boundary_edge_count_abs": 0.0,
                    "boundary_total_length_abs": 0.0,
                    "nonmanifold_edge_count_abs": 0.0,
                    "component_count_abs": 0.0,
                },
                "topology_changed": {
                    "mesh_success": False,
                    "is_watertight": True,
                    "zero_boundary": False,
                    "nonmanifold_free": False,
                },
            }
        ]
        self.assertFalse(repeat_floors(rows)["passed"])

    def test_continuous_boundary_regression_cannot_hide_behind_binary_rate(
        self,
    ) -> None:
        records, mapping, _ = sealed_records()
        for row in records:
            row["structure"]["is_watertight"] = False
            row["structure"]["boundary_edge_count"] = (
                10 if row["side"] == "A" else 10000
            )
            row["structure"]["boundary_total_length"] = (
                0.01 if row["side"] == "A" else 10.0
            )
        result = aggregate_unblinded(records, mapping, protocol=protocol())
        self.assertFalse(result["automatic_passed"])
        self.assertTrue(result["checks"]["zero_boundary_rate_non_degrading"])
        self.assertFalse(result["checks"]["boundary_edge_count_mean_bounded"])
        self.assertFalse(
            result["checks"]["no_catastrophic_object_topology_regression"]
        )

    def test_three_rater_object_aggregation(self) -> None:
        _, mapping, pair_to_object = sealed_records()
        pair_ids = sorted(mapping)
        rater_files = []
        for index in range(3):
            rater_files.append(
                {
                    "path": f"R{index}.csv",
                    "sha256": str(index),
                    "rater_id": f"R{index}",
                    "rows": [
                        {
                            "rater_id": f"R{index}",
                            "pair_id": pair_id,
                            "scores": {
                                "main_structure": {"A": 3, "B": 4},
                                "missing_parts": {"A": 1, "B": 0},
                                "floating_fragments": {"A": 1, "B": 0},
                                "thin_spikes": {"A": 1, "B": 0},
                                "holes_open_boundaries": {"A": 1, "B": 0},
                                "overall_score": {"A": 3, "B": 4},
                            },
                            "preference": "B",
                        }
                        for pair_id in pair_ids
                    ],
                }
            )
        result = aggregate_ratings(
            rater_files,
            mapping=mapping,
            pair_to_object=pair_to_object,
            bootstrap_samples=100,
            checks_config=rating_checks(),
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["summary"]["overall_score_delta"]["mean"], 1.0)

    def test_all_ties_do_not_pass_blind_review(self) -> None:
        _, mapping, pair_to_object = sealed_records()
        pair_ids = sorted(mapping)
        rater_files = [
            {
                "path": f"R{index}.csv",
                "sha256": str(index),
                "rater_id": f"R{index}",
                "rows": [
                    {
                        "rater_id": f"R{index}",
                        "pair_id": pair_id,
                        "scores": {
                            "main_structure": {"A": 3, "B": 3},
                            "missing_parts": {"A": 0, "B": 0},
                            "floating_fragments": {"A": 0, "B": 0},
                            "thin_spikes": {"A": 0, "B": 0},
                            "holes_open_boundaries": {"A": 0, "B": 0},
                            "overall_score": {"A": 3, "B": 3},
                        },
                        "preference": "tie",
                    }
                    for pair_id in pair_ids
                ],
            }
            for index in range(3)
        ]
        result = aggregate_ratings(
            rater_files,
            mapping=mapping,
            pair_to_object=pair_to_object,
            bootstrap_samples=100,
            checks_config=rating_checks(),
        )
        self.assertFalse(result["passed"])
        self.assertFalse(
            result["checks"]["overall_score_strictly_favors_full"]
        )
        self.assertFalse(
            result["checks"]["overall_preference_strictly_favors_full"]
        )

    def test_rater_csv_requires_exact_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=RATER_COLUMNS)
                writer.writeheader()
                writer.writerow(
                    {
                        "rater_id": "R1",
                        "pair_id": "p1",
                        "main_structure_A": "3",
                        "main_structure_B": "4",
                        "missing_parts_A": "1",
                        "missing_parts_B": "0",
                        "floating_fragments_A": "1",
                        "floating_fragments_B": "0",
                        "thin_spikes_A": "1",
                        "thin_spikes_B": "0",
                        "holes_open_boundaries_A": "1",
                        "holes_open_boundaries_B": "0",
                        "overall_score_A": "3",
                        "overall_score_B": "4",
                        "overall_preference": "B",
                        "notes": "",
                    }
                )
            value = read_and_validate_rater_csv(
                path, expected_pair_ids=["p1"]
            )
            self.assertEqual(value["rater_id"], "R1")
            with self.assertRaises(ValueError):
                read_and_validate_rater_csv(
                    path, expected_pair_ids=["p1", "p2"]
                )


def rating_checks() -> dict:
    return {
        "main_structure_mean_delta_min": 0.0,
        "main_structure_ci_lower_min": -0.25,
        "defect_mean_delta_max": 0.0,
        "severe_defect_rate_delta_max": 0.05,
        "severe_defect_rate_ci_upper_max": 0.10,
        "overall_score_mean_min_exclusive": 0.0,
        "overall_score_ci_lower_min": 0.0,
        "overall_preference_mean_min_exclusive": 0.0,
        "overall_preference_ci_lower_min": 0.0,
    }


if __name__ == "__main__":
    unittest.main()
