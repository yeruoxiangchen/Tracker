from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pose_point_depth_mv.direct_slat_blind import (
    HOLDOUT_INTEGRITY_FORMAT,
    PROTOCOL_FORMAT,
    PUBLIC_BUNDLE_FORMAT,
    RATINGS_FREEZE_FORMAT,
    RATER_COLUMNS,
    canonical_sha256,
    sha256_file,
)
from pose_point_depth_mv.direct_slat_flow import DIRECT_SLAT_CACHE_VERSION
from pose_point_depth_mv.freeze_direct_slat_holdout_integrity import (
    main as integrity_main,
)
from pose_point_depth_mv.freeze_direct_slat_ratings import main as freeze_main
from pose_point_depth_mv.package_direct_slat_public_bundle import (
    main as package_main,
)


class DirectSLatReleaseTest(unittest.TestCase):
    def write_csv(
        self, path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def score_row(self, rater_id: str, pair_id: str) -> dict[str, str]:
        return {
            "rater_id": rater_id,
            "pair_id": pair_id,
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

    def test_package_is_portable_and_ratings_bind_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path = root / "protocol.json"
            protocol = {
                "format": PROTOCOL_FORMAT,
                "mode": "confirmatory",
                "formal": True,
                "protocol_name": "test",
            }
            protocol["protocol_sha256"] = canonical_sha256(protocol)
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

            blind = root / "blind"
            pair_id = "pair001"
            for side in ("A", "B"):
                path = blind / "blind_pairs" / pair_id / side / "mesh_view.glb"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"mesh-{side}".encode("ascii"))
            self.write_csv(
                blind / "blind_manifest.csv",
                [
                    {
                        "pair_id": pair_id,
                        "object_id": "object_0000",
                        "seed": "42",
                        "A_mesh": f"blind_pairs/{pair_id}/A/mesh_view.glb",
                        "B_mesh": f"blind_pairs/{pair_id}/B/mesh_view.glb",
                        "A_preview": "",
                        "B_preview": "",
                    }
                ],
                (
                    "pair_id",
                    "object_id",
                    "seed",
                    "A_mesh",
                    "B_mesh",
                    "A_preview",
                    "B_preview",
                ),
            )
            for index in range(1, 4):
                self.write_csv(
                    blind / "score_templates" / f"scores_R{index}.csv",
                    [
                        {
                            key: (
                                f"R{index}"
                                if key == "rater_id"
                                else pair_id
                                if key == "pair_id"
                                else ""
                            )
                            for key in RATER_COLUMNS
                        }
                    ],
                    RATER_COLUMNS,
                )
            sealed = blind / "sealed" / "sealed_metrics.json"
            sealed.parent.mkdir(parents=True)
            sealed.write_text("{}", encoding="utf-8")
            files = [
                {
                    "path": path.relative_to(blind).as_posix(),
                    "sha256": sha256_file(path),
                }
                for path in sorted(blind.rglob("*"))
                if path.is_file()
            ]
            completion = {
                "complete": True,
                "formal": True,
                "mode": "confirmatory",
                "all_records_passed": True,
                "runtime_exit_code": 0,
                "science_decision_emitted": False,
                "protocol_sha256": protocol["protocol_sha256"],
                "sealed_report_sha256": sha256_file(sealed),
                "files": files,
            }
            (blind / "completion_manifest.json").write_text(
                json.dumps(completion), encoding="utf-8"
            )

            public = root / "public"
            archive = root / "public.tar"
            archive_manifest = root / "public.tar.manifest.json"
            with mock.patch(
                "sys.argv",
                [
                    "package",
                    "--protocol",
                    str(protocol_path),
                    "--blind_output_dir",
                    str(blind),
                    "--output_dir",
                    str(public),
                    "--archive",
                    str(archive),
                    "--archive_manifest",
                    str(archive_manifest),
                ],
            ):
                package_main()
            manifest = json.loads(
                (public / "public_bundle_manifest.json").read_text()
            )
            self.assertEqual(manifest["format"], PUBLIC_BUNDLE_FORMAT)
            self.assertTrue(manifest["portable_paths"])
            with (public / "blind_manifest.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                public_row = next(csv.DictReader(handle))
            self.assertFalse(Path(public_row["A_mesh"]).is_absolute())
            self.assertTrue(archive.is_file())

            ratings = root / "ratings"
            score_paths = []
            for index in range(1, 4):
                score = ratings / f"scores_R{index}.csv"
                self.write_csv(
                    score,
                    [self.score_row(f"R{index}", pair_id)],
                    RATER_COLUMNS,
                )
                score_paths.append(score)
            freeze_output = ratings / "freeze.json"
            with mock.patch(
                "sys.argv",
                [
                    "freeze",
                    "--public_bundle_manifest",
                    str(public / "public_bundle_manifest.json"),
                    "--public_archive_manifest",
                    str(archive_manifest),
                    "--rater_scores",
                    ",".join(str(path) for path in score_paths),
                    "--output",
                    str(freeze_output),
                ],
            ):
                freeze_main()
            freeze = json.loads(freeze_output.read_text())
            self.assertEqual(freeze["format"], RATINGS_FREEZE_FORMAT)
            self.assertFalse(freeze["blind_key_read"])
            self.assertEqual(
                freeze["public_bundle"]["public_bundle_sha256"],
                manifest["public_bundle_sha256"],
            )

    def test_post_selection_integrity_binds_exact_object_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            uids = ["object-a", "object-b"]
            selection = {
                "format": "pose_point_depth_mv.direct_slat_holdout_manifest.v1",
                "passed": True,
                "model_outputs_read": False,
                "selected": [{"object_uid": uid} for uid in uids],
            }
            selection["audit_sha256"] = canonical_sha256(selection)
            lifting = {
                "object_count": 2,
                "sample_count": 2,
                "samples": [{"object_uid": uid} for uid in uids],
            }
            paths = {
                "selection": root / "selection.json",
                "lifting": root / "lifting.json",
                "lifting_audit": root / "lifting_audit.json",
                "stock_audit": root / "stock_audit.json",
                "cache": root / "cache.json",
                "target_audit": root / "target_audit.json",
            }
            paths["selection"].write_text(json.dumps(selection))
            paths["lifting"].write_text(json.dumps(lifting))
            lifting_sha = sha256_file(paths["lifting"])
            paths["lifting_audit"].write_text(
                json.dumps(
                    {"passed": True, "cache_manifest_sha256": lifting_sha}
                )
            )
            paths["stock_audit"].write_text(
                json.dumps(
                    {"passed": True, "cache_manifest_sha256": lifting_sha}
                )
            )
            cache = {
                "format": DIRECT_SLAT_CACHE_VERSION,
                "materialized": True,
                "object_count": 2,
                "sequence_count": 2,
                "sample_count": 6,
                "objects": [{"object_uid": uid} for uid in uids],
            }
            paths["cache"].write_text(json.dumps(cache))
            paths["target_audit"].write_text(
                json.dumps(
                    {
                        "passed": True,
                        "cache_manifest_sha256": sha256_file(paths["cache"]),
                        "records": [{"object_uid": uid} for uid in uids],
                    }
                )
            )
            output = root / "integrity.json"
            invalid = root / "INVALID.json"
            with mock.patch(
                "sys.argv",
                [
                    "integrity",
                    "--mode",
                    "freeze",
                    "--invalid_marker",
                    str(invalid),
                    "--selection_audit",
                    str(paths["selection"]),
                    "--lifting_manifest",
                    str(paths["lifting"]),
                    "--lifting_audit",
                    str(paths["lifting_audit"]),
                    "--stock_replay_audit",
                    str(paths["stock_audit"]),
                    "--cache_manifest",
                    str(paths["cache"]),
                    "--target_decoder_audit",
                    str(paths["target_audit"]),
                    "--expected_objects",
                    "2",
                    "--output",
                    str(output),
                ],
            ):
                integrity_main()
            report = json.loads(output.read_text())
            self.assertEqual(report["format"], HOLDOUT_INTEGRITY_FORMAT)
            self.assertTrue(report["passed"])


if __name__ == "__main__":
    unittest.main()
