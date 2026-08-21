from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import trimesh

from manual_mesh_reconstruction.canonicalization import array_sha256
from manual_mesh_reconstruction.common import sha256_file
from manual_mesh_reconstruction.mesh_coordinates import mesh_frame_contract_fields
from manual_mesh_reconstruction.optimize_o2w import (
    compose_o2w_candidate,
    load_all_view_contract,
    resolve_existing_reconstruction,
)
from manual_mesh_reconstruction.server import export_world_and_mobile_meshes


class InputMaskO2WRefinementTests(unittest.TestCase):
    def _write_contract(self, root: Path) -> tuple[Path, Path, Path]:
        object_key = "fixture:object"
        images = root / "images"
        masks = root / "masks"
        images.mkdir(parents=True)
        masks.mkdir(parents=True)
        names = [f"view_{index:04d}.png" for index in range(8)]
        for index, name in enumerate(names):
            Image.new("RGB", (32, 32), (index, 20, 30)).save(images / name)
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[10:22, 10:22] = 255
            Image.fromarray(mask, mode="L").save(masks / name)

        K = np.repeat(np.eye(3, dtype=np.float64)[None], len(names), axis=0)
        K[:, 0, 0] = 30.0
        K[:, 1, 1] = 30.0
        K[:, 0, 2] = 15.5
        K[:, 1, 2] = 15.5
        T_W2C = np.repeat(np.eye(4, dtype=np.float64)[None], len(names), axis=0)
        T_W2C[:, 2, 3] = 2.0
        raw_cache = root / "raw_cache.npz"
        np.savez_compressed(
            raw_cache,
            frame_name=np.asarray(names),
            source_frame_name=np.asarray([f"frame_{i:04d}.jpg" for i in range(8)]),
            source_frame_index=np.arange(8, dtype=np.int64),
            K=K,
            T_W2C=T_W2C,
        )
        cameras = [
            {
                "frame_name": name,
                "model": "PINHOLE",
                "distortion": [],
                "width": 32,
                "height": 32,
            }
            for name in names
        ]
        raw_report = root / "raw_cache_report.json"
        raw_report.write_text(
            json.dumps(
                {
                    "passed": True,
                    "objects": [
                        {
                            "object_key": object_key,
                            "cache_npz": str(raw_cache.resolve()),
                            "images_dir": str(images.resolve()),
                            "masks_dir": str(masks.resolve()),
                            "cameras": cameras,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        T_O2W = np.eye(4, dtype=np.float64)
        T_O2W[:3, :3] *= 0.2
        T_O2W[:3, 3] = [0.1, -0.2, 0.3]
        runtime_cache = root / "runtime_cache.npz"
        np.savez_compressed(
            runtime_cache,
            T_O2W=T_O2W,
            T_W2O=np.linalg.inv(T_O2W),
        )
        runtime_manifest = root / "runtime_input_manifest.json"
        runtime_manifest.write_text(
            json.dumps(
                {
                    "passed": True,
                    "source_raw_cache_report": str(raw_report.resolve()),
                    "objects": [
                        {
                            "object_key": object_key,
                            "cache_npz": str(runtime_cache.resolve()),
                            "source_raw_cache": str(raw_cache.resolve()),
                            "source_raw_cache_sha256": sha256_file(raw_cache),
                            "all_input_view_count": 8,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return runtime_manifest, raw_cache, runtime_cache

    def test_similarity_composition_does_not_scale_world_translation(self) -> None:
        initial = np.eye(4, dtype=np.float64)
        initial[:3, :3] *= 0.2
        initial[:3, 3] = [1.0, 2.0, 3.0]
        rotation = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        candidate = compose_o2w_candidate(
            initial, rotation, [0.1, -0.2, 0.3], 1.25
        )
        np.testing.assert_allclose(candidate[:3, 3], [1.1, 1.8, 3.3])
        np.testing.assert_allclose(candidate[:3, :3], 0.25 * rotation)

    def test_load_all_view_contract_uses_preselection_camera_domain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime, raw_cache, runtime_cache = self._write_contract(Path(temporary))
            contract = load_all_view_contract(runtime, object_key="fixture:object")
            self.assertEqual(len(contract["frame_names"]), 8)
            self.assertEqual(contract["valid_indices"], list(range(8)))
            self.assertEqual(contract["source_indices"], list(range(8)))
            self.assertEqual(contract["cameras"][0]["model"], "PINHOLE")
            self.assertEqual(contract["bindings"]["raw_cache_sha256"], sha256_file(raw_cache))
            self.assertEqual(
                contract["bindings"]["runtime_cache_sha256"],
                sha256_file(runtime_cache),
            )

    def test_existing_reconstruction_resolver_and_refined_world_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime, _raw_cache, _runtime_cache = self._write_contract(root)
            session = root / "session"
            branch = session / "branches" / "01_training_spherical_farthest8"
            model = branch / "04_current_ss30k_slat30k"
            model.mkdir(parents=True)
            runtime_dir = branch / "02_runtime_o_all_views_then_spherical_fps8"
            runtime_dir.mkdir(parents=True)
            runtime_copy = runtime_dir / "runtime_input_manifest.json"
            runtime_copy.write_bytes(runtime.read_bytes())

            mesh_o = model / "mesh_o.obj"
            trimesh.Trimesh(
                vertices=np.asarray(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
                ),
                faces=np.asarray([[0, 1, 2]]),
                process=False,
            ).export(mesh_o)
            mesh_report = model / "result.json"
            mesh_report.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "mesh": str(mesh_o.resolve()),
                        "mesh_sha256": sha256_file(mesh_o),
                        "output_frame": "runtime-O",
                        **mesh_frame_contract_fields(
                            export_policy="decoded.to_trimesh(transform_pose=False)"
                        ),
                    }
                ),
                encoding="utf-8",
            )
            inference = model / "inference_manifest.json"
            inference.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "objects": [
                            {
                                "passed": True,
                                "object_key": "fixture:object",
                                "seed": 42,
                                "mesh": str(mesh_o.resolve()),
                                "result": str(mesh_report.resolve()),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            resolved = resolve_existing_reconstruction(session)
            self.assertEqual(resolved.runtime_input_manifest, runtime_copy.resolve())
            self.assertEqual(resolved.mesh_o, mesh_o.resolve())
            self.assertEqual(resolved.mesh_frame_report, mesh_report.resolve())

            selected = np.eye(4, dtype=np.float64)
            selected[:3, :3] *= 0.3
            selected[:3, 3] = [0.4, 0.5, 0.6]
            refinement = branch / "refinement_report.json"
            refinement.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "accepted": True,
                        "runtime_input_manifest": str(runtime_copy.resolve()),
                        "mesh_o": str(mesh_o.resolve()),
                        "selected_T_O2W_sha256": array_sha256(selected),
                        "decision": {"selected": "candidate_T_O2W"},
                    }
                ),
                encoding="utf-8",
            )
            exported = export_world_and_mobile_meshes(
                mesh_o=mesh_o,
                mesh_frame_report=mesh_report,
                runtime_input_manifest=runtime_copy,
                object_key="fixture:object",
                output_dir=branch / "world_export",
                T_O2W_override=selected,
                T_O2W_refinement_report=refinement,
            )
            world = trimesh.load(
                exported["internal_world_obj"], force="mesh", process=False
            )
            self.assertEqual(
                {tuple(np.round(row, 6)) for row in np.asarray(world.vertices)},
                {(0.4, 0.5, 0.6), (0.7, 0.5, 0.6), (0.4, 0.8, 0.6)},
            )
            self.assertEqual(
                exported["T_O2A0_source"], "input_mask_o2w_refinement_selected"
            )
            self.assertEqual(exported["T_O2A0_sha256"], array_sha256(selected))


if __name__ == "__main__":
    unittest.main()
