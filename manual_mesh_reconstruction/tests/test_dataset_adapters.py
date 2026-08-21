from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from manual_mesh_reconstruction.data_adapters import ADAPTER_FORMAT
from manual_mesh_reconstruction.data_adapters.colmap import adapt as adapt_colmap
from manual_mesh_reconstruction.data_adapters.common import (
    select_indices,
    time_uniform_indices,
    validate_reusable_adapter_report,
)
from manual_mesh_reconstruction.data_adapters.phone import adapt as adapt_phone


def _write_rgb(path: Path, value: int) -> None:
    array = np.full((24, 32, 3), int(value), dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def _write_mask(path: Path) -> None:
    array = np.zeros((24, 32), dtype=np.uint8)
    array[5:20, 8:25] = 255
    Image.fromarray(array, mode="L").save(path)


def _write_colmap_fixture(dataset: Path, *, frame_count: int = 11) -> None:
    color = dataset / "color"
    masks = dataset / "masks"
    sparse = dataset / "sparse/0"
    color.mkdir(parents=True)
    masks.mkdir(parents=True)
    sparse.mkdir(parents=True)
    image_lines = [
        "# Image list with two lines of data per image:",
        "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME",
    ]
    for index in range(frame_count):
        name = f"{index:05d}.jpg"
        _write_rgb(color / name, index)
        _write_mask(masks / f"{index:05d}.png")
        image_lines.extend(
            [
                f"{index + 1} 1 0 0 0 {index * 0.01:.6f} 0 0 1 {name}",
                "0 0 -1",
            ]
        )
    (sparse / "cameras.txt").write_text(
        "1 PINHOLE 32 24 30 30 15.5 11.5\n", encoding="utf-8"
    )
    (sparse / "images.txt").write_text(
        "\n".join(image_lines) + "\n", encoding="utf-8"
    )
    (sparse / "points3D.txt").write_text("# empty fixture\n", encoding="utf-8")


class DatasetAdapterTests(unittest.TestCase):
    def test_time_uniform_selection_includes_endpoints(self) -> None:
        selected = time_uniform_indices(23, 8)
        self.assertEqual(selected.tolist(), [0, 3, 6, 9, 13, 16, 19, 22])
        self.assertEqual(len(set(selected.tolist())), 8)

    def test_random_selection_is_reproducible_and_chronological(self) -> None:
        first, record = select_indices(31, 8, policy="random", random_seed=17)
        second, _ = select_indices(31, 8, policy="random", random_seed=17)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.all(first[1:] > first[:-1]))
        self.assertEqual(record["execution_order"], "selected source indices sorted chronologically")

    def test_colmap_reuse_does_not_require_or_execute_colmap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            _write_colmap_fixture(dataset)
            result = adapt_colmap(
                input_path=dataset,
                output_dir=root / "adapter/raw_cache",
                selected_view_count=8,
                selection_policy="time_uniform",
                random_seed=42,
                colmap_mode="reuse",
                colmap_sparse=None,
                colmap_bin="intentionally_missing_colmap_binary",
                matcher="sequential",
                use_foreground_masks=False,
                use_gpu=False,
                resume=False,
            )
            self.assertEqual(result["source_binding"]["colmap"]["effective_mode"], "reuse")
            self.assertFalse(result["source_binding"]["colmap"]["colmap_executed"])
            self.assertFalse(
                result["source_binding"]["colmap"]["colmap_model_reconstruction_executed"]
            )
            self.assertFalse(
                result["source_binding"]["colmap"]["colmap_model_converter_executed"]
            )
            self.assertTrue(result["selection"]["selection_deferred_to_runtime_o"])
            self.assertEqual(
                len(result["selection"]["eligible_source_frame_names"]), 11
            )
            report = json.loads(Path(result["raw_cache_report"]).read_text(encoding="utf-8"))
            row = report["objects"][0]
            self.assertEqual(row["input_view_count"], 11)
            self.assertEqual(row["selected_source_indices"], [])
            self.assertEqual(row["eligible_source_indices"], list(range(11)))
            self.assertEqual(
                row["view_selection"]["selection_domain"],
                "successfully COLMAP-registered frames with masks",
            )
            adapter_report = root / "adapter_report.json"
            adapter_payload = {
                "format": ADAPTER_FORMAT,
                "passed": True,
                **result,
            }
            adapter_report.write_text(
                json.dumps(adapter_payload), encoding="utf-8"
            )
            self.assertTrue(validate_reusable_adapter_report(adapter_report)["passed"])
            adapter_payload["format"] = "manual_mesh_reconstruction.dataset_adapter.v1"
            adapter_report.write_text(
                json.dumps(adapter_payload), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "did not pass"):
                validate_reusable_adapter_report(adapter_report)

    def test_colmap_auto_prefers_complete_existing_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            _write_colmap_fixture(dataset, frame_count=9)
            result = adapt_colmap(
                input_path=dataset,
                output_dir=root / "adapter/raw_cache",
                selected_view_count=8,
                selection_policy="time_uniform",
                random_seed=42,
                colmap_mode="auto",
                colmap_sparse=None,
                colmap_bin="intentionally_missing_colmap_binary",
                matcher="sequential",
                use_foreground_masks=False,
                use_gpu=False,
                resume=False,
            )
            colmap = result["source_binding"]["colmap"]
            self.assertEqual(colmap["requested_mode"], "auto")
            self.assertEqual(colmap["effective_mode"], "reuse")
            self.assertFalse(colmap["colmap_executed"])

    def test_colmap_strict_reuse_fails_without_existing_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            (dataset / "color").mkdir(parents=True)
            (dataset / "masks").mkdir(parents=True)
            _write_rgb(dataset / "color/00000.jpg", 0)
            _write_mask(dataset / "masks/00000.png")
            with self.assertRaisesRegex(FileNotFoundError, "reuse requested"):
                adapt_colmap(
                    input_path=dataset,
                    output_dir=root / "adapter/raw_cache",
                    selected_view_count=1,
                    selection_policy="time_uniform",
                    random_seed=42,
                    colmap_mode="reuse",
                    colmap_sparse=None,
                    colmap_bin="intentionally_missing_colmap_binary",
                    matcher="sequential",
                    use_foreground_masks=False,
                    use_gpu=False,
                    resume=False,
                )

    def test_phone_runtime_input_is_normalized_without_colmap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            data = runtime / "data/session_a"
            masks = runtime / "masks/session_a"
            data.mkdir(parents=True)
            masks.mkdir(parents=True)
            pose_lines = ["frame_name,px,py,pz,rx,ry,rz"]
            for index in range(10):
                name = f"frame_{index:04d}.jpg"
                _write_rgb(data / name, index)
                _write_mask(masks / f"frame_{index:04d}.png")
                pose_lines.append(f"{name},{index * 0.01},0,0,0,0,0")
            (data / "poses.txt").write_text("\n".join(pose_lines) + "\n", encoding="utf-8")
            result = adapt_phone(
                input_path=runtime,
                output_dir=root / "adapter/raw_cache",
                selected_view_count=8,
                selection_policy="time_uniform",
                random_seed=42,
                session_id="session_a",
            )
            self.assertEqual(result["geometry_mode"], "pose_mask")
            self.assertTrue(result["selection"]["selection_deferred_to_runtime_o"])
            self.assertEqual(result["selection"]["requested_view_count"], 8)
            self.assertEqual(result["selection"]["eligible_source_indices"], list(range(10)))
            self.assertEqual(result["source_binding"]["source_kind"], "ar_foundation_runtime_capture")


if __name__ == "__main__":
    unittest.main()
