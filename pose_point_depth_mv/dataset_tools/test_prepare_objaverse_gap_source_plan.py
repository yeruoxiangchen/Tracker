from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pose_point_depth_mv.dataset_tools.prepare_objaverse_gap_source_plan import (
    select_gap_sources,
    shard_index,
)


class ObjaverseGapSourcePlanTests(unittest.TestCase):
    def test_selects_exact_balanced_fresh_hard_valid_pool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = {}
            audits = {}
            for index in range(200):
                uid = f"objaverse_{index:04d}"
                mesh = root / f"{index:04d}.glb"
                mesh.write_bytes(b"mesh")
                sources[uid] = str(mesh)
                audits[str(mesh.resolve())] = {
                    "source_glb": str(mesh),
                    "auto_tier": "B",
                    "mesh_audit": {"mesh_valid": True},
                }
            excluded_uid = next(uid for uid in sources if shard_index(uid, 2) == 0)
            excluded = {str(Path(sources[excluded_uid]).resolve())}
            selected, report = select_gap_sources(
                sources,
                audits,
                excluded,
                allowed_tiers={"A", "B"},
                seed=42,
                shard_count=2,
                objects_per_shard=4,
            )
            self.assertEqual(len(selected), 8)
            self.assertNotIn(excluded_uid, selected)
            self.assertEqual(
                [sum(shard_index(uid, 2) == shard for uid in selected) for shard in range(2)],
                [4, 4],
            )
            self.assertEqual(report["selected_count_by_shard"], [4, 4])


if __name__ == "__main__":
    unittest.main()
