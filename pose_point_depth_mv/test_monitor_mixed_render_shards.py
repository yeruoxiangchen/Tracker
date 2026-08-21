from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pose_point_depth_mv.dataset_tools.monitor_mixed_render_shards import (
    current_worker_shard,
    shard_state,
)


class MonitorMixedRenderShardsTest(unittest.TestCase):
    def test_progress_and_worker_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            render_root = root / "render"
            log_root = root / "logs"
            log = log_root / "omni" / "shard_001.log"
            log.parent.mkdir(parents=True)
            log.write_text(
                "Building:  25%|██▌       | 16/64 "
                "[10:00<30:00, accepted=12, failed=4, last=ok]\n",
                encoding="utf-8",
            )
            states = [
                shard_state(render_root, log_root, "omni", index, 64)
                for index in range(4)
            ]
            active = current_worker_shard(states, 1, 2)
            self.assertIsNotNone(active)
            assert active is not None
            self.assertEqual(active["index"], 1)
            self.assertEqual(active["done"], 16)
            self.assertEqual(active["accepted"], 12)
            self.assertEqual(active["failed"], 4)

    def test_complete_manifest_and_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            render_root = root / "render"
            log_root = root / "logs"
            shard = render_root / "omni" / "shard_000"
            shard.mkdir(parents=True)
            (shard / "manifest.json").write_text(
                json.dumps(
                    {
                        "samples": [{"uid": str(index)} for index in range(50)],
                        "failures": [{"uid": str(index)} for index in range(14)],
                    }
                ),
                encoding="utf-8",
            )
            (shard / "_WORKER_COMPLETE.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            state = shard_state(render_root, log_root, "omni", 0, 64)
            self.assertTrue(state["marker"])
            self.assertEqual(state["status"], "完成")
            self.assertEqual(state["done"], 64)
            self.assertEqual(state["accepted"], 50)
            self.assertEqual(state["failed"], 14)


if __name__ == "__main__":
    unittest.main()
