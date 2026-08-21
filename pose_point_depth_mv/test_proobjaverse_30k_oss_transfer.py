from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pose_point_depth_mv.tools.proobjaverse_30k_oss_transfer import (
    _EMPTY_OBJECT_ETAG,
    _parse_oss_listing,
)


class OssDirectoryMarkerTest(unittest.TestCase):
    OSS_ROOT = "oss://bucket/frozen-30k"

    def _parse(self, rows: list[tuple[int, str, str]]):
        with tempfile.TemporaryDirectory() as directory:
            listing = Path(directory) / "listing.txt"
            listing.write_text(
                "".join(
                    "2026-08-16 13:18:00 +0000 UTC "
                    f"{size} Standard {etag} "
                    f"{self.OSS_ROOT}/payload/{relative}\n"
                    for size, etag, relative in rows
                ),
                encoding="utf-8",
            )
            return _parse_oss_listing(listing, self.OSS_ROOT)

    def test_exact_zero_byte_directory_marker_is_transport_metadata(self):
        objects, duplicates, markers = self._parse(
            [
                (0, _EMPTY_OBJECT_ETAG.upper(), "lh-slats/shard-0001/"),
                (7, "a" * 32, "lh-slats/shard-0001/object.npz"),
            ]
        )
        self.assertEqual(set(objects), {"lh-slats/shard-0001/object.npz"})
        self.assertEqual(duplicates, [])
        self.assertEqual(markers, ["lh-slats/shard-0001/"])

    def test_nonzero_slash_object_is_not_ignored(self):
        objects, duplicates, markers = self._parse(
            [(1, "a" * 32, "lh-slats/shard-0001/")]
        )
        self.assertEqual(set(objects), {"lh-slats/shard-0001/"})
        self.assertEqual(duplicates, [])
        self.assertEqual(markers, [])

    def test_wrong_etag_slash_object_is_not_ignored(self):
        objects, duplicates, markers = self._parse(
            [(0, "a" * 32, "renders_random_env/shard-0001/")]
        )
        self.assertEqual(set(objects), {"renders_random_env/shard-0001/"})
        self.assertEqual(duplicates, [])
        self.assertEqual(markers, [])


if __name__ == "__main__":
    unittest.main()
