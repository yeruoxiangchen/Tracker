from __future__ import annotations

import copy
import unittest

from pose_aligned_reconstruction.proobjaverse_official_slat_protocol import (
    COMBINED_30K_PAIR_COUNT,
    COMBINED_30K_TRAIN_COUNT,
    COMBINED_SELECTION_FORMAT,
    SELECTION_FORMAT,
    _selection_canonical_sha256,
    partition_rows,
    split_counts_for_train,
    validate_frozen_selection,
)


def _row(index: int) -> dict:
    uid = f"uid{index:05d}"
    shard = f"shard-{index // 1000 + 1:04d}"
    return {
        "uid": uid,
        "shard": shard,
        "render": {
            "path": f"renders_random_env/{shard}/{uid}.tar",
            "size": index + 101,
        },
        "slat": {
            "path": f"lh-slats/{shard}/{uid}.npz",
            "size": index + 11,
        },
    }


def _legacy_selection(rows: list[dict]) -> dict:
    body = {
        "format": SELECTION_FORMAT,
        "repo_id": "Stable-X/ProObjaverse-300K",
        "repo_type": "dataset",
        "revision": "revision",
        "selected_pair_count": len(rows),
        "selected": rows,
    }
    return {
        "created_at_utc": "fixture",
        **body,
        "selection_sha256": _selection_canonical_sha256(body),
    }


def _combined_selection(rows: list[dict]) -> dict:
    body = {
        "format": COMBINED_SELECTION_FORMAT,
        "formal": False,
        "data_root": "/fixture",
        "repository": {
            "repo_id": "Stable-X/ProObjaverse-300K",
            "revision": "revision",
        },
        "pair_count": len(rows),
        "selected": rows,
    }
    return {
        **body,
        "combined_selection_sha256": _selection_canonical_sha256(body),
    }


class OfficialSLatProtocolCompatibilityTests(unittest.TestCase):
    def test_legacy_protocol_identity_and_partition_regression(self) -> None:
        rows = [_row(index) for index in range(2000)]
        binding = validate_frozen_selection(_legacy_selection(rows))
        self.assertEqual(binding["format"], SELECTION_FORMAT)
        self.assertEqual(binding["identity_field"], "selection_sha256")
        self.assertEqual(binding["repository"]["revision"], "revision")
        splits = partition_rows(rows, split_counts_for_train(1872), seed=20260813)
        self.assertEqual(
            {name: len(value) for name, value in splits.items()},
            {
                "decoder_audit": 32,
                "predicted_support_bridge": 32,
                "dev": 64,
                "train": 1872,
            },
        )

    def test_combined_30k_fixture_counts_and_disjointness(self) -> None:
        rows = [_row(index) for index in range(COMBINED_30K_PAIR_COUNT)]
        binding = validate_frozen_selection(_combined_selection(rows))
        self.assertEqual(binding["format"], COMBINED_SELECTION_FORMAT)
        self.assertEqual(binding["identity_field"], "combined_selection_sha256")
        counts = split_counts_for_train(COMBINED_30K_TRAIN_COUNT)
        splits = partition_rows(rows, counts, seed=20260813)
        self.assertEqual(len(splits["train"]), 29872)
        self.assertEqual(len(splits["decoder_audit"]), 32)
        self.assertEqual(len(splits["predicted_support_bridge"]), 32)
        self.assertEqual(len(splits["dev"]), 64)
        uid_sets = [
            {str(row["uid"]) for row in splits[name]}
            for name in ("decoder_audit", "predicted_support_bridge", "dev", "train")
        ]
        self.assertEqual(sum(map(len, uid_sets)), 30000)
        self.assertEqual(len(set().union(*uid_sets)), 30000)
        for index, left in enumerate(uid_sets):
            for right in uid_sets[index + 1 :]:
                self.assertFalse(left & right)

    def test_subset_and_combined_identity_fields_cannot_mix(self) -> None:
        legacy = _legacy_selection([_row(0)])
        legacy["combined_selection_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "identity fields"):
            validate_frozen_selection(legacy)

        combined = _combined_selection([_row(0)])
        combined["selection_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "identity fields"):
            validate_frozen_selection(combined)

    def test_selection_hash_and_uid_accounting_remain_strict(self) -> None:
        selection = _combined_selection([_row(0), _row(1)])
        changed = copy.deepcopy(selection)
        changed["selected"][0]["render"]["size"] += 1
        with self.assertRaisesRegex(ValueError, "identity hash"):
            validate_frozen_selection(changed)

        duplicate = _legacy_selection([_row(0), _row(0)])
        with self.assertRaisesRegex(ValueError, "UID accounting"):
            validate_frozen_selection(duplicate)


if __name__ == "__main__":
    unittest.main()
