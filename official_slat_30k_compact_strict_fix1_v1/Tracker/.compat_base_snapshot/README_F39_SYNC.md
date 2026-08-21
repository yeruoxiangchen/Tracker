# F39 code-only snapshot

Snapshot: `Tracker_Aug16_30k_compact_compat_20260816_v2`.

This archive contains source/configuration/tests only. It excludes datasets, caches, outputs, logs, checkpoints, model weights, compiled artifacts and `__pycache__`.

Deploy into a new directory such as `/root/Tracker_30k_compact_compat_v2`; do not overwrite `/root/Tracker_perf_v1` or `/root/Tracker_perf_v1_fix1`.

The final distribution archive SHA256 is intentionally stored in the detached `.tar.zst.sha256` and `.tar.zst.manifest.json` sidecars. A file cannot contain the cryptographic digest of the final archive that contains that same file without creating a self-hash cycle.
