#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
PYTHON=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
RUN_ROOT=${RUN_ROOT:-/data/zjr/proobjaverse_official_native_ss_train2000_with_vggt_20260817_v1}
TRAIN_REPORT=${TRAIN_REPORT:-${RUN_ROOT}/cache_train2000_official_ss_with_vggt_sidecar_historical_v2_fix2_v1/report.json}
DEV_REPORT=${DEV_REPORT:-${RUN_ROOT}/cache_dev64_official_ss_with_vggt_sidecar_historical_v2_fix2_v1/report.json}

cd "${PROJECT_ROOT}"
test -s "${TRAIN_REPORT}"
test -s "${DEV_REPORT}"

"${PYTHON}" - "${TRAIN_REPORT}" "${DEV_REPORT}" <<'PY'
import json
import sys

train, dev = [json.load(open(path, encoding="utf-8")) for path in sys.argv[1:]]
assert train["passed"] is True and train["object_count"] == 2000
assert dev["passed"] is True and dev["object_count"] == 64
for report, count in ((train, 2000), (dev, 64)):
    assert report["complete"] is True
    assert report["vggt_forward_call_count"] == count
    assert report["vggt_model_executed"] is True
    assert report["vggt_camera_consumed"] is False
    assert report["known_K_T_replaced"] is False
    assert report["official_ss_target_changed"] is False
    assert report["base_cache_rewritten"] is False
assert train["pair_identity"] != dev["pair_identity"]
print(json.dumps({
    "passed": True,
    "train_objects": train["object_count"],
    "dev_objects": dev["object_count"],
    "train_pair_identity": train["pair_identity"],
    "dev_pair_identity": dev["pair_identity"],
    "train_sidecar_bytes": train["sidecar_bytes"],
    "dev_sidecar_bytes": dev["sidecar_bytes"],
}, indent=2, ensure_ascii=False))
PY

