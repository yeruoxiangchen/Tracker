#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

ROOT=/home/zjr/Tracker/pose_point_depth_mv/outputs2/Snoopy全153帧同一OfficialO_Fixed8_时间均匀8_质量球面8_SS30K_SLat30K_轮廓_20260819_v1
RAW=${ROOT}/00_shared_dataset_adapter/raw_cache/raw_cache_report.json
LOGS=${ROOT}/logs

mkdir -p "${LOGS}"
test -s "${RAW}"

run_fixed() {
  bash manual_mesh_reconstruction/run_reconstruction.sh \
    --raw-cache-report "${RAW}" \
    --selected-view-count 8 \
    --resume \
    --output-dir "${ROOT}/01_fixed8" \
    --gpu 0 \
    --view-selection-policy fixed_frame_names_valid_mask \
    --fixed-frame-name 00001.jpg \
    --fixed-frame-name 00021.jpg \
    --fixed-frame-name 00051.jpg \
    --fixed-frame-name 00089.jpg \
    --fixed-frame-name 00101.jpg \
    --fixed-frame-name 00125.jpg \
    --fixed-frame-name 00131.jpg \
    --fixed-frame-name 00150.jpg
}

run_time_uniform() {
  bash manual_mesh_reconstruction/run_reconstruction.sh \
    --raw-cache-report "${RAW}" \
    --selected-view-count 8 \
    --resume \
    --output-dir "${ROOT}/02_time_uniform8" \
    --gpu 3 \
    --view-selection-policy time_uniform_valid_mask
}

run_quality_spherical() {
  bash manual_mesh_reconstruction/run_reconstruction.sh \
    --raw-cache-report "${RAW}" \
    --selected-view-count 8 \
    --resume \
    --output-dir "${ROOT}/03_quality_spherical8_full_visibility_v2" \
    --gpu 4 \
    --view-selection-policy object_spherical_farthest_valid_mask
}

run_fixed >"${LOGS}/01_fixed8.log" 2>&1 &
PID_FIXED=$!
run_time_uniform >"${LOGS}/02_time_uniform8.log" 2>&1 &
PID_TIME=$!
run_quality_spherical >"${LOGS}/03_quality_spherical8_full_visibility_v2.log" 2>&1 &
PID_SPHERICAL=$!

set +e
wait "${PID_FIXED}"
RC_FIXED=$?
wait "${PID_TIME}"
RC_TIME=$?
wait "${PID_SPHERICAL}"
RC_SPHERICAL=$?
set -e

printf 'fixed8_rc=%s\ntime_uniform8_rc=%s\nquality_spherical8_full_visibility_v2_rc=%s\n' \
  "${RC_FIXED}" "${RC_TIME}" "${RC_SPHERICAL}" \
  >"${ROOT}/THREE_SELECTION_EXIT_CODES.txt"

if ((RC_FIXED != 0 || RC_TIME != 0 || RC_SPHERICAL != 0)); then
  exit 91
fi

date -Is >"${ROOT}/THREE_SELECTION_COMPLETE.txt"
