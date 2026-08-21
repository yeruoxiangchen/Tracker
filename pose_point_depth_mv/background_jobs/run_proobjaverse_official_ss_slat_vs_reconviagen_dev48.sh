#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

source /home/zjr/anaconda3/etc/profile.d/conda.sh
conda activate reconviagen

export SPCONV_ALGO=native

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
ROOT=/data/zjr/proobjaverse_official_slat_train2000_20260813_v1

STRICT_ROOT=${ROOT}/eval_dev64_reconviagen_original_seed424344_quantitative_v1
CURRENT_ROOT=${ROOT}/eval_trajectory_step15000_20000_25000_seed424344_5gpu_strict_fix1_v1/step_025000/dev48_predicted

DEV_SPLIT=${ROOT}/protocol2128_train2000_v1/dev.json
CACHE_REPORT=${ROOT}/cache_dev64_protocol2128_views8_v1/report.json
TARGET_REPORT=${ROOT}/eval_dev64_B_scale_step4000_seed424344_v1/report.json
TARGET_MESH_ROOT=${ROOT}/eval_dev64_B_scale_step4000_seed424344_v1/targets

RECON_REPORTS=${STRICT_ROOT}/worker_00_of_05/report.json,${STRICT_ROOT}/worker_01_of_05/report.json,${STRICT_ROOT}/worker_02_of_05/report.json,${STRICT_ROOT}/worker_03_of_05/report.json,${STRICT_ROOT}/worker_04_of_05/report.json
CURRENT_REPORTS=${CURRENT_ROOT}/shard0_16_26/report.json,${CURRENT_ROOT}/shard1_26_36/report.json,${CURRENT_ROOT}/shard2_36_46/report.json,${CURRENT_ROOT}/shard3_46_55/report.json,${CURRENT_ROOT}/shard4_55_64/report.json
PAIRED_TARGET_CACHE_ROOTS=${CURRENT_ROOT}/shard0_16_26/target_mesh_cache,${CURRENT_ROOT}/shard1_26_36/target_mesh_cache,${CURRENT_ROOT}/shard2_36_46/target_mesh_cache,${CURRENT_ROOT}/shard3_46_55/target_mesh_cache,${CURRENT_ROOT}/shard4_55_64/target_mesh_cache

EXPECTED_STEP=${EXPECTED_STEP:-25000}
EXPECTED_SHA256=${EXPECTED_SHA256:-5092422900fe7d1e467684f0168aaa2cce67c754f6a48ff33d91c3772b2bcf58}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/eval_dev48_official_ss2k_slat25k_vs_strict_reconviagen_v1}
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-5000}
DRY_RUN=${DRY_RUN:-0}

for path in "$DEV_SPLIT" "$CACHE_REPORT" "$TARGET_REPORT"; do
    test -s "$path"
done
test -d "$TARGET_MESH_ROOT"

IFS=, read -r -a report_paths <<<"${RECON_REPORTS},${CURRENT_REPORTS}"
for path in "${report_paths[@]}"; do
    test -s "$path"
done
IFS=, read -r -a target_cache_paths <<<"$PAIRED_TARGET_CACHE_ROOTS"
for path in "${target_cache_paths[@]}"; do
    test -d "$path"
done

args=(
    --dev_split "$DEV_SPLIT"
    --cache_report "$CACHE_REPORT"
    --target_report "$TARGET_REPORT"
    --target_mesh_root "$TARGET_MESH_ROOT"
    --paired_target_cache_roots "$PAIRED_TARGET_CACHE_ROOTS"
    --recon_reports "$RECON_REPORTS"
    --current_reports "$CURRENT_REPORTS"
    --expected_current_step "$EXPECTED_STEP"
    --expected_current_sha256 "$EXPECTED_SHA256"
    --bootstrap_samples "$BOOTSTRAP_SAMPLES"
    --output_dir "$OUTPUT_DIR"
)

if [[ "$DRY_RUN" == "1" ]]; then
    args+=(--dry_run)
else
    test ! -e "$OUTPUT_DIR"
fi

echo "============================================================"
echo "Official Dev48 SS/SLat vs strict ReconViaGen (CPU aggregate)"
echo "============================================================"
echo "DRY_RUN=$DRY_RUN"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "No GPU inference; no Train64; no GT-support Dev evaluation."

"$PY" -u -m \
    pose_point_depth_mv.aggregate_proobjaverse_official_ss_slat_vs_reconviagen \
    "${args[@]}"
