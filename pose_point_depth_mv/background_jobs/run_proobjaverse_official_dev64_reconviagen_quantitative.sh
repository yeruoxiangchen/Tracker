#!/usr/bin/env bash
set -euo pipefail

# Strict full ReconViaGen baseline on official ProObjaverse Dev64.
# No predicted OBJ/GLB and no preview rendering are written; only metric JSON.

TRACKER=${TRACKER:-/home/zjr/Tracker}
PY=${PY:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
ROOT=${ROOT:-/data/zjr/proobjaverse_official_slat_train2000_20260813_v1}
EVAL_GPUS=${EVAL_GPUS:-3,4,5,6,7}
RESUME=${RESUME:-1}
DRY_RUN=${DRY_RUN:-0}

DEV_SPLIT=${DEV_SPLIT:-${ROOT}/protocol2128_train2000_v1/dev.json}
CACHE_REPORT=${CACHE_REPORT:-${ROOT}/cache_dev64_protocol2128_views8_v1/report.json}
TARGET_REPORT=${TARGET_REPORT:-${ROOT}/eval_dev64_B_scale_step4000_seed424344_v1/report.json}
TARGET_MESH_ROOT=${TARGET_MESH_ROOT:-${ROOT}/eval_dev64_B_scale_step4000_seed424344_v1/targets}

OUT=${OUT:-${ROOT}/eval_dev64_reconviagen_original_seed424344_quantitative_v1}
FINAL=${FINAL:-${OUT}/aggregate_v1}

CURRENT_ROOT=${CURRENT_ROOT:-${ROOT}/eval_trajectory_step15000_20000_25000_seed424344_5gpu_strict_fix1_v1/step_025000/dev48_predicted}
CURRENT_REPORTS=${CURRENT_REPORTS:-${CURRENT_ROOT}/shard0_16_26/report.json,${CURRENT_ROOT}/shard1_26_36/report.json,${CURRENT_ROOT}/shard2_36_46/report.json,${CURRENT_ROOT}/shard3_46_55/report.json,${CURRENT_ROOT}/shard4_55_64/report.json}
PAIRED_TARGET_CACHE_ROOTS=${PAIRED_TARGET_CACHE_ROOTS:-${CURRENT_ROOT}/shard0_16_26/target_mesh_cache,${CURRENT_ROOT}/shard1_26_36/target_mesh_cache,${CURRENT_ROOT}/shard2_36_46/target_mesh_cache,${CURRENT_ROOT}/shard3_46_55/target_mesh_cache,${CURRENT_ROOT}/shard4_55_64/target_mesh_cache}
CURRENT_STEP=${CURRENT_STEP:-25000}
CURRENT_SHA256=${CURRENT_SHA256:-5092422900fe7d1e467684f0168aaa2cce67c754f6a48ff33d91c3772b2bcf58}

export PYTHONPATH="${TRACKER}:${TRACKER}/ReconViaGen:${TRACKER}/ReconViaGen/wheels/vggt${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}
export SPCONV_ALGO=${SPCONV_ALGO:-native}
export ATTN_BACKEND=${ATTN_BACKEND:-flash_attn}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

cd "$TRACKER"

IFS=, read -r -a GPUS <<<"$EVAL_GPUS"
WORKERS=${#GPUS[@]}
if (( WORKERS < 1 )); then
    echo "ERROR: EVAL_GPUS is empty" >&2
    exit 90
fi

for path in "$DEV_SPLIT" "$CACHE_REPORT" "$TARGET_REPORT"; do
    test -s "$path"
done
test -d "$TARGET_MESH_ROOT"
IFS=, read -r -a CURRENT_ARRAY <<<"$CURRENT_REPORTS"
for path in "${CURRENT_ARRAY[@]}"; do
    test -s "$path"
done
IFS=, read -r -a TARGET_CACHE_ARRAY <<<"$PAIRED_TARGET_CACHE_ROOTS"
for path in "${TARGET_CACHE_ARRAY[@]}"; do
    test -d "$path"
done

COMMON=(
    --dev_split "$DEV_SPLIT"
    --cache_report "$CACHE_REPORT"
    --target_report "$TARGET_REPORT"
    --target_mesh_root "$TARGET_MESH_ROOT"
    --paired_target_cache_roots "$PAIRED_TARGET_CACHE_ROOTS"
)

echo "============================================================"
echo "Official Dev64 strict ReconViaGen quantitative comparison"
echo "============================================================"
echo "GPUs=$EVAL_GPUS workers=$WORKERS"
echo "OUT=$OUT"
echo "pipeline=VGGT -> Stock SS -> Stock SLat -> Stock Mesh decoder"
echo "predicted Mesh files/render previews=disabled"

echo
echo "===== read-only contract preflight ====="
"$PY" -u -m pose_point_depth_mv.evaluate_proobjaverse_official_reconviagen \
    worker \
    "${COMMON[@]}" \
    --output_dir "$OUT/preflight_unused" \
    --worker_index 0 \
    --num_workers 1 \
    --dry_run

if [ "$DRY_RUN" = 1 ]; then
    echo "DRY RUN PASS; no GPU worker was launched."
    exit 0
fi

mkdir -p "$OUT/logs"
if [ -e "$FINAL" ]; then
    echo "ERROR: aggregate output already exists: $FINAL" >&2
    echo "Keep it as evidence or choose a new OUT path." >&2
    exit 91
fi

echo
echo "===== launch workers ====="
pids=()
for ((worker=0; worker<WORKERS; worker++)); do
    gpu=${GPUS[$worker]}
    worker_out=$(printf '%s/worker_%02d_of_%02d' "$OUT" "$worker" "$WORKERS")
    worker_log=$(printf '%s/logs/worker_%02d_gpu%s.log' "$OUT" "$worker" "$gpu")
    resume_args=()
    if [ "$RESUME" = 1 ]; then
        resume_args+=(--resume)
    fi
    CUDA_VISIBLE_DEVICES="$gpu" \
        "$PY" -u -m pose_point_depth_mv.evaluate_proobjaverse_official_reconviagen \
        worker \
        "${COMMON[@]}" \
        --output_dir "$worker_out" \
        --worker_index "$worker" \
        --num_workers "$WORKERS" \
        --seeds 42,43,44 \
        --device cuda:0 \
        --low_vram \
        --surface_samples 20000 \
        "${resume_args[@]}" \
        >"$worker_log" 2>&1 &
    pids+=("$!")
    echo "worker=$worker gpu=$gpu pid=${pids[-1]} log=$worker_log"
done

failed=0
for ((worker=0; worker<WORKERS; worker++)); do
    if ! wait "${pids[$worker]}"; then
        echo "ERROR: worker $worker failed" >&2
        failed=1
    fi
done
if (( failed != 0 )); then
    echo "At least one worker failed. Outputs are preserved; rerun with RESUME=1." >&2
    exit 92
fi

reports=()
for ((worker=0; worker<WORKERS; worker++)); do
    report=$(printf '%s/worker_%02d_of_%02d/report.json' "$OUT" "$worker" "$WORKERS")
    test -s "$report"
    reports+=("$report")
done
RECON_REPORTS=$(IFS=,; echo "${reports[*]}")

echo
echo "===== aggregate + paired held-out comparison ====="
"$PY" -u -m pose_point_depth_mv.evaluate_proobjaverse_official_reconviagen \
    aggregate \
    "${COMMON[@]}" \
    --recon_reports "$RECON_REPORTS" \
    --current_reports "$CURRENT_REPORTS" \
    --expected_current_step "$CURRENT_STEP" \
    --expected_current_sha256 "$CURRENT_SHA256" \
    --bootstrap_samples 5000 \
    --output_dir "$FINAL"

echo
cat "$FINAL/summary.txt"
echo
echo "COMPLETE: $FINAL/report.json"
