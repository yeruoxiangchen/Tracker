#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
DATA=${OMNI200_ROOT:-/data/zjr/omniobject3d_reconviagen_style_omni200_20cat_render4_20260821_v1}
BENCHMARK=${DATA}/manifest.json
BUILD_REPORT=${DATA}/report.json
OUT=${OUTPUT_ROOT:-/data/zjr/omniobject3d_omni200_ss30k_slat30k_step30k_metrics_seed42_20260821_v1}
GPUS_CSV=${EVAL_GPUS:-0,1,2,3}
SEED=${EVAL_SEED:-42}
SURFACE_POINTS=${SURFACE_POINTS:-100000}
FSCORE_RADIUS=${FSCORE_RADIUS:-0.1}

SS_REPORT=/data/zjr/proobjaverse_official_30k_heldout_dev64_ss30k_slat30k_20260818_v1/ss30k_dev64_aggregate/report.json
SLAT=/data/zjr/proobjaverse_official_30k_checkpoint_archives/ProObjaverse_30K_noVGGT_SS_SLat_numbered_checkpoints_20260818_v1/slat/checkpoints/step_030000.pt
BRIDGE=/data/zjr/proobjaverse_official_30k_heldout_dev64_ss30k_slat30k_20260818_v1/abc_r_dev64_aggregate/report.json
FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json

test -x "${PY}"
test -s "${BUILD_REPORT}"
test -s "${BENCHMARK}"
test -s "${SS_REPORT}"
test -s "${SLAT}"
test -s "${BRIDGE}"
test -s "${FREEZE}"

IFS=, read -r -a GPU_ARRAY <<<"${GPUS_CSV}"
WORKERS=${#GPU_ARRAY[@]}
if (( WORKERS < 1 )); then
  echo "ERROR: EVAL_GPUS is empty" >&2
  exit 90
fi

mkdir -p "${OUT}/logs" "${OUT}/workers"
export PYTHONPATH="${PWD}:${PWD}/ReconViaGen"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export ATTN_BACKEND=flash_attn
export SPCONV_ALGO=native
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

echo "===== P0 exact frozen model-O runtime preparation (CPU; no GT read) ====="
"${PY}" -u -m pose_point_depth_mv.evaluate_omni200_ss30k_slat30k \
  prepare \
  --benchmark_manifest "${BENCHMARK}" \
  --output_dir "${OUT}/00_exact_model_o_runtime" \
  --expected_objects 200 \
  --resume

RUNTIME=${OUT}/00_exact_model_o_runtime/runtime_input_manifest.json
test -s "${RUNTIME}"

echo "===== P1 launch ${WORKERS} SS30K+SLat30K-only workers on GPUs ${GPUS_CSV} ====="
pids=()
for worker in "${!GPU_ARRAY[@]}"; do
  gpu=${GPU_ARRAY[$worker]}
  worker_name=$(printf 'worker_%02d' "${worker}")
  worker_root=${OUT}/workers/${worker_name}
  log=${OUT}/logs/${worker_name}_gpu${gpu}.log
  mkdir -p "${worker_root}"

  mapfile -t object_keys < <(
    "${PY}" - "${BENCHMARK}" "${worker}" "${WORKERS}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
worker, workers = map(int, sys.argv[2:])
for index, row in enumerate(p["objects"]):
    if index % workers == worker:
        print(f"{row['category']}:{row['uid']}")
PY
  )
  if (( ${#object_keys[@]} < 1 )); then
    echo "ERROR: worker ${worker} has no objects" >&2
    exit 91
  fi
  object_args=()
  for key in "${object_keys[@]}"; do
    object_args+=(--object "${key}")
  done

  (
    set -euo pipefail
    echo "[worker ${worker}] objects=${#object_keys[@]} physical_gpu=${gpu}"
    echo "[worker ${worker}] DINO-only input encoding"
    CUDA_VISIBLE_DEVICES="${gpu}" \
      "${PY}" -u -m manual_mesh_reconstruction.model_inputs \
        --runtime_input_manifest "${RUNTIME}" \
        --output_dir "${worker_root}/01_model_inputs" \
        --pretrained Stable-X/trellis-vggt-v0-2 \
        --device cuda \
        --resume \
        "${object_args[@]}"

    echo "[worker ${worker}] SS30K + SLat30K + Stock Mesh decoder"
    CUDA_VISIBLE_DEVICES="${gpu}" \
      "${PY}" -u -m manual_mesh_reconstruction.current_model \
        --model_input_manifest "${worker_root}/01_model_inputs/model_input_manifest.json" \
        --native_ss_report "${SS_REPORT}" \
        --native_slat_checkpoint "${SLAT}" \
        --expected_slat_step 30000 \
        --cross_deployment_bridge_report "${BRIDGE}" \
        --stock_slat_freeze "${FREEZE}" \
        --output_dir "${worker_root}/02_current_ss30k_slat30k" \
        --pretrained Stable-X/trellis-vggt-v0-2 \
        --seeds "${SEED}" \
        --weights ema \
        --device cuda \
        --amp_dtype bf16

    echo "[worker ${worker}] CD/F-score metric"
    "${PY}" -u -m pose_point_depth_mv.evaluate_omni200_ss30k_slat30k \
      metric-worker \
      --benchmark_manifest "${BENCHMARK}" \
      --inference_manifest "${worker_root}/02_current_ss30k_slat30k/inference_manifest.json" \
      --output_dir "${worker_root}/03_metrics" \
      --worker_index "${worker}" \
      --surface_points "${SURFACE_POINTS}" \
      --fscore_radius "${FSCORE_RADIUS}" \
      --seed "${SEED}"
    echo "[worker ${worker}] COMPLETE"
  ) >"${log}" 2>&1 &
  pids+=("$!")
  echo "worker=${worker} gpu=${gpu} pid=$! objects=${#object_keys[@]} log=${log}"
done

failed=0
for worker in "${!pids[@]}"; do
  if ! wait "${pids[$worker]}"; then
    echo "ERROR: worker ${worker} failed; outputs are resumable" >&2
    failed=1
  fi
done
if (( failed != 0 )); then
  exit 92
fi

echo "===== P2 aggregate 200 objects ====="
"${PY}" -u -m pose_point_depth_mv.evaluate_omni200_ss30k_slat30k \
  aggregate \
  --benchmark_manifest "${BENCHMARK}" \
  --workers_root "${OUT}/workers" \
  --output_dir "${OUT}/aggregate_v1" \
  --expected_workers "${WORKERS}" \
  --expected_objects 200 \
  --surface_points "${SURFACE_POINTS}" \
  --fscore_radius "${FSCORE_RADIUS}"

echo "OMNI200 SS30K+SLAT30K METRICS COMPLETE: ${OUT}/aggregate_v1/report.json"

