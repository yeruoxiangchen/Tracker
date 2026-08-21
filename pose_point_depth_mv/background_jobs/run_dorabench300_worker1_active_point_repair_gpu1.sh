#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
GPU=${REPAIR_GPU:-1}
DATA=/data/zjr/dorabench_reconviagen_style_dora300_trellis40_input0_9_19_29_20260821_v1
OUT=/data/zjr/dorabench_dora300_ss30k_slat30k_step30k_metrics_seed42_trellis40_input0_9_19_29_7gpu_20260821_v1
WORKER=${OUT}/workers/worker_01
MODEL_INPUT=${WORKER}/01_model_inputs/model_input_manifest.json
INFERENCE_ROOT=${WORKER}/02_current_ss30k_slat30k
TEMPLATE=${OUT}/workers/worker_00/02_current_ss30k_slat30k/inference_manifest.json
REPAIR=${OUT}/repair_worker01_active_point_v1
PLAN=${REPAIR}/plan.json
LOG=${OUT}/logs/worker_01_active_point_repair_gpu${GPU}_v1.log

SS_REPORT=/data/zjr/proobjaverse_official_30k_heldout_dev64_ss30k_slat30k_20260818_v1/ss30k_dev64_aggregate/report.json
SLAT=/data/zjr/proobjaverse_official_30k_checkpoint_archives/ProObjaverse_30K_noVGGT_SS_SLat_numbered_checkpoints_20260818_v1/slat/checkpoints/step_030000.pt
BRIDGE=/data/zjr/proobjaverse_official_30k_heldout_dev64_ss30k_slat30k_20260818_v1/abc_r_dev64_aggregate/report.json
FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json

test -x "${PY}"
test -s "${DATA}/manifest.json"
test -s "${MODEL_INPUT}"
test -s "${TEMPLATE}"
test -s "${SS_REPORT}"
test -s "${SLAT}"
test -s "${BRIDGE}"
test -s "${FREEZE}"
mkdir -p "${REPAIR}" "${OUT}/logs"

export PYTHONPATH="${PWD}:${PWD}/ReconViaGen:${PWD}/ReconViaGen/wheels/vggt"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export ATTN_BACKEND=flash_attn
export SPCONV_ALGO=native
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1

exec > >(tee -a "${LOG}") 2>&1

echo "===== P0 register the one real model-output failure and exact repair suffix ====="
"${PY}" -u -m pose_point_depth_mv.repair_dorabench300_active_point_failure \
  plan \
  --benchmark_manifest "${DATA}/manifest.json" \
  --model_input_manifest "${MODEL_INPUT}" \
  --inference_root "${INFERENCE_ROOT}" \
  --worker_index 1 \
  --num_workers 7 \
  --seed 42 \
  --output "${PLAN}"

mapfile -t PENDING_OBJECTS < <(
  "${PY}" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); [print(x["object_key"]) for x in d["pending_objects"]]' \
    "${PLAN}"
)
OFFSET=$(
  "${PY}" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); print(d["master_position_offset"] if d["master_position_offset"] is not None else 0)' \
    "${PLAN}"
)

if (( ${#PENDING_OBJECTS[@]} > 0 )); then
  if (( ${#PENDING_OBJECTS[@]} != 10 || OFFSET != 33 )); then
    echo "ERROR: registered repair suffix changed: count=${#PENDING_OBJECTS[@]} offset=${OFFSET}" >&2
    exit 90
  fi
  OBJECT_ARGS=()
  for object_key in "${PENDING_OBJECTS[@]}"; do
    OBJECT_ARGS+=(--object "${object_key}")
  done
  echo "===== P1 reconstruct only the ten skipped suffix objects on GPU ${GPU} ====="
  CUDA_VISIBLE_DEVICES="${GPU}" \
    "${PY}" -u -m manual_mesh_reconstruction.current_model \
      --model_input_manifest "${MODEL_INPUT}" \
      --native_ss_report "${SS_REPORT}" \
      --native_slat_checkpoint "${SLAT}" \
      --expected_slat_step 30000 \
      --cross_deployment_bridge_report "${BRIDGE}" \
      --stock_slat_freeze "${FREEZE}" \
      --output_dir "${INFERENCE_ROOT}" \
      --pretrained Stable-X/trellis-vggt-v0-2 \
      --seeds 42 \
      --weights ema \
      --device cuda \
      --amp_dtype bf16 \
      --master_position_offset "${OFFSET}" \
      "${OBJECT_ARGS[@]}"
else
  echo "===== P1 repair suffix already complete; reuse ====="
fi

echo "===== P2 finalize worker 1 as 42 valid Meshes + one registered failure ====="
"${PY}" -u -m pose_point_depth_mv.repair_dorabench300_active_point_failure \
  finalize-worker \
  --benchmark_manifest "${DATA}/manifest.json" \
  --model_input_manifest "${MODEL_INPUT}" \
  --inference_root "${INFERENCE_ROOT}" \
  --worker_index 1 \
  --num_workers 7 \
  --seed 42 \
  --template_manifest "${TEMPLATE}" \
  --output "${INFERENCE_ROOT}/inference_manifest.json"

echo "===== P3 compute worker 1 metrics for all 42 valid Meshes ====="
"${PY}" -u -m pose_point_depth_mv.evaluate_dorabench300_ss30k_slat30k \
  metric-worker \
  --benchmark_manifest "${DATA}/manifest.json" \
  --inference_manifest "${INFERENCE_ROOT}/inference_manifest.json" \
  --output_dir "${WORKER}/03_metrics" \
  --worker_index 1 \
  --surface_points 100000 \
  --fscore_radius 0.1 \
  --seed 42

echo "===== P4 aggregate 299 valid surfaces + one explicit model-output failure ====="
"${PY}" -u -m pose_point_depth_mv.repair_dorabench300_active_point_failure \
  aggregate \
  --benchmark_manifest "${DATA}/manifest.json" \
  --workers_root "${OUT}/workers" \
  --output_dir "${OUT}/aggregate_failure_aware_v1" \
  --expected_workers 7 \
  --expected_objects 300 \
  --expected_failures 1 \
  --surface_points 100000 \
  --fscore_radius 0.1

echo "DORA300 FAILURE-AWARE REPAIR COMPLETE: ${OUT}/aggregate_failure_aware_v1/report.json"
