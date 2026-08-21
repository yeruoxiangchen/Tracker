#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

# The shared environment validates a three/four-GPU evaluator topology.  This
# qualitative exporter deliberately runs two independent object queues, so use
# a harmless three-GPU value while sourcing it and bind the real queues below.
export EVAL_GPUS=0,3,4
source pose_point_depth_mv/background_jobs/source_proobjaverse_30k_dev64_abc_r_env.sh

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUMM_CUDA_ARCH_LIST=${CUMM_CUDA_ARCH_LIST:-8.6}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

OUTPUT=${OUTPUT:-/home/zjr/Tracker/pose_point_depth_mv/outputs2/ProObjaverse30K_Dev64_CminusR优势最明显Top4_GT_ReconViaGen_SS30K_SLat30K_输入Mask_20260819_v1}
EXPORT_GPUS=${EXPORT_GPUS:-0,3}
EXPORT_ROW_POSITIONS=${EXPORT_ROW_POSITIONS:-0,1,2,3}
MAX_USED_MIB=${MAX_USED_MIB:-4096}
WAIT_SECONDS=${WAIT_SECONDS:-20}
CURRENT_MAX_ATTEMPTS=${CURRENT_MAX_ATTEMPTS:-4}
PREPARE_MODULE=pose_point_depth_mv.export_proobjaverse_30k_dev64_top4_advantage

IFS=, read -r -a GPU_ARRAY <<<"${EXPORT_GPUS}"
if [[ ${#GPU_ARRAY[@]} -ne 2 || ${GPU_ARRAY[0]} == "${GPU_ARRAY[1]}" ]]; then
  echo "ERROR: EXPORT_GPUS must contain exactly two distinct physical GPU ids" >&2
  exit 90
fi

mkdir -p "${OUTPUT}/logs"

echo "===== prepare frozen Top4 inputs, masks and official GT Meshes ====="
"${PY}" -u -m "${PREPARE_MODULE}" prepare --output "${OUTPUT}"

mapfile -t OBJECT_ROWS < <(
  "${PY}" - "${OUTPUT}/selection.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
assert r["passed"] is True and len(r["objects"]) == 4
for row in r["objects"]:
    print(f'{int(row["rank"])}\t{int(row["object_index"])}\t{row["object_uid"]}\t{row["directory"]}')
PY
)
if [[ ${#OBJECT_ROWS[@]} -ne 4 ]]; then
  echo "ERROR: frozen selection did not contain four rows" >&2
  exit 91
fi

gpu_used_mib() {
  local requested=$1
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F, -v requested="${requested}" '$1 + 0 == requested {gsub(/ /, "", $2); print $2}'
}

echo "===== wait for two usable CUDA devices ====="
while true; do
  ready=1
  if [[ ! -e /dev/nvidiactl ]] || ! nvidia-smi -L >/dev/null 2>&1; then
    ready=0
    echo "[$(date -Is)] CUDA device nodes/driver query unavailable; waiting ${WAIT_SECONDS}s"
  else
    for gpu in "${GPU_ARRAY[@]}"; do
      used=$(gpu_used_mib "${gpu}")
      if [[ -z ${used} || ${used} -gt ${MAX_USED_MIB} ]]; then
        ready=0
        echo "[$(date -Is)] GPU${gpu} used_mib=${used:-unknown}; require <=${MAX_USED_MIB}; waiting"
      fi
    done
  fi
  if [[ ${ready} -eq 1 ]]; then
    break
  fi
  sleep "${WAIT_SECONDS}"
done
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits

TARGETS=${ABC_ROOT}/shard0_0_16/target_mesh_cache,${ABC_ROOT}/shard1_16_32/target_mesh_cache,${ABC_ROOT}/shard2_32_48/target_mesh_cache,${ABC_ROOT}/shard3_48_64/target_mesh_cache

run_object() {
  local row=$1
  local gpu=$2
  local rank index uid object_dir
  IFS=$'\t' read -r rank index uid object_dir <<<"${row}"
  local rank_label
  rank_label=$(printf '%02d' "${rank}")
  local current_out=${object_dir}/runtime/current_seed42
  local recon_out=${object_dir}/runtime/reconviagen_all_seeds
  local log=${OUTPUT}/logs/rank_${rank_label}_${uid:0:12}_gpu${gpu}.log

  {
    echo "============================================================"
    echo "[$(date -Is)] rank=${rank} index=${index} uid=${uid} GPU=${gpu}"
    echo "===== replay SS30K + SLat30K seed42 and export Mesh ====="
    # A previous process may have preserved a fail-closed CUDA marker before
    # exiting.  Keeping that marker in its live location intentionally makes
    # the evaluator materialize a failed branch, so archive it before a true
    # identical retry.  Never do this to an already finalized worker report.
    if [[ ! -s ${current_out}/report.json ]]; then
      mapfile -t prior_failure_markers < <(
        find "${current_out}/mesh_pairs" -type f \
          -path '*/recorded_cuda_branch_failures/*.json' 2>/dev/null | sort
      )
      if [[ ${#prior_failure_markers[@]} -gt 0 ]]; then
        failure_archive=${current_out}/recorded_failed_attempts
        mkdir -p "${failure_archive}"
        for marker in "${prior_failure_markers[@]}"; do
          marker_pair=$(basename "$(dirname "$(dirname "${marker}")")")
          marker_target=${failure_archive}/attempt_0_${marker_pair}_$(basename "${marker}")
          test ! -e "${marker_target}"
          mv "${marker}" "${marker_target}"
          echo "preserved prior retryable CUDA topology marker: ${marker_target}"
        done
      fi
    fi
    current_attempt=1
    while [[ ! -s ${current_out}/report.json ]]; do
      set +e
      CUDA_VISIBLE_DEVICES=${gpu} "${PY}" -u -m \
        pose_point_depth_mv.evaluate_proobjaverse_official_native_ss_stock_slat \
        worker \
        --cache_manifest "${DEV_COMPACT}/slat_manifest.json" \
        --lifting_cache_manifest "${DEV_COMPACT}/lifting_manifest.json" \
        --native_ss_report "${SS_DEV_AGGREGATE}/report.json" \
        --allow_native_ss_science_failed \
        --stock_slat_freeze "${STOCK_SLAT_FREEZE}" \
        --trained_slat_checkpoint "${SLAT30K_CHECKPOINT}" \
        --trained_slat_weights ema \
        --expected_trained_slat_step 30000 \
        --expected_checkpoint_training_membership all_disjoint \
        --output_dir "${current_out}" \
        --weights ema \
        --joint_seeds 42 \
        --object_start "${index}" \
        --object_end "$((index + 1))" \
        --surface_samples 20000 \
        --amp_dtype bf16 \
        --save_meshes \
        --resume
      current_rc=$?
      set -e
      if [[ ${current_rc} -eq 0 ]]; then
        break
      fi

      mapfile -t failure_markers < <(
        find "${current_out}/mesh_pairs" -type f \
          -path '*/recorded_cuda_branch_failures/*.json' 2>/dev/null | sort
      )
      if [[ ${#failure_markers[@]} -eq 0 || ${current_attempt} -ge ${CURRENT_MAX_ATTEMPTS} ]]; then
        echo "ERROR: current endpoint rank=${rank} failed rc=${current_rc}; " \
             "retryable_markers=${#failure_markers[@]} attempt=${current_attempt}" >&2
        return "${current_rc}"
      fi
      failure_archive=${current_out}/recorded_failed_attempts
      mkdir -p "${failure_archive}"
      for marker in "${failure_markers[@]}"; do
        marker_pair=$(basename "$(dirname "$(dirname "${marker}")")")
        marker_target=${failure_archive}/attempt_${current_attempt}_${marker_pair}_$(basename "${marker}")
        test ! -e "${marker_target}"
        mv "${marker}" "${marker_target}"
        echo "preserved retryable CUDA topology marker: ${marker_target}"
      done
      current_attempt=$((current_attempt + 1))
      echo "retrying identical current endpoint in a fresh CUDA process: " \
           "rank=${rank} attempt=${current_attempt}/${CURRENT_MAX_ATTEMPTS}"
    done
    if [[ -s ${current_out}/report.json ]]; then
      echo "reuse finalized current endpoint report: ${current_out}/report.json"
    fi

    echo "===== replay strict ReconViaGen exact 42/43/44 contract and export Meshes ====="
    CUDA_VISIBLE_DEVICES=${gpu} "${PY}" -u -m \
      pose_point_depth_mv.evaluate_proobjaverse_official_reconviagen \
      worker \
      --dev_split "${RELOCATED_PROTOCOL_DIR}/dev.json" \
      --cache_report "${DEV_COMPACT}/report.json" \
      --paired_target_cache_roots "${TARGETS}" \
      --paired_targets_cover_all_objects \
      --output_dir "${recon_out}" \
      --pretrained Stable-X/trellis-vggt-v0-2 \
      --seeds 42,43,44 \
      --device cuda \
      --ss_steps 30 \
      --ss_guidance 7.5 \
      --ss_guidance_rescale 0.7 \
      --ss_rescale_t 5.0 \
      --slat_steps 12 \
      --slat_guidance 7.5 \
      --slat_guidance_rescale 0.5 \
      --slat_rescale_t 3.0 \
      --multiimage_algo multidiffusion \
      --surface_samples 20000 \
      --worker_index "${index}" \
      --num_workers 64 \
      --save_meshes \
      --resume
    echo "[$(date -Is)] rank=${rank} complete"
  } >>"${log}" 2>&1
}

run_queue() {
  local gpu=$1
  shift
  local position
  for position in "$@"; do
    run_object "${OBJECT_ROWS[${position}]}" "${gpu}"
  done
}

echo "===== launch two independent GPU queues ====="
IFS=, read -r -a POSITION_ARRAY <<<"${EXPORT_ROW_POSITIONS}"
QUEUE0_POSITIONS=()
QUEUE1_POSITIONS=()
for position_index in "${!POSITION_ARRAY[@]}"; do
  position=${POSITION_ARRAY[${position_index}]}
  if [[ ! ${position} =~ ^[0-3]$ ]]; then
    echo "ERROR: EXPORT_ROW_POSITIONS entries must be in 0..3" >&2
    exit 92
  fi
  if ((position_index % 2 == 0)); then
    QUEUE0_POSITIONS+=("${position}")
  else
    QUEUE1_POSITIONS+=("${position}")
  fi
done
run_queue "${GPU_ARRAY[0]}" "${QUEUE0_POSITIONS[@]}" &
QUEUE0_PID=$!
run_queue "${GPU_ARRAY[1]}" "${QUEUE1_POSITIONS[@]}" &
QUEUE1_PID=$!
echo "queue0 pid=${QUEUE0_PID} GPU=${GPU_ARRAY[0]} positions=${QUEUE0_POSITIONS[*]:-none}"
echo "queue1 pid=${QUEUE1_PID} GPU=${GPU_ARRAY[1]} positions=${QUEUE1_POSITIONS[*]:-none}"

rc=0
wait "${QUEUE0_PID}" || rc=$?
if wait "${QUEUE1_PID}"; then
  other_rc=0
else
  other_rc=$?
  if [[ ${rc} -eq 0 ]]; then
    rc=${other_rc}
  fi
fi
if [[ ${rc} -ne 0 ]]; then
  echo "ERROR: one Top4 replay queue failed rc=${rc}; inspect ${OUTPUT}/logs" >&2
  exit "${rc}"
fi

echo "===== verify reproduced metrics and finalize user-facing package ====="
"${PY}" -u -m "${PREPARE_MODULE}" finalize --output "${OUTPUT}"

echo "============================================================"
echo "TOP4 ADVANTAGE EXPORT COMPLETE"
echo "OUTPUT=${OUTPUT}"
echo "REPORT=${OUTPUT}/report.json"
echo "============================================================"
