#!/usr/bin/env bash
set -euo pipefail

# Two-/four-GPU compatibility trajectory for the F39 29,861-object checkpoints.
# The reused legacy Train64/Dev64 objects are all in the 30K checkpoint's
# training UID set.  Every worker therefore performs an explicit checkpoint UID
# membership audit and the outputs must never be described as held-out results.

source /home/zjr/anaconda3/etc/profile.d/conda.sh
conda activate reconviagen

PROJECT=${PROJECT:-/home/zjr/Tracker}
PY=${PY:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
EVAL_GPUS=${EVAL_GPUS:-4,5,6,7}
STEPS=${STEPS:-10000,30000,60000}
EVAL_GROUPS=${EVAL_GROUPS:-train64_gt,legacy_dev64_gt_training_overlap,legacy_dev48_predicted_training_overlap}
WAIT_SECONDS=${WAIT_SECONDS:-20}
POLL_SECONDS=${POLL_SECONDS:-20}
MAX_IDLE_MEMORY_MIB=${MAX_IDLE_MEMORY_MIB:-1024}
PRESERVE_INTERRUPTED_OUTPUTS=${PRESERVE_INTERRUPTED_OUTPUTS:-0}
SS_COORD_REUSE_ROOT=${SS_COORD_REUSE_ROOT:-}
SS_COORD_REUSE_SOURCE_STEP=${SS_COORD_REUSE_SOURCE_STEP:-10000}
FROZEN_STOCK_FLOOR_ROOT=${FROZEN_STOCK_FLOOR_ROOT:-}
SKIP_GPU_SETUP=${SKIP_GPU_SETUP:-0}

CKPT_ROOT=${CKPT_ROOT:-/data/zjr/proobjaverse_official_slat_train29861_20260817_v1/F39_fresh_step150000_seed42_8gpu_strict_fix1_warmup3000_v1/checkpoints}
LEGACY_ROOT=${LEGACY_ROOT:-/data/zjr/proobjaverse_official_slat_train2000_20260813_v1}
SS_ROOT=${SS_ROOT:-/data/zjr/proobjaverse_official_native_ss_train2000_20260815_v1}
OUTPUT_ROOT=${OUTPUT_ROOT:-}
MASTER_LOG=${MASTER_LOG:-}
SUMMARY_PATH=${SUMMARY_PATH:-}

TRAIN_CACHE=${TRAIN_CACHE:-${LEGACY_ROOT}/cache_train2000_protocol2128_views8_v1}
DEV_CACHE=${DEV_CACHE:-${LEGACY_ROOT}/cache_dev64_protocol2128_views8_v1}
TRAINING_SS_REPORT=${TRAINING_SS_REPORT:-/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1/ss_eval_synthetic_dev32_fixedcfg3_count125_v3/report.json}
PREDICTED_SS_REPORT=${PREDICTED_SS_REPORT:-${SS_ROOT}/dev64_step2000_eval16_64_seed424344_6gpu_v1/aggregate_v1/report.json}
STOCK_FREEZE=${STOCK_FREEZE:-/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json}

STRICT_ROOT=${STRICT_ROOT:-${LEGACY_ROOT}/eval_dev64_reconviagen_original_seed424344_quantitative_v1}
DEV_SPLIT=${DEV_SPLIT:-${LEGACY_ROOT}/protocol2128_train2000_v1/dev.json}
CACHE_REPORT=${CACHE_REPORT:-${DEV_CACHE}/report.json}
TARGET_REPORT=${TARGET_REPORT:-${LEGACY_ROOT}/eval_dev64_B_scale_step4000_seed424344_v1/report.json}
TARGET_MESH_ROOT=${TARGET_MESH_ROOT:-${LEGACY_ROOT}/eval_dev64_B_scale_step4000_seed424344_v1/targets}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export ATTN_BACKEND=flash_attn
export SPCONV_ALGO=native
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${PROJECT}:${PROJECT}/ReconViaGen:${PROJECT}/ReconViaGen/wheels/vggt${PYTHONPATH:+:${PYTHONPATH}}"

source "${PROJECT}/pose_point_depth_mv/background_jobs/eval_gpu_reservation.sh"

cd "${PROJECT}"

IFS=, read -r -a GPU_ARRAY <<<"${EVAL_GPUS}"
EVAL_GPU_COUNT=${#GPU_ARRAY[@]}
if (( EVAL_GPU_COUNT != 2 && EVAL_GPU_COUNT != 4 )); then
  echo "ERROR: EVAL_GPUS must contain exactly two or four GPUs" >&2
  exit 90
fi
EVAL_TAG=${EVAL_TAG:-${EVAL_GPU_COUNT}gpu}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/zjr/proobjaverse_official_slat_train29861_20260817_v1/eval_legacy_protocol2128_step10k30k60k_seed424344_${EVAL_TAG}_v1}
MASTER_LOG=${MASTER_LOG:-/data/zjr/proobjaverse_official_slat_train29861_20260817_v1/logs/eval_legacy_protocol2128_step10k30k60k_${EVAL_TAG}_v1.log}
SUMMARY_PATH=${SUMMARY_PATH:-${OUTPUT_ROOT}/trajectory_summary.json}
EVAL_GPU_RESERVATION_ROOT=${EVAL_GPU_RESERVATION_ROOT:-${OUTPUT_ROOT}/runtime}

cleanup() {
  local rc=$?
  trap - EXIT
  stop_eval_gpu_reservations
  exit "${rc}"
}
trap cleanup EXIT
IFS=, read -r -a STEP_ARRAY <<<"${STEPS}"
if (( ${#STEP_ARRAY[@]} == 0 )); then
  echo "ERROR: STEPS must not be empty" >&2
  exit 91
fi
previous_step=0
for step in "${STEP_ARRAY[@]}"; do
  case "${step}" in
    10000|30000|60000|70000) ;;
    *) echo "ERROR: unregistered checkpoint step=${step}" >&2; exit 91 ;;
  esac
  if (( step <= previous_step )); then
    echo "ERROR: STEPS must be unique and strictly increasing: ${STEPS}" >&2
    exit 91
  fi
  previous_step=${step}
done
IFS=, read -r -a GROUP_ARRAY <<<"${EVAL_GROUPS}"
if (( ${#GROUP_ARRAY[@]} == 0 )); then
  echo "ERROR: EVAL_GROUPS must not be empty" >&2
  exit 91
fi
declare -A SEEN_GROUPS=()
for group in "${GROUP_ARRAY[@]}"; do
  case "${group}" in
    train64_gt|legacy_dev64_gt_training_overlap|legacy_dev48_predicted_training_overlap) ;;
    *) echo "ERROR: unregistered evaluation group=${group}" >&2; exit 91 ;;
  esac
  if [[ -n "${SEEN_GROUPS[$group]:-}" ]]; then
    echo "ERROR: EVAL_GROUPS contains duplicate group=${group}" >&2
    exit 91
  fi
  SEEN_GROUPS[$group]=1
done

for required in \
  "${PY}" \
  "${TRAIN_CACHE}/slat_manifest.json" \
  "${TRAIN_CACHE}/lifting_manifest.json" \
  "${DEV_CACHE}/slat_manifest.json" \
  "${DEV_CACHE}/lifting_manifest.json" \
  "${TRAINING_SS_REPORT}" \
  "${PREDICTED_SS_REPORT}" \
  "${STOCK_FREEZE}" \
  "${DEV_SPLIT}" \
  "${CACHE_REPORT}" \
  "${TARGET_REPORT}"
do
  test -s "${required}" || { echo "ERROR: missing ${required}" >&2; exit 92; }
done
test -d "${TARGET_MESH_ROOT}"

RECON_REPORTS=""
for worker in 00 01 02 03 04; do
  report="${STRICT_ROOT}/worker_${worker}_of_05/report.json"
  test -s "${report}" || { echo "ERROR: missing strict R report ${report}" >&2; exit 93; }
  RECON_REPORTS+="${RECON_REPORTS:+,}${report}"
done
FROZEN_TARGET_RECON_REPORTS=${FROZEN_TARGET_RECON_REPORTS:-${RECON_REPORTS}}

mkdir -p "$(dirname "${MASTER_LOG}")"

log_master() {
  echo "$*" | tee -a "${MASTER_LOG}"
}

preserve_interrupted_dir() {
  local path=$1 label=$2 timestamp archived
  timestamp=$(date -u +%Y%m%dT%H%M%SZ)
  archived="${path}.interrupted_${timestamp}"
  test ! -e "${archived}" || {
    echo "ERROR: interrupted-output archive already exists: ${archived}" >&2
    exit 96
  }
  mv "${path}" "${archived}"
  log_master "preserved interrupted ${label}: ${path} -> ${archived}"
}

wait_for_selected_gpus() {
  local idle gpu used
  while :; do
    idle=1
    for gpu in "${GPU_ARRAY[@]}"; do
      used=$(nvidia-smi -i "${gpu}" --query-gpu=memory.used \
        --format=csv,noheader,nounits | awk 'NR == 1 {print int($1)}')
      (( used <= MAX_IDLE_MEMORY_MIB )) || idle=0
    done
    (( idle == 1 )) && return 0
    log_master "[$(date -u -Is)] waiting for selected eval GPUs=${EVAL_GPUS}"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits | tee -a "${MASTER_LOG}" || true
    sleep "${POLL_SECONDS}"
  done
}

wait_for_checkpoint() {
  local step=$1
  local checkpoint sidecar manifest filename expected actual matches
  checkpoint=$(printf '%s/step_%06d.pt' "${CKPT_ROOT}" "${step}")
  sidecar="${checkpoint}.sha256"
  manifest="${CKPT_ROOT}/SHA256SUMS.txt"
  filename=$(basename "${checkpoint}")
  while [[ ! -s "${checkpoint}" || ( ! -s "${sidecar}" && ! -s "${manifest}" ) ]]; do
    log_master "waiting for checkpoint+SHA evidence: ${checkpoint} ($(date -Is))" >&2
    sleep "${WAIT_SECONDS}"
  done

  if [[ -s "${sidecar}" ]]; then
    expected=$(awk 'NR==1 {print $1}' "${sidecar}")
  else
    # The versioned F39 transfer bundle uses one canonical SHA256SUMS.txt
    # instead of four per-checkpoint sidecars.  Bind the requested basename
    # exactly once; never accept a prefix/substring match.
    matches=$(awk -v target="${filename}" '
      {
        name=$2
        sub(/^\*/, "", name)
        sub(/^\.\//, "", name)
        if (name == target) print $1
      }
    ' "${manifest}")
    if [[ $(printf '%s\n' "${matches}" | awk 'NF {n++} END {print n+0}') -ne 1 ]]; then
      echo "ERROR: expected exactly one ${filename} entry in ${manifest}" >&2
      exit 94
    fi
    expected=${matches}
  fi
  [[ "${expected}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "ERROR: malformed SHA evidence for ${checkpoint}" >&2
    exit 94
  }
  actual=$(sha256sum "${checkpoint}" | awk '{print $1}')
  if [[ "${actual}" != "${expected}" ]]; then
    echo "ERROR: checkpoint SHA mismatch: ${checkpoint}" >&2
    echo "expected=${expected}" >&2
    echo "actual=${actual}" >&2
    exit 95
  fi
  printf '%s\n' "${checkpoint}"
}

gpu_env_prefix() {
  local gpu=$1
  printf 'CUDA_VISIBLE_DEVICES=%q' "${gpu}"
}

run_wave() {
  local step=$1 group=$2 checkpoint=$3
  local cache range_start range_count shard
  if [[ "${group}" == "train64_gt" ]]; then
    cache="${TRAIN_CACHE}"
    range_start=0
    range_count=64
  elif [[ "${group}" == "legacy_dev64_gt_training_overlap" ]]; then
    cache="${DEV_CACHE}"
    range_start=0
    range_count=64
  elif [[ "${group}" == "legacy_dev48_predicted_training_overlap" ]]; then
    cache="${DEV_CACHE}"
    range_start=16
    range_count=48
  else
    echo "ERROR: unknown group=${group}" >&2
    exit 96
  fi

  local group_root="${OUTPUT_ROOT}/$(printf 'step_%06d' "${step}")/${group}"
  local log_root="${OUTPUT_ROOT}/logs"
  mkdir -p "${log_root}"
  local pids=() shard_dirs=() statuses=()

  log_master "===== wave step=${step} group=${group} GPUs=${EVAL_GPUS} ====="
  for ((shard=0; shard<EVAL_GPU_COUNT; shard++)); do
    local start=$((range_start + range_count * shard / EVAL_GPU_COUNT))
    local end=$((range_start + range_count * (shard + 1) / EVAL_GPU_COUNT))
    local gpu=${GPU_ARRAY[$shard]}
    local out="${group_root}/shard${shard}_${start}_${end}"
    local report="${out}/report.json"
    local log="${log_root}/step_${step}_${group}_shard${shard}_gpu${gpu}.log"
    shard_dirs+=("${out}")
    if [[ -s "${report}" ]]; then
      log_master "reuse finalized shard=${shard} report=${report}"
      pids+=("")
      continue
    fi
    if [[ "${group}" != "legacy_dev48_predicted_training_overlap" && -e "${out}" ]]; then
      if (( PRESERVE_INTERRUPTED_OUTPUTS == 1 )); then
        preserve_interrupted_dir "${out}" "GT shard"
      else
        echo "ERROR: partial non-resumable GT output exists: ${out}" >&2
        exit 97
      fi
    fi
    mkdir -p "$(dirname "${out}")"
    if [[ "${group}" == "legacy_dev48_predicted_training_overlap" ]]; then
      if [[ -n "${SS_COORD_REUSE_ROOT}" && ! -s "${out}/report.json" ]]; then
        local reuse_source reuse_source_checkpoint reuse_log
        reuse_source="${SS_COORD_REUSE_ROOT}/shard${shard}_${start}_${end}"
        reuse_source_checkpoint=$(printf '%s/step_%06d.pt' \
          "${CKPT_ROOT}" "${SS_COORD_REUSE_SOURCE_STEP}")
        reuse_log="${log_root}/step_${step}_${group}_shard${shard}_ss_coord_reuse.log"
        test -s "${reuse_source}/report.json" || {
          echo "ERROR: missing SS-coordinate reuse source ${reuse_source}/report.json" >&2
          exit 97
        }
        "${PY}" -u -m official_ss_with_vggt_perf_v1.reuse_endpoint_artifacts \
          --source_worker "${reuse_source}" \
          --target_worker "${out}" \
          --source_step "${SS_COORD_REUSE_SOURCE_STEP}" \
          --target_step "${step}" \
          --source_checkpoint "${reuse_source_checkpoint}" \
          --target_checkpoint "${checkpoint}" \
          --ss_coords_only >"${reuse_log}" 2>&1
        log_master "reused verified Native-SS coords shard=${shard} source=${reuse_source}"
      fi
      local resume=()
      local floor_args=()
      if [[ -n "${FROZEN_STOCK_FLOOR_ROOT}" ]]; then
        local floor_report
        floor_report="${FROZEN_STOCK_FLOOR_ROOT}/shard${shard}_${start}_${end}/report.json"
        test -s "${floor_report}" || {
          echo "ERROR: missing frozen Stock-floor worker ${floor_report}" >&2
          exit 97
        }
        floor_args+=(--frozen_stock_floor_worker_report "${floor_report}")
      fi
      [[ -e "${out}" ]] && resume+=(--resume)
      CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u -m \
        pose_point_depth_mv.evaluate_proobjaverse_official_native_ss_stock_slat \
        worker \
        --cache_manifest "${cache}/slat_manifest.json" \
        --lifting_cache_manifest "${cache}/lifting_manifest.json" \
        --native_ss_report "${PREDICTED_SS_REPORT}" \
        --stock_slat_freeze "${STOCK_FREEZE}" \
        --trained_slat_checkpoint "${checkpoint}" \
        --trained_slat_weights ema \
        --expected_trained_slat_step "${step}" \
        --allow_trained_slat_target_protocol_mismatch \
        --expected_checkpoint_training_membership all_training \
        --output_dir "${out}" \
        --object_start "${start}" \
        --object_end "${end}" \
        --joint_seeds 42,43,44 \
        --weights ema \
        --surface_samples 20000 \
        --amp_dtype bf16 \
        --frozen_target_recon_reports "${FROZEN_TARGET_RECON_REPORTS}" \
        "${floor_args[@]}" \
        "${resume[@]}" >"${log}" 2>&1 &
    else
      CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u -m \
        pose_point_depth_mv.evaluate_proobjaverse_official_slat_gt_support_cross_protocol \
        --arm condition_lora \
        --cache_manifest "${cache}/slat_manifest.json" \
        --lifting_cache_manifest "${cache}/lifting_manifest.json" \
        --checkpoint "${checkpoint}" \
        --native_ss_report "${TRAINING_SS_REPORT}" \
        --stock_slat_freeze "${STOCK_FREEZE}" \
        --output_dir "${out}" \
        --weights ema \
        --joint_seeds 42,43,44 \
        --max_objects 64 \
        --object_start "${start}" \
        --object_end "${end}" \
        --surface_samples 20000 \
        --bootstrap_samples 5000 \
        --amp_dtype bf16 \
        --allow_checkpoint_data_path_relocation \
        --allow_checkpoint_target_protocol_mismatch \
        --expected_checkpoint_training_membership all_training \
        >"${log}" 2>&1 &
    fi
    pids+=("$!")
    log_master "started shard=${shard} gpu=${gpu} pid=$! range=[${start},${end}) log=${log}"
  done

  local failed=0
  for ((shard=0; shard<EVAL_GPU_COUNT; shard++)); do
    local pid=${pids[$shard]}
    [[ -z "${pid}" ]] && continue
    set +e
    wait "${pid}"
    local rc=$?
    set -e
    if [[ "${group}" == "legacy_dev48_predicted_training_overlap" ]]; then
      [[ ${rc} -eq 0 || ${rc} -eq 2 ]] || failed=1
    else
      [[ ${rc} -eq 0 ]] || failed=1
    fi
    [[ -s "${shard_dirs[$shard]}/report.json" ]] || failed=1
    log_master "finished shard=${shard} rc=${rc}"
  done
  [[ ${failed} -eq 0 ]] || { echo "ERROR: worker wave failed" >&2; exit 98; }

  local reports=""
  for out in "${shard_dirs[@]}"; do
    reports+="${reports:+,}${out}/report.json"
  done
  local final="${group_root}/aggregate_v1"
  if [[ ! -s "${final}/report.json" ]]; then
    if [[ -e "${final}" ]]; then
      if (( PRESERVE_INTERRUPTED_OUTPUTS == 1 )); then
        preserve_interrupted_dir "${final}" "aggregate"
      else
        echo "ERROR: partial aggregate exists ${final}" >&2
        exit 99
      fi
    fi
    if [[ "${group}" == "legacy_dev48_predicted_training_overlap" ]]; then
      set +e
      "${PY}" -u -m pose_point_depth_mv.evaluate_proobjaverse_official_native_ss_stock_slat \
        aggregate \
        --cache_manifest "${cache}/slat_manifest.json" \
        --lifting_cache_manifest "${cache}/lifting_manifest.json" \
        --shard_reports "${reports}" \
        --output_dir "${final}" \
        --object_start 16 \
        --object_end 64 \
        --expected_objects 48 \
        --joint_seeds 42,43,44 \
        --bootstrap_samples 5000 \
        --chamfer_win_rate_min 0.55 \
        --largest_component_delta_min -0.02
      local rc=$?
      set -e
      [[ ${rc} -eq 0 || ${rc} -eq 3 ]] || exit "${rc}"
    else
      "${PY}" -u -m pose_point_depth_mv.aggregate_proobjaverse_official_slat_gt_support \
        --shard_reports "${reports}" \
        --output_dir "${final}" \
        --expected_objects 64 \
        --bootstrap_samples 5000
    fi
  fi
  test -s "${final}/report.json"

  if [[ "${group}" == "legacy_dev48_predicted_training_overlap" ]]; then
    local expected_sha current_reports target_roots strict_out
    expected_sha=$(sha256sum "${checkpoint}" | awk '{print $1}')
    current_reports="${reports}"
    target_roots=""
    for out in "${shard_dirs[@]}"; do
      target_roots+="${target_roots:+,}${out}/target_mesh_cache"
    done
    strict_out="${OUTPUT_ROOT}/$(printf 'step_%06d' "${step}")/legacy_dev48_vs_strict_reconviagen_training_overlap/aggregate_v1"
    if [[ ! -s "${strict_out}/report.json" ]]; then
      if [[ -e "${strict_out}" ]]; then
        if (( PRESERVE_INTERRUPTED_OUTPUTS == 1 )); then
          preserve_interrupted_dir "${strict_out}" "strict aggregate"
        else
          echo "ERROR: partial strict aggregate exists ${strict_out}" >&2
          exit 100
        fi
      fi
      "${PY}" -u -m pose_point_depth_mv.aggregate_proobjaverse_official_ss_slat_vs_reconviagen \
        --dev_split "${DEV_SPLIT}" \
        --cache_report "${CACHE_REPORT}" \
        --target_report "${TARGET_REPORT}" \
        --target_mesh_root "${TARGET_MESH_ROOT}" \
        --paired_target_cache_roots "${target_roots}" \
        --recon_reports "${RECON_REPORTS}" \
        --current_reports "${current_reports}" \
        --expected_current_step "${step}" \
        --expected_current_sha256 "${expected_sha}" \
        --bootstrap_samples 5000 \
        --evaluation_membership_scope checkpoint_training_overlap \
        --require_exact_strict_target_sha256 \
        --output_dir "${strict_out}"
    fi
    test -s "${strict_out}/report.json"
  fi
}

log_master "============================================================"
log_master "30K no-VGGT checkpoint compatibility evaluation"
log_master "GPUs=${EVAL_GPUS} steps=${STEPS} groups=${EVAL_GROUPS}"
log_master "IMPORTANT: legacy Train64 and legacy Dev64/Dev48 are checkpoint-training-overlap"
log_master "started=$(date -Is)"
log_master "============================================================"

if (( SKIP_GPU_SETUP == 0 )); then
  wait_for_selected_gpus
  start_eval_gpu_reservations \
    "${EVAL_GPUS}" "${EVAL_GPU_RESERVATION_ROOT}" \
    "slat29861_${EVAL_TAG}" "${PY}" "${PROJECT}"
else
  log_master "GPU setup skipped: all requested worker reports are already complete"
fi

for step in "${STEP_ARRAY[@]}"; do
  checkpoint=$(wait_for_checkpoint "${step}")
  for group in "${GROUP_ARRAY[@]}"; do
    run_wave "${step}" "${group}" "${checkpoint}"
  done
done

"${PY}" - "${OUTPUT_ROOT}" "${EVAL_GPUS}" "${EVAL_GPU_COUNT}" "${STEPS}" "${EVAL_GROUPS}" "${SUMMARY_PATH}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
eval_gpus = sys.argv[2]
worker_count = int(sys.argv[3])
steps = tuple(int(value) for value in sys.argv[4].split(",") if value)
requested_groups = tuple(value for value in sys.argv[5].split(",") if value)
summary_path = Path(sys.argv[6])
groups = []
for group in requested_groups:
    groups.append(group)
    if group == "legacy_dev48_predicted_training_overlap":
        groups.append("legacy_dev48_vs_strict_reconviagen_training_overlap")
rows = []
for step in steps:
    for group in groups:
        path = root / f"step_{step:06d}" / group / "aggregate_v1" / "report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "step": step,
            "group": group,
            "report": str(path),
            "runtime_passed": report.get("passed"),
            "scope_guard": report.get("scope_guard"),
        })
payload = {
    "format": (
        "pose_point_depth_mv.proobjaverse_slat29861_legacy_eval_4gpu.v2"
        if worker_count == 4
        else "pose_point_depth_mv.proobjaverse_slat29861_legacy_eval_2gpu.v2"
    ),
    "completed": True,
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "steps": list(steps),
    "aggregate_count": len(rows),
    "expected_aggregate_count": len(steps) * len(groups),
    "all_expected_aggregates_generated": len(rows) == len(steps) * len(groups),
    "all_16_aggregates_generated": len(rows) == 16,
    "runtime_topology": {
        "physical_gpus": [int(value) for value in eval_gpus.split(",")],
        "worker_count": worker_count,
        "gpu_reservation": "one low-memory CUDA holder per selected GPU",
    },
    "scientific_scope": (
        "All reused legacy Train64/Dev64 objects are in the 29,861-object checkpoint "
        "training UID set. Results are compatibility/training-overlap diagnostics, "
        "not 30K held-out generalization."
    ),
    "results": rows,
}
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2, ensure_ascii=False))
PY

log_master "COMPLETE: ${OUTPUT_ROOT}"
