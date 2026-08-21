#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
BASELINE=${BASELINE:?set BASELINE=trellis_s or trellis_m}
case "${BASELINE}" in
  trellis_s|trellis_m) ;;
  *) echo "ERROR: unsupported BASELINE=${BASELINE}" >&2; exit 90 ;;
esac

SUBSET=${SUBSET_MANIFEST:-/data/zjr/dorabench_dora299_strict_reconviagen_seed42_trellis40_input0_9_19_29_8gpu_20260821_v1/protocol/dora299_current_valid_subset.json}
CURRENT=${CURRENT_REPORT:-/data/zjr/dorabench_dora300_ss30k_slat30k_step30k_metrics_seed42_trellis40_input0_9_19_29_7gpu_20260821_v1/aggregate_failure_aware_v1/report.json}
ROOT=${OUTPUT_ROOT:-/data/zjr/dorabench_dora299_trellis_s_m_seed42_trellis40_input0_9_19_29_8gpu_20260821_v1/${BASELINE}}
GPUS_CSV=${EVAL_GPUS:-0,1,2,3,4,5,6,7}
SEED=${EVAL_SEED:-42}
PRETRAINED=${PRETRAINED:-microsoft/TRELLIS-image-large}
REVISION=${MODEL_REVISION:-25e0d31ffbebe4b5a97464dd851910efc3002d96}
SS_STEPS=${SS_STEPS:-30}
SS_CFG=${SS_CFG:-7.5}
SLAT_STEPS=${SLAT_STEPS:-12}
SLAT_CFG=${SLAT_CFG:-3.0}
INFERENCE=${ROOT}/inference_aggregate_v1/report.json
METRICS=${ROOT}/metric_workers
FINAL=${ROOT}/aggregate_v1

test -x "${PY}"
test -s "${SUBSET}"
test -s "${CURRENT}"
IFS=, read -r -a GPU_ARRAY <<<"${GPUS_CSV}"
WORKERS=${#GPU_ARRAY[@]}
if (( WORKERS != 8 )); then
  echo "ERROR: this launcher requires exactly 8 GPUs" >&2
  exit 91
fi
if [[ "$(printf '%s\n' "${GPU_ARRAY[@]}" | sort -u | wc -l)" -ne 8 ]]; then
  echo "ERROR: EVAL_GPUS contains duplicates" >&2
  exit 92
fi

mkdir -p "${ROOT}/logs" "${ROOT}/plans" "${METRICS}"
export PYTHONPATH="${PWD}:${PWD}/ReconViaGen"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export ATTN_BACKEND=flash_attn
export SPCONV_ALGO=native
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false

common=(
  --subset_manifest "${SUBSET}"
  --baseline "${BASELINE}"
  --pretrained "${PRETRAINED}"
  --model_revision "${REVISION}"
  --seed "${SEED}"
  --ss_steps "${SS_STEPS}"
  --ss_cfg "${SS_CFG}"
  --slat_steps "${SLAT_STEPS}"
  --slat_cfg "${SLAT_CFG}"
)

if [[ -s "${FINAL}/report.json" ]]; then
  echo "${BASELINE} ALREADY COMPLETE: ${FINAL}/report.json"
  exit 0
fi

echo "===== ${BASELINE}: primary inference / 299 objects / 8 GPUs ====="
pids=()
for worker in "${!GPU_ARRAY[@]}"; do
  gpu=${GPU_ARRAY[$worker]}
  log=$(printf '%s/logs/primary_worker_%02d_gpu%s.log' "${ROOT}" "${worker}" "${gpu}")
  mapfile -t keys < <(
    "${PY}" - "${SUBSET}" "${worker}" "${WORKERS}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
worker, workers = map(int, sys.argv[2:])
for index, row in enumerate(payload["objects"]):
    if index % workers == worker:
        print(f"{row['category']}:{row['uid']}")
PY
  )
  object_args=()
  for key in "${keys[@]}"; do object_args+=(--object "${key}"); done
  (
    set -euo pipefail
    echo "[primary] baseline=${BASELINE} worker=${worker} gpu=${gpu} objects=${#keys[@]}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u -m \
      pose_point_depth_mv.evaluate_dorabench299_trellis_baselines \
      inference-worker "${common[@]}" \
      --output_root "${ROOT}" --device cuda "${object_args[@]}"
    echo "[primary] baseline=${BASELINE} worker=${worker} COMPLETE"
  ) >"${log}" 2>&1 &
  pids+=("$!")
  echo "worker=${worker} gpu=${gpu} pid=${pids[-1]} objects=${#keys[@]} log=${log}"
  sleep 2
done

primary_failures=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then primary_failures=$((primary_failures + 1)); fi
done
echo "${BASELINE} primary_failures=${primary_failures}; atomic successes preserved"

run_repair_wave() {
  local stage=$1
  "${PY}" -m pose_point_depth_mv.evaluate_dorabench299_trellis_baselines \
    plan "${common[@]}" --output_root "${ROOT}" \
    --plan_root "${ROOT}/plans" --stage "${stage}" --worker_count "${WORKERS}"
  local repair_pids=()
  for worker in "${!GPU_ARRAY[@]}"; do
    local gpu=${GPU_ARRAY[$worker]}
    local plan=${ROOT}/plans/${stage}_worker${worker}.txt
    local log=${ROOT}/logs/${stage}_worker${worker}_gpu${gpu}.log
    (
      set -u
      attempted=0
      failures=0
      while IFS= read -r key; do
        [[ -n "${key}" ]] || continue
        attempted=$((attempted + 1))
        echo "[${stage}] baseline=${BASELINE} worker=${worker} key=${key}"
        CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u -m \
          pose_point_depth_mv.evaluate_dorabench299_trellis_baselines \
          inference-worker "${common[@]}" --output_root "${ROOT}" \
          --device cuda --object "${key}"
        rc=$?
        if (( rc != 0 )); then
          failures=$((failures + 1))
          echo "WARNING: ${stage} key=${key} rc=${rc}"
        fi
      done < "${plan}"
      echo "[${stage}] attempted=${attempted} failures=${failures}"
    ) >"${log}" 2>&1 &
    repair_pids+=("$!")
    sleep 2
  done
  for pid in "${repair_pids[@]}"; do wait "${pid}" || true; done
}

run_repair_wave attempt1
run_repair_wave attempt2

echo "===== ${BASELINE}: aggregate exact inference matrix ====="
"${PY}" -m pose_point_depth_mv.evaluate_dorabench299_trellis_baselines \
  aggregate-inference "${common[@]}" --output_root "${ROOT}" --output "${INFERENCE}"

echo "===== ${BASELINE}: CPU surface metrics / 8 workers ====="
metric_pids=()
for worker in $(seq 0 7); do
  worker_root=$(printf '%s/worker_%02d' "${METRICS}" "${worker}")
  log=$(printf '%s/logs/metric_worker_%02d.log' "${ROOT}" "${worker}")
  "${PY}" -u -m pose_point_depth_mv.evaluate_dorabench299_trellis_baselines \
    metric-worker --subset_manifest "${SUBSET}" \
    --inference_aggregate "${INFERENCE}" --baseline "${BASELINE}" \
    --output_dir "${worker_root}" --worker_index "${worker}" \
    --worker_count 8 --surface_points 100000 --fscore_radius 0.1 \
    --seed "${SEED}" >"${log}" 2>&1 &
  metric_pids+=("$!")
done
metric_failures=0
for pid in "${metric_pids[@]}"; do
  if ! wait "${pid}"; then metric_failures=$((metric_failures + 1)); fi
done
if (( metric_failures != 0 )); then
  echo "ERROR: ${BASELINE} metric worker failures=${metric_failures}" >&2
  exit 96
fi

"${PY}" -u -m pose_point_depth_mv.evaluate_dorabench299_trellis_baselines \
  aggregate-metrics --subset_manifest "${SUBSET}" \
  --inference_aggregate "${INFERENCE}" --current_report "${CURRENT}" \
  --workers_root "${METRICS}" --output_dir "${FINAL}" \
  --baseline "${BASELINE}" --expected_workers 8 \
  --surface_points 100000 --fscore_radius 0.1 --seed "${SEED}"

echo "${BASELINE} COMPLETE: ${FINAL}/report.json"

