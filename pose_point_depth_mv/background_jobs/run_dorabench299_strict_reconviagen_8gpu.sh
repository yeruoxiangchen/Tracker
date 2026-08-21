#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
BENCHMARK=${BENCHMARK_MANIFEST:-/data/zjr/dorabench_reconviagen_style_dora300_trellis40_input0_9_19_29_20260821_v1/manifest.json}
CURRENT=${CURRENT_REPORT:-/data/zjr/dorabench_dora300_ss30k_slat30k_step30k_metrics_seed42_trellis40_input0_9_19_29_7gpu_20260821_v1/aggregate_failure_aware_v1/report.json}
RUNTIME=${RUNTIME_INPUT_MANIFEST:-/data/zjr/dorabench_dora300_ss30k_slat30k_step30k_metrics_seed42_trellis40_input0_9_19_29_7gpu_20260821_v1/00_exact_model_o_runtime/runtime_input_manifest.json}
OUT=${OUTPUT_ROOT:-/data/zjr/dorabench_dora299_strict_reconviagen_seed42_trellis40_input0_9_19_29_8gpu_20260821_v1}
GPUS_CSV=${EVAL_GPUS:-0,1,2,3,4,5,6,7}
REPAIR_GPUS_CSV=${REPAIR_GPUS:-0,2,4,6}
STARTUP_TIMEOUT_SECONDS=${STARTUP_TIMEOUT_SECONDS:-420}
SEED=${EVAL_SEED:-42}
PRETRAINED=${PRETRAINED:-Stable-X/trellis-vggt-v0-2}
SUBSET=${OUT}/protocol/dora299_current_valid_subset.json
INFERENCE=${OUT}/inference_aggregate_v1/report.json
METRIC_ROOT=${OUT}/metric_workers_model_o_v2
FINAL=${OUT}/aggregate_model_o_v2

test -x "${PY}"
test -s "${BENCHMARK}"
test -s "${CURRENT}"
test -s "${RUNTIME}"

IFS=, read -r -a GPU_ARRAY <<<"${GPUS_CSV}"
IFS=, read -r -a REPAIR_GPU_ARRAY <<<"${REPAIR_GPUS_CSV}"
WORKERS=${#GPU_ARRAY[@]}
REPAIR_WORKERS=${#REPAIR_GPU_ARRAY[@]}
if (( WORKERS != 8 )); then
  echo "ERROR: Dora299 strict run requires exactly eight primary GPUs" >&2
  exit 90
fi
if [[ "$(printf '%s\n' "${GPU_ARRAY[@]}" | sort -u | wc -l)" -ne 8 ]]; then
  echo "ERROR: EVAL_GPUS contains duplicates" >&2
  exit 91
fi
if (( REPAIR_WORKERS < 1 || REPAIR_WORKERS > 4 )); then
  echo "ERROR: REPAIR_GPUS must contain one to four GPUs" >&2
  exit 92
fi

mkdir -p "${OUT}/logs" "${OUT}/workers" "${OUT}/repair/logs" \
  "${OUT}/repair/plans" "${METRIC_ROOT}"
export PYTHONPATH="${PWD}:${PWD}/ReconViaGen:${PWD}/ReconViaGen/wheels/vggt"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export ATTN_BACKEND=flash_attn
export SPCONV_ALGO=native
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false

echo "===== P0 freeze exact current-valid Dora299 subset ====="
"${PY}" -m pose_point_depth_mv.dorabench299_strict_reconviagen_runtime prepare \
  --benchmark_manifest "${BENCHMARK}" \
  --runtime_input_manifest "${RUNTIME}" \
  --current_report "${CURRENT}" \
  --output "${SUBSET}"

if [[ -s "${FINAL}/report.json" ]]; then
  echo "DORA299 STRICT RECONVIAGEN ALREADY COMPLETE: ${FINAL}/report.json"
  exit 0
fi

echo "===== P1 strict ReconViaGen primary inference: 299 objects / 8 GPUs ====="
echo "GPUs=${GPUS_CSV} seed=${SEED} output=${OUT}"
echo "excluded=Level4:dora_39170f9710c47fb395de"
pids=()
worker_ids=()
primary_start_failures=0
for worker in "${!GPU_ARRAY[@]}"; do
  gpu=${GPU_ARRAY[$worker]}
  worker_name=$(printf 'worker_%02d' "${worker}")
  worker_root=${OUT}/workers/${worker_name}
  log=${OUT}/logs/${worker_name}_gpu${gpu}.log
  mkdir -p "${worker_root}"
  mapfile -t object_keys < <(
    "${PY}" - "${SUBSET}" "${worker}" "${WORKERS}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
worker, workers = map(int, sys.argv[2:])
for index, row in enumerate(p["objects"]):
    if index % workers == worker:
        print(f"{row['category']}:{row['uid']}")
PY
  )
  expected=$((worker < 3 ? 38 : 37))
  if (( ${#object_keys[@]} != expected )); then
    echo "ERROR: worker ${worker} object count ${#object_keys[@]} != ${expected}" >&2
    exit 93
  fi
  object_args=()
  for key in "${object_keys[@]}"; do object_args+=(--object "${key}"); done
  (
    set -euo pipefail
    echo "[worker ${worker}] objects=${#object_keys[@]} physical_gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
      "${PY}" -u -m manual_mesh_reconstruction.reconviagen \
        --runtime_input_manifest "${RUNTIME}" \
        --output_dir "${worker_root}/01_strict_reconviagen" \
        --pretrained "${PRETRAINED}" \
        --seeds "${SEED}" \
        --device cuda \
        "${object_args[@]}"
    echo "[worker ${worker}] INFERENCE COMPLETE"
  ) >"${log}" 2>&1 &
  pid=$!
  pids+=("${pid}")
  worker_ids+=("${worker}")
  echo "worker=${worker} gpu=${gpu} pid=${pid} objects=${#object_keys[@]} log=${log}"
  startup_begin=$SECONDS
  while ! grep -q 'Sampling:' "${log}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "WARNING: worker ${worker} exited before sampling; automatic repair will cover its objects" >&2
      wait "${pid}" || true
      primary_start_failures=$((primary_start_failures + 1))
      break
    fi
    if (( SECONDS - startup_begin > STARTUP_TIMEOUT_SECONDS )); then
      echo "WARNING: worker ${worker} startup exceeded ${STARTUP_TIMEOUT_SECONDS}s; continuing other launches" >&2
      primary_start_failures=$((primary_start_failures + 1))
      break
    fi
    sleep 2
  done
  if kill -0 "${pid}" 2>/dev/null && grep -q 'Sampling:' "${log}"; then
    echo "worker=${worker} gpu=${gpu} startup_gate=PASS elapsed=$((SECONDS-startup_begin))s"
  fi
done

primary_runtime_failures=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "WARNING: primary worker ${worker_ids[$index]} failed; valid atomic outputs preserved" >&2
    primary_runtime_failures=$((primary_runtime_failures + 1))
  fi
done
echo "primary_start_failures=${primary_start_failures} primary_runtime_failures=${primary_runtime_failures}"

run_repair_list() {
  local stage=$1 worker=$2 gpu=$3
  local list=${OUT}/repair/plans/${stage}_worker${worker}.txt
  local position=0 failures=0
  while IFS= read -r key; do
    [[ -n "${key}" ]] || continue
    slug=$(printf '%03d' "${position}")
    destination=${OUT}/repair/${stage}_worker${worker}_${slug}
    echo "[${stage}] worker=${worker} gpu=${gpu} key=${key}"
    set +e
    CUDA_VISIBLE_DEVICES="${gpu}" \
      "${PY}" -u -m manual_mesh_reconstruction.reconviagen \
        --runtime_input_manifest "${RUNTIME}" \
        --output_dir "${destination}" \
        --pretrained "${PRETRAINED}" \
        --seeds "${SEED}" \
        --device cuda \
        --object "${key}"
    rc=$?
    set -e
    if (( rc != 0 )); then
      echo "WARNING: ${stage} key=${key} rc=${rc}; next attempt uses a fresh process" >&2
      failures=$((failures + 1))
    fi
    position=$((position + 1))
  done < "${list}"
  echo "[${stage}] worker=${worker} attempted=${position} failures=${failures}"
}

run_repair_wave() {
  local stage=$1
  echo "===== ${stage}: discover and retry missing objects in fresh processes ====="
  "${PY}" -m pose_point_depth_mv.dorabench299_strict_reconviagen_runtime plan \
    --subset_manifest "${SUBSET}" \
    --runtime_input_manifest "${RUNTIME}" \
    --output_root "${OUT}" \
    --plan_root "${OUT}/repair/plans" \
    --stage "${stage}" \
    --worker_count "${REPAIR_WORKERS}" \
    --seed "${SEED}"
  local repair_pids=()
  for worker in "${!REPAIR_GPU_ARRAY[@]}"; do
    gpu=${REPAIR_GPU_ARRAY[$worker]}
    log=${OUT}/repair/logs/${stage}_worker${worker}_gpu${gpu}.log
    run_repair_list "${stage}" "${worker}" "${gpu}" >"${log}" 2>&1 &
    pid=$!
    repair_pids+=("${pid}")
    # Stagger model initialization to avoid a host-RAM load spike.
    startup_begin=$SECONDS
    while kill -0 "${pid}" 2>/dev/null && ! grep -Eq 'Sampling:|attempted=0' "${log}"; do
      if (( SECONDS - startup_begin > STARTUP_TIMEOUT_SECONDS )); then break; fi
      sleep 2
    done
  done
  for pid in "${repair_pids[@]}"; do wait "${pid}" || true; done
}

run_repair_wave attempt1
run_repair_wave attempt2

echo "===== P2 aggregate exact 299 inference records ====="
"${PY}" -m pose_point_depth_mv.dorabench299_strict_reconviagen_runtime aggregate \
  --subset_manifest "${SUBSET}" \
  --runtime_input_manifest "${RUNTIME}" \
  --output_root "${OUT}" \
  --output "${INFERENCE}" \
  --pretrained "${PRETRAINED}" \
  --seed "${SEED}"

echo "===== P3 CPU surface metrics: 8 workers ====="
metric_pids=()
for worker in $(seq 0 7); do
  worker_out=$(printf '%s/worker_%02d' "${METRIC_ROOT}" "${worker}")
  log=$(printf '%s/logs/metric_worker_%02d.log' "${OUT}" "${worker}")
  "${PY}" -u -m pose_point_depth_mv.evaluate_dorabench299_strict_reconviagen worker \
    --subset_manifest "${SUBSET}" \
    --inference_aggregate "${INFERENCE}" \
    --output_dir "${worker_out}" \
    --worker_index "${worker}" \
    --worker_count 8 \
    --surface_points 100000 \
    --fscore_radius 0.1 \
    --seed "${SEED}" >"${log}" 2>&1 &
  metric_pids+=("$!")
done
metric_failed=0
for pid in "${metric_pids[@]}"; do
  if ! wait "${pid}"; then metric_failed=1; fi
done
if (( metric_failed != 0 )); then
  echo "ERROR: at least one Dora299 metric worker failed" >&2
  exit 96
fi

"${PY}" -u -m pose_point_depth_mv.evaluate_dorabench299_strict_reconviagen aggregate \
  --subset_manifest "${SUBSET}" \
  --inference_aggregate "${INFERENCE}" \
  --current_report "${CURRENT}" \
  --workers_root "${METRIC_ROOT}" \
  --output_dir "${FINAL}" \
  --expected_workers 8 \
  --surface_points 100000 \
  --fscore_radius 0.1 \
  --seed "${SEED}"

echo "DORA299 STRICT RECONVIAGEN COMPLETE: ${FINAL}/report.json"
