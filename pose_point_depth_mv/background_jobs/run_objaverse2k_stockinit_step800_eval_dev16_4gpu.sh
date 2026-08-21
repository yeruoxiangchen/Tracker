#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
GPUS=${OBJAVERSE2K_STOCKINIT_EVAL_GPUS:-1,2,6,7}
RUN=${OBJAVERSE2K_SLAT_RUN:-/data/zjr/objaverse2k_no_vggt_slat_20260811_v1}
EXPECTED_OBJECTS=16
EXPECTED_WORKERS=4
DEV_CACHE=${RUN}/slat_cache_dev64_seed424344_merged_v1/manifest.json
DEV_LIFT=${RUN}/split_dev64_v1/dev/lifting_manifest.json
SS_RUN=/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1
SS_REPORT=${SS_RUN}/ss_eval_synthetic_dev32_fixedcfg3_count125_v3/report.json
STOCK_FREEZE=/data/zjr/native_slat_genrecon_v2_mixed1k_20260802_v1/stock_slat_freeze_v2.json
M8_STEP800=${SS_RUN}/slat_mixed_step2000_seed42_2gpu_v1/checkpoints/step_000800.pt
STOCKINIT_STEP800=${RUN}/slat_objaverse2135_stockinit_step2000_seed42_8gpu_bs8_v1/checkpoints/step_000800.pt
CURRENT800=${RUN}/eval_dev16_step000800_stock_m8_objaverse2k_8gpu_v1
CURRENT2000=${RUN}/eval_dev16_step002000_stock_m8_objaverse2k_8gpu_v1
OUT=${RUN}/eval_stockinit_step800_dev16_4gpu_v1
STATE=${RUN}/logs/eval_stockinit_step800_dev16_4gpu.state
EXIT_CODE=${RUN}/logs/eval_stockinit_step800_dev16_4gpu.exit_code
LOCK=${RUN}/logs/eval_stockinit_step800_dev16_4gpu.lock

IFS=',' read -r -a GPU_ARRAY <<<"${GPUS}"
if [ "${#GPU_ARRAY[@]}" -ne 4 ] || \
   [ "$(printf '%s\n' "${GPU_ARRAY[@]}" | sort -u | wc -l)" -ne 4 ]; then
  echo "OBJAVERSE2K_STOCKINIT_EVAL_GPUS requires four distinct GPUs" >&2
  exit 96
fi
for REQUIRED in \
  "${DEV_CACHE}" "${DEV_LIFT}" "${SS_REPORT}" "${STOCK_FREEZE}" \
  "${M8_STEP800}" "${STOCKINIT_STEP800}"; do
  test -s "${REQUIRED}"
done
for WORKER in 0 1 2 3; do
  test -s "${CURRENT800}/objaverse2k_worker_${WORKER}/report.json"
  test -s "${CURRENT2000}/objaverse2k_worker_${WORKER}/report.json"
done
"${PY}" - "${M8_STEP800}" "${STOCKINIT_STEP800}" \
  "${RUN}/slat_objaverse2135_stockinit_step2000_seed42_8gpu_bs8_v1/stage_report_step_000800.json" \
  "${CURRENT800}/objaverse2k_worker_0/report.json" \
  "${CURRENT2000}/objaverse2k_worker_0/report.json" <<'PY'
import json, sys, torch
m8 = torch.load(sys.argv[1], map_location="cpu")
stockinit = torch.load(sys.argv[2], map_location="cpu")
stage, current800, current2000 = [
    json.load(open(path, encoding="utf-8")) for path in sys.argv[3:]
]
assert m8["step"] == 800 and m8["args"]["max_steps"] == 2000
assert stockinit["step"] == 800 and stockinit["args"]["max_steps"] == 2000
assert stockinit["args"]["init_checkpoint"] == ""
assert stockinit["args"]["grad_accum"] == 1 and stockinit["args"]["seed"] == 42
assert "initialization" not in stockinit["model_summary"]
assert stage["stage_complete"] is True and stage["initial_stock_audit"]["passed"] is True
assert current800["run_config"]["checkpoint_step"] == 800
assert current2000["run_config"]["checkpoint_step"] == 2000
print({
    "new": "Stock-init Objaverse2K step800",
    "historical_reference": "M8 step800",
    "current_references": [
        "eight-GPU M8-init Objaverse2K step800",
        "eight-GPU M8-init Objaverse2K step2000",
    ],
})
PY

mkdir -p "${RUN}/logs" "${OUT}"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "Objaverse2K Stock-init step800 dev16 evaluation is already running" >&2
  exit 99
fi
finish() {
  RC=$?
  trap - EXIT
  printf '%s\n' "${RC}" >"${EXIT_CODE}"
  printf 'finished_at=%s rc=%s\n' "$(date --iso-8601=seconds)" "${RC}" >"${STATE}"
  exit "${RC}"
}
trap finish EXIT
printf 'started_at=%s state=running gpus=%s objects=%s\n' \
  "$(date --iso-8601=seconds)" "${GPUS}" "${EXPECTED_OBJECTS}" >"${STATE}"
rm -f "${EXIT_CODE}"

run_m8_step800() {
  local pids=()
  for WORKER in 0 1 2 3; do
    WORKER_OUT=${OUT}/m8_step800_worker_${WORKER}
    if [ -s "${WORKER_OUT}/report.json" ]; then continue; fi
    RESUME=()
    if [ -e "${WORKER_OUT}" ]; then RESUME=(--resume); fi
    CUDA_VISIBLE_DEVICES="${GPU_ARRAY[${WORKER}]}" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
    MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
    TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PY}" -u -m pose_point_depth_mv.evaluate_objaverse2k_no_vggt_slat worker \
      --cache_manifest "${DEV_CACHE}" \
      --lifting_cache_manifest "${DEV_LIFT}" \
      --checkpoint "${M8_STEP800}" \
      --model_label m8_step800 \
      --native_ss_report "${SS_REPORT}" \
      --stock_slat_freeze "${STOCK_FREEZE}" \
      --output_dir "${WORKER_OUT}" \
      --weights ema --joint_seeds 42,43,44 --noise_seed 20260811 \
      --worker_index "${WORKER}" --num_workers "${EXPECTED_WORKERS}" \
      --expected_objects "${EXPECTED_OBJECTS}" \
      --surface_samples 20000 --amp_dtype bf16 "${RESUME[@]}" \
      >"${RUN}/logs/eval_stockinit_step800_m8_step800_worker_${WORKER}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for PID in "${pids[@]}"; do if ! wait "${PID}"; then failed=1; fi; done
  if [ "${failed}" -ne 0 ]; then return 2; fi
}

run_stockinit_fourway() {
  local pids=()
  for WORKER in 0 1 2 3; do
    WORKER_OUT=${OUT}/stockinit_fourway_worker_${WORKER}
    if [ -s "${WORKER_OUT}/report.json" ]; then continue; fi
    RESUME=()
    if [ -e "${WORKER_OUT}" ]; then RESUME=(--resume); fi
    CUDA_VISIBLE_DEVICES="${GPU_ARRAY[${WORKER}]}" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
    MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba_cache \
    TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PY}" -u -m pose_point_depth_mv.evaluate_objaverse2k_no_vggt_slat_fourway worker \
      --cache_manifest "${DEV_CACHE}" \
      --lifting_cache_manifest "${DEV_LIFT}" \
      --checkpoint "${STOCKINIT_STEP800}" \
      --native_ss_report "${SS_REPORT}" \
      --stock_slat_freeze "${STOCK_FREEZE}" \
      --output_dir "${WORKER_OUT}" \
      --weights ema --joint_seeds 42,43,44 --noise_seed 20260811 \
      --worker_index "${WORKER}" --num_workers "${EXPECTED_WORKERS}" \
      --expected_objects "${EXPECTED_OBJECTS}" \
      --surface_samples 20000 --amp_dtype bf16 "${RESUME[@]}" \
      >"${RUN}/logs/eval_stockinit_step800_fourway_worker_${WORKER}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for PID in "${pids[@]}"; do if ! wait "${PID}"; then failed=1; fi; done
  if [ "${failed}" -ne 0 ]; then return 2; fi
}

run_m8_step800
run_stockinit_fourway

M8_REPORTS=""
STOCKINIT_REPORTS=""
CURRENT800_REPORTS=""
CURRENT2000_REPORTS=""
for WORKER in 0 1 2 3; do
  M8_REPORT=${OUT}/m8_step800_worker_${WORKER}/report.json
  STOCKINIT_REPORT=${OUT}/stockinit_fourway_worker_${WORKER}/report.json
  CURRENT800_REPORT=${CURRENT800}/objaverse2k_worker_${WORKER}/report.json
  CURRENT2000_REPORT=${CURRENT2000}/objaverse2k_worker_${WORKER}/report.json
  for REPORT in \
    "${M8_REPORT}" "${STOCKINIT_REPORT}" \
    "${CURRENT800_REPORT}" "${CURRENT2000_REPORT}"; do
    test -s "${REPORT}"
  done
  if [ -z "${M8_REPORTS}" ]; then
    M8_REPORTS=${M8_REPORT}
    STOCKINIT_REPORTS=${STOCKINIT_REPORT}
    CURRENT800_REPORTS=${CURRENT800_REPORT}
    CURRENT2000_REPORTS=${CURRENT2000_REPORT}
  else
    M8_REPORTS=${M8_REPORTS},${M8_REPORT}
    STOCKINIT_REPORTS=${STOCKINIT_REPORTS},${STOCKINIT_REPORT}
    CURRENT800_REPORTS=${CURRENT800_REPORTS},${CURRENT800_REPORT}
    CURRENT2000_REPORTS=${CURRENT2000_REPORTS},${CURRENT2000_REPORT}
  fi
done

if [ ! -s "${OUT}/fourway/report.json" ]; then
  "${PY}" -u -m pose_point_depth_mv.evaluate_objaverse2k_no_vggt_slat_fourway aggregate \
    --worker_reports "${STOCKINIT_REPORTS}" \
    --output_dir "${OUT}/fourway" \
    --expected_workers "${EXPECTED_WORKERS}" \
    --expected_objects "${EXPECTED_OBJECTS}" \
    --bootstrap_samples 10000
fi
if [ ! -s "${OUT}/comparison/report.json" ]; then
  "${PY}" -u -m pose_point_depth_mv.summarize_objaverse2k_stockinit_step800 \
    --m8_step800_reports "${M8_REPORTS}" \
    --stockinit_reports "${STOCKINIT_REPORTS}" \
    --current800_reports "${CURRENT800_REPORTS}" \
    --current2000_reports "${CURRENT2000_REPORTS}" \
    --output_dir "${OUT}/comparison" \
    --expected_objects "${EXPECTED_OBJECTS}" \
    --bootstrap_samples 10000
fi

test -s "${OUT}/comparison/summary.txt"
test -s "${OUT}/fourway/summary.txt"
cat "${OUT}/comparison/summary.txt"
cat "${OUT}/fourway/summary.txt"
