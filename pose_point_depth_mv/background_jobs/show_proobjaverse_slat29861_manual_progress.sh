#!/usr/bin/env bash
set -u

ROOT=${ROOT:-/data/zjr/proobjaverse_official_slat_train29861_20260817_v1}
STEP=${1:?usage: $0 STEP}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/eval_legacy_protocol2128_step10k30k60k_seed424344_4gpu_targetlocked_v2}
STEP_ROOT=${OUTPUT_ROOT}/step_$(printf '%06d' "${STEP}")
PREDICTED=${STEP_ROOT}/legacy_dev48_predicted_training_overlap
STRICT=${STEP_ROOT}/legacy_dev48_vs_strict_reconviagen_training_overlap/aggregate_v1/report.json
TRAIN_GT=${STEP_ROOT}/train64_gt/aggregate_v1/report.json
DEV_GT=${STEP_ROOT}/legacy_dev64_gt_training_overlap/aggregate_v1/report.json
PREDICTED_AGG=${PREDICTED}/aggregate_v1/report.json
MANUAL_SUMMARY=${STEP_ROOT}/manual_legacy_dev48_predicted_training_overlap_summary.json

date -Is
echo "step=${STEP}"
echo "============================================================"
echo "whole-step stages"
[[ -s "${TRAIN_GT}" ]] && train_status=complete || train_status=pending_or_not_requested
[[ -s "${DEV_GT}" ]] && dev_status=complete || dev_status=pending_or_not_requested
[[ -s "${PREDICTED_AGG}" ]] && predicted_status=complete || predicted_status=pending
[[ -s "${STRICT}" ]] && strict_status=complete || strict_status=pending
[[ -s "${MANUAL_SUMMARY}" ]] && summary_status=complete || summary_status=pending
printf 'train64_gt=%s\n' "${train_status}"
printf 'legacy_dev64_gt=%s\n' "${dev_status}"

echo "------------------------------------------------------------"
echo "Dev48 predicted-support A/B/C"
total_pairs=0
complete_workers=0
for spec in "0 16 28" "1 28 40" "2 40 52" "3 52 64"; do
  read -r shard start end <<<"${spec}"
  worker=${PREDICTED}/shard${shard}_${start}_${end}
  pairs=$(find "${worker}/mesh_pairs" -type f -name pair_record.json 2>/dev/null | wc -l)
  report=pending
  if [[ -s "${worker}/report.json" ]]; then
    report=complete
    complete_workers=$((complete_workers + 1))
  fi
  total_pairs=$((total_pairs + pairs))
  printf 'shard%d [%d,%d) pairs=%d/36 report=%s\n' \
    "${shard}" "${start}" "${end}" "${pairs}" "${report}"
done
pair_percent=$(awk -v done="${total_pairs}" 'BEGIN {printf "%.1f", 100*done/144}')
printf 'predicted_pair_records=%d/144 (%s%%)\n' "${total_pairs}" "${pair_percent}"
printf 'predicted_worker_reports=%d/4\n' "${complete_workers}"

printf 'predicted_aggregate=%s\n' "${predicted_status}"
printf 'strict_comparison=%s\n' "${strict_status}"
printf 'manual_summary=%s\n' "${summary_status}"

if [[ "${strict_status}" == complete ]]; then
  current_stage=complete
elif [[ "${predicted_status}" == complete ]]; then
  current_stage=strict_reconviagen_aggregate
elif (( complete_workers == 4 )); then
  current_stage=predicted_support_aggregate
else
  current_stage=predicted_support_mesh_generation
fi
printf 'CURRENT_STAGE=%s\n' "${current_stage}"

echo "------------------------------------------------------------"
echo "tmux sessions"
tmux list-sessions 2>/dev/null | grep -E 'slat30k|targetlocked' || echo "no matching tmux session visible"

DISARM_STATE=${ROOT}/logs/slat30k_targetv2_disarmed_parent.state
if [[ -s "${DISARM_STATE}" ]]; then
  read -r disarmed_pid disarmed_start <"${DISARM_STATE}"
  if kill -0 "${disarmed_pid}" 2>/dev/null; then
    printf 'old_scheduler_guard_target_pid=%s stat=%s\n' \
      "${disarmed_pid}" "$(ps -o stat= -p "${disarmed_pid}" | awk '{print $1}')"
  else
    printf 'old_scheduler_guard_state=stale_or_finishing pid=%s\n' "${disarmed_pid}"
  fi
else
  echo "old_scheduler_guard_state=not_armed_or_already_finished"
fi
tmux has-session -t slat30k_targetv2_guard 2>/dev/null \
  && echo "step10_completion_guard=running" \
  || echo "step10_completion_guard=not_running_or_finished"

echo "------------------------------------------------------------"
echo "live evaluator processes"
pgrep -af \
  '[e]valuate_proobjaverse_official_native_ss_stock_slat|[r]un_proobjaverse_slat29861_legacy_eval.*targetlocked|[r]un_proobjaverse_slat29861_manual_one_checkpoint_targetlocked' \
  || echo "no matching evaluator process visible"

echo "------------------------------------------------------------"
echo "GPU 4,5,6,7:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits 2>&1 | awk -F, '$1+0 >= 4 && $1+0 <= 7' || true
