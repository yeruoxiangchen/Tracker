#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zjr/anaconda3/envs/mpp_llava/bin/python}"
DATA_DIR="${FEEDBACK_DATA_DIR:-$PROJECT_ROOT/datasets/processed/lingoqa_feedback_opd_40k_v2}"
OUTPUT_DIR="${FEEDBACK_DIAGNOSTIC_DIR:-$PROJECT_ROOT/outputs/feedback_opd_diagnostics_v1}"
PERSISTENT="$PROJECT_ROOT/../mpp_llava_video_workspace/checkpoints"
CHECKPOINT_ROOT="${MPP_CHECKPOINT_ROOT:-/tmp/zjr_mpp_fast/checkpoints}"
if [[ ! -f "$CHECKPOINT_ROOT/mpp-qwen-next-sft/sharded/model.safetensors.index.json" ]]; then
  CHECKPOINT_ROOT="$PERSISTENT"
fi
UPSTREAM="${MPP_UPSTREAM_ROOT:-$PROJECT_ROOT/../mpp_llava_video_workspace/third_party/MPP-LLaVA}"
export PYTHONPATH="$PROJECT_ROOT:$UPSTREAM${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
common=(--data "$DATA_DIR/train.jsonl" --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR"
  --initial-dir "${FEEDBACK_INITIAL_DIR:-$PROJECT_ROOT/outputs/mpp_lingoqa_qformer_all_1epoch_v1}"
  --comparison-dir "${FEEDBACK_PROBE_DIR:-$PROJECT_ROOT/outputs/feedback_teacher_compare_v2}"
  --source-probe-dir "${FEEDBACK_SOURCE_PROBE_DIR:-$PROJECT_ROOT/outputs/feedback_opd_probe_v1}"
  --train-dir "${FEEDBACK_TRAIN_DIR:-$PROJECT_ROOT/outputs/mpp_feedback_opd_40k_2epoch_v2}"
  --control-dir "${FEEDBACK_CONTROL_DIR:-$PROJECT_ROOT/outputs/mpp_feedback_sft_control_40k_2epoch_v2}"
  --eval-dir "${FEEDBACK_EVAL_DIR:-$PROJECT_ROOT/outputs/feedback_opd_eval_v2}"
  --cache-dir "${MPP_EVA_CACHE_DIR:-/tmp/zjr_mpp_fast/cache/lingoqa_5f_eva_v1}"
  --upstream-root "$UPSTREAM" --checkpoint-root "$CHECKPOINT_ROOT"
  --visual-checkpoint-root "${MPP_VISUAL_CHECKPOINT_ROOT:-$PERSISTENT}"
  --bridge-init "${MPP_BRIDGE_INIT:-$PROJECT_ROOT/checkpoints/visual_bridge_init.pt}")
diagnose() { "$PYTHON_BIN" -u scripts/diagnose_feedback_opd.py "${common[@]}" --action "$@"; }
action="${1:-help}"
if (( $# )); then shift; fi
case "$action" in
  prepare|status|report|score) diagnose "$action" "$@" ;;
  infer)
    arm="${1:?Specify parent, ce or opd}"; shift
    diagnose infer --arm "$arm" "$@"
    ;;
  run)
    # Resumes complete JSONL records. Models run sequentially on one visible GPU.
    diagnose prepare "$@"
    for arm in parent ce opd; do
      diagnose infer --arm "$arm" "$@"
      diagnose score --arm "$arm" "$@"
    done
    diagnose report "$@"
    ;;
  run_parallel)
    IFS=',' read -r -a diagnostic_gpus <<< "$CUDA_VISIBLE_DEVICES"
    if (( ${#diagnostic_gpus[@]} != 3 )) || [[ "${diagnostic_gpus[0]}" == "${diagnostic_gpus[1]}" || "${diagnostic_gpus[0]}" == "${diagnostic_gpus[2]}" || "${diagnostic_gpus[1]}" == "${diagnostic_gpus[2]}" ]]; then
      echo 'run_parallel requires three distinct visible GPUs, e.g. CUDA_VISIBLE_DEVICES=0,1,2' >&2
      exit 2
    fi
    diagnose prepare "$@"
    arms=(parent ce opd)
    diagnostic_pids=()
    for index in 0 1 2; do
      arm="${arms[$index]}"
      mkdir -p "$OUTPUT_DIR/$arm"
      (
        export CUDA_VISIBLE_DEVICES="${diagnostic_gpus[$index]}"
        diagnose infer --arm "$arm" "$@"
        diagnose score --arm "$arm" "$@"
      ) > "$OUTPUT_DIR/$arm/runtime.log" 2>&1 &
      diagnostic_pids+=("$!")
      echo "Started diagnostic arm=$arm GPU=${diagnostic_gpus[$index]} log=$OUTPUT_DIR/$arm/runtime.log"
    done
    diagnostic_failed=0
    for diagnostic_pid in "${diagnostic_pids[@]}"; do
      wait "$diagnostic_pid" || diagnostic_failed=1
    done
    diagnose report "$@"
    exit "$diagnostic_failed"
    ;;
  *)
    echo 'usage: bash scripts/run_feedback_opd_diagnostics.sh prepare|infer parent/ce/opd|score|report|status|run|run_parallel [Python options]'
    [[ "$action" == help ]] || exit 2
    ;;
esac
