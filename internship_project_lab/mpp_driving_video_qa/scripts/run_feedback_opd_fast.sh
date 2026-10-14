#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export FEEDBACK_FAST=1
export FEEDBACK_RUNTIME_JUDGE_DEVICE="${FEEDBACK_RUNTIME_JUDGE_DEVICE:-local}"
export FEEDBACK_SPEED_VALIDATION_DIR="${FEEDBACK_SPEED_VALIDATION_DIR:-$PROJECT_ROOT/outputs/feedback_opd_speed_validation_v1}"
export FEEDBACK_JUDGE_EQUIVALENCE_REPORT="${FEEDBACK_JUDGE_EQUIVALENCE_REPORT:-$FEEDBACK_SPEED_VALIDATION_DIR/judge.json}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
PYTHON_BIN="${PYTHON_BIN:-/home/zjr/anaconda3/envs/mpp_llava/bin/python}"
case "${1:-help}" in
  verify_judge|verify_model|benchmark)
    CHECKPOINT_ROOT="${MPP_CHECKPOINT_ROOT:-/tmp/zjr_mpp_fast/checkpoints}"
    if [[ ! -f "$CHECKPOINT_ROOT/mpp-qwen-next-sft/sharded/model.safetensors.index.json" ]]; then
      CHECKPOINT_ROOT="$PROJECT_ROOT/../mpp_llava_video_workspace/checkpoints"
    fi
    args=(--data "${FEEDBACK_DATA_DIR:-$PROJECT_ROOT/datasets/processed/lingoqa_feedback_opd_40k_v2}/train.jsonl"
          --output-dir "$FEEDBACK_SPEED_VALIDATION_DIR"
          --initial-dir "${FEEDBACK_INITIAL_DIR:-$PROJECT_ROOT/outputs/mpp_lingoqa_qformer_all_1epoch_v1}"
          --cache-dir "${MPP_EVA_CACHE_DIR:-/tmp/zjr_mpp_fast/cache/lingoqa_5f_eva_v1}"
          --checkpoint-root "$CHECKPOINT_ROOT"
          --visual-checkpoint-root "${MPP_VISUAL_CHECKPOINT_ROOT:-$PROJECT_ROOT/../mpp_llava_video_workspace/checkpoints}"
          --bridge-init "${MPP_BRIDGE_INIT:-$PROJECT_ROOT/checkpoints/visual_bridge_init.pt}"
          --teacher-mode reference_only)
    if [[ "$1" == verify_judge ]]; then
      args+=(--action judge --cases "${FEEDBACK_JUDGE_CHECK_CASES:-128}")
    else
      args+=(--action model)
      if [[ "$1" == benchmark ]]; then
        args+=(--benchmark-updates "${FEEDBACK_BENCHMARK_UPDATES:-20}" --benchmark-accumulation 4)
        if [[ "${FEEDBACK_BENCHMARK_GPU_JUDGE:-0}" == 1 ]]; then args+=(--benchmark-gpu-judge); fi
      fi
    fi
    exec "$PYTHON_BIN" -u scripts/validate_feedback_opd_speed.py "${args[@]}"
    ;;
  perf)
    exec "$PYTHON_BIN" scripts/report_feedback_opd_speed.py \
      --output-dir "${FEEDBACK_TRAIN_DIR:-$PROJECT_ROOT/outputs/mpp_feedback_opd_40k_2epoch_v2}"
    ;;
  train_opd|train_control|status|eval_dev|eval_selected|report|report_eval)
    exec bash scripts/run_feedback_opd_v2.sh "$@" ;;
  *) echo "usage: bash scripts/run_feedback_opd_fast.sh verify_judge|verify_model|benchmark|train_opd|train_control|status|perf|eval_dev|eval_selected|report|report_eval" ;;
esac
