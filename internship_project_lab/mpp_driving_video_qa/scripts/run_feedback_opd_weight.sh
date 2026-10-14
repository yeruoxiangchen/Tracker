#!/usr/bin/env bash
# New weight-3 training; existing CE/weight-1 predictions stay read-only.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zjr/anaconda3/envs/mpp_llava/bin/python}"
DATA_DIR="${OPD_WEIGHT_DATA_DIR:-$PROJECT_ROOT/datasets/processed/lingoqa_feedback_opd_40k_v2}"
INITIAL_DIR="${OPD_WEIGHT_INITIAL_DIR:-$PROJECT_ROOT/outputs/mpp_lingoqa_qformer_all_1epoch_v1}"
CONTROL_DIR="${OPD_WEIGHT_CONTROL_DIR:-$PROJECT_ROOT/outputs/mpp_feedback_sft_control_40k_2epoch_v2}"
REFERENCE_DIR="${OPD_WEIGHT_REFERENCE_DIR:-$PROJECT_ROOT/outputs/mpp_feedback_opd_40k_2epoch_v2}"
REFERENCE_EVAL_DIR="${OPD_WEIGHT_REFERENCE_EVAL_DIR:-$PROJECT_ROOT/outputs/feedback_opd_eval_v2}"
TRAIN_DIR="${OPD_WEIGHT_TRAIN_DIR:-$PROJECT_ROOT/outputs/mpp_feedback_opd_weight3_40k_2epoch_v1}"
EVAL_DIR="${OPD_WEIGHT_EVAL_DIR:-$PROJECT_ROOT/outputs/feedback_opd_weight_eval_v1}"
UPSTREAM="${MPP_UPSTREAM_ROOT:-$PROJECT_ROOT/../mpp_llava_video_workspace/third_party/MPP-LLaVA}"
PERSISTENT="$PROJECT_ROOT/../mpp_llava_video_workspace/checkpoints"
CHECKPOINT_ROOT="${MPP_CHECKPOINT_ROOT:-/tmp/zjr_mpp_fast/checkpoints}"
if [[ ! -f "$CHECKPOINT_ROOT/mpp-qwen-next-sft/sharded/model.safetensors.index.json" ]]; then CHECKPOINT_ROOT="$PERSISTENT"; fi
VISUAL_ROOT="${MPP_VISUAL_CHECKPOINT_ROOT:-$PERSISTENT}"
BRIDGE_INIT="${MPP_BRIDGE_INIT:-$PROJECT_ROOT/checkpoints/visual_bridge_init.pt}"
CACHE_DIR="${MPP_EVA_CACHE_DIR:-/tmp/zjr_mpp_fast/cache/lingoqa_5f_eva_v1}"
export PYTHONPATH="$PROJECT_ROOT:$UPSTREAM${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MASTER_PORT="${MASTER_PORT:-29721}"
report_args=(--data-dir "$DATA_DIR" --initial-dir "$INITIAL_DIR" --control-dir "$CONTROL_DIR"
             --reference-dir "$REFERENCE_DIR" --reference-eval-dir "$REFERENCE_EVAL_DIR"
             --train-dir "$TRAIN_DIR" --eval-dir "$EVAL_DIR" --opd-weight 3)
report() { "$PYTHON_BIN" -u scripts/report_feedback_opd_weight.py "${report_args[@]}" --action "$1" "${@:2}"; }

training_layout() {
  local devices=() visible_world
  IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
  visible_world="${#devices[@]}"
  case "$visible_world" in 2|3|4) ;; *) echo 'Training requires 2, 3 or 4 explicitly visible GPUs.' >&2; exit 2 ;; esac
  local seen=() device existing
  for device in "${devices[@]}"; do
    if [[ -z "${device//[[:space:]]/}" ]]; then echo 'Empty CUDA device identifier.' >&2; exit 2; fi
    for existing in "${seen[@]}"; do
      if [[ "$device" == "$existing" ]]; then echo 'Duplicate CUDA device identifier.' >&2; exit 2; fi
    done
    seen+=("$device")
  done
  if [[ -n "${NPROC_PER_NODE:-}" && "$NPROC_PER_NODE" != "$visible_world" ]]; then
    echo 'NPROC_PER_NODE must match the number of CUDA_VISIBLE_DEVICES.' >&2; exit 2
  fi
  export NPROC_PER_NODE="$visible_world"
  export GRAD_ACCUMULATION="$(( (16 + visible_world - 1) / visible_world ))"
  # Four-GPU resumes retain the original configuration and sampler identity.
  export FEEDBACK_GLOBAL_BATCH_SIZE=16
  if [[ "$visible_world" == 4 ]]; then export FEEDBACK_GLOBAL_BATCH_SIZE=0; fi
}

check_training() {
  report check --launch-world-size "$NPROC_PER_NODE" \
    --launch-grad-accumulation "$GRAD_ACCUMULATION" \
    --launch-global-batch-size "$FEEDBACK_GLOBAL_BATCH_SIZE" "$@"
}

evaluate() {
  local key="$1" split="$2" destination data_path label arm bundle
  local weights=()
  case "$split" in
    action|scenery) data_path="$DATA_DIR/dev_${split}.jsonl" ;;
    eval)
      if [[ "$key" == parent ]]; then report selected --model opd3 >/dev/null;
      else
        arm="${key%%_*}"
        label="$(report selected --model "$arm")"
        if [[ "$key" != "${arm}_${label}" ]]; then echo 'Official evaluation requires the dev-selected checkpoint.' >&2; exit 2; fi
      fi
      data_path="$DATA_DIR/eval.jsonl"
      ;;
    *) echo "Unknown split: $split" >&2; exit 2 ;;
  esac
  case "$key" in
    parent) weights=(--adapter "$INITIAL_DIR/final_adapter" --projector "$INITIAL_DIR/final_projector.pt"
                     --qformer-adapter "$INITIAL_DIR/final_qformer_adapter.pt") ;;
    ce_*|opd1_*|opd3_*)
      arm="${key%%_*}"; label="${key#*_}"
      case "$label" in checkpoint-001000|checkpoint-002500|final) ;; *) echo "Unknown checkpoint: $label" >&2; exit 2 ;; esac
      case "$arm" in ce) bundle="$CONTROL_DIR/$label" ;; opd1) bundle="$REFERENCE_DIR/$label" ;; opd3) bundle="$TRAIN_DIR/$label" ;; esac
      weights=(--adapter "$bundle/adapter" --projector "$bundle/projector.pt" --qformer-adapter "$bundle/qformer_adapter.pt")
      ;;
    *) echo "Unknown model: $key" >&2; exit 2 ;;
  esac
  destination="$(report eval_path --model "$key" --split "$split")"
  if [[ -f "$destination/scored/summary.json" ]]; then
    printf 'Reusing scored answers: %s\n' "$destination"
    return
  fi
  "$PYTHON_BIN" -u scripts/generate_answers.py --data "$data_path" --output-dir "$destination" \
    --cache-dir "$CACHE_DIR" --upstream-root "$UPSTREAM" --checkpoint-root "$CHECKPOINT_ROOT" \
    --visual-checkpoint-root "$VISUAL_ROOT" --bridge-init "$BRIDGE_INIT" --feature-stage eva \
    --precision-protocol fp32_trainables_v1 --qformer-mode auto --task-instruction-version legacy_v1 \
    --max-new-tokens 96 --batch-size 1 "${weights[@]}"
  "$PYTHON_BIN" -u scripts/evaluate_answers.py --predictions "$destination/predictions.jsonl" \
    --output-dir "$destination/scored" --lingo-judge
}

case "${1:-help}" in
  check) training_layout; check_training ;;
  train)
    training_layout
    extra=()
    if [[ -n "${FEEDBACK_RESUME_FROM:-}" ]]; then extra+=(--resume-from "$FEEDBACK_RESUME_FROM"); fi
    check_training "${extra[@]}"
    export FEEDBACK_DATA_DIR="$DATA_DIR" FEEDBACK_INITIAL_DIR="$INITIAL_DIR"
    export FEEDBACK_CONTROL_DIR="$CONTROL_DIR" FEEDBACK_TRAIN_DIR="$TRAIN_DIR" FEEDBACK_EVAL_DIR="$EVAL_DIR"
    export FEEDBACK_TEACHER_MODE=reference_only FEEDBACK_CE_WEIGHT=1 FEEDBACK_OPD_WEIGHT=3 FEEDBACK_SEED=44
    export FEEDBACK_LR=2.5e-6 FEEDBACK_PROJECTOR_LR=2.5e-6 FEEDBACK_QFORMER_LR=5e-6
    export FEEDBACK_MAX_TXT_LEN=1536 FEEDBACK_MAX_NEW_TOKENS=96
    export MPP_CHECKPOINT_ROOT="$CHECKPOINT_ROOT" MPP_VISUAL_CHECKPOINT_ROOT="$VISUAL_ROOT"
    export MPP_EVA_CACHE_DIR="$CACHE_DIR" MPP_BRIDGE_INIT="$BRIDGE_INIT"
    exec bash scripts/run_feedback_opd_fast.sh train_opd
    ;;
  status|eval_status) report status ;;
  eval) evaluate "${2:?model required}" "${3:?split required}" ;;
  eval_dev)
    for split in action scenery; do evaluate parent "$split"; done
    for arm in ce opd1 opd3; do
      for label in checkpoint-001000 checkpoint-002500 final; do
        for split in action scenery; do evaluate "${arm}_${label}" "$split"; done
      done
    done
    report report
    ;;
  eval_selected)
    report report
    evaluate parent eval
    for arm in ce opd1 opd3; do
      label="$(report selected --model "$arm")"
      evaluate "${arm}_${label}" eval
    done
    report report_eval
    ;;
  report|report_eval) report "$1" ;;
  diagnose|diagnostic_report|diagnostic_status)
    export OPD_WEIGHT_DATA_DIR="$DATA_DIR" OPD_WEIGHT_INITIAL_DIR="$INITIAL_DIR"
    export OPD_WEIGHT_CONTROL_DIR="$CONTROL_DIR" OPD_WEIGHT_REFERENCE_DIR="$REFERENCE_DIR"
    export OPD_WEIGHT_REFERENCE_EVAL_DIR="$REFERENCE_EVAL_DIR" OPD_WEIGHT_TRAIN_DIR="$TRAIN_DIR" OPD_WEIGHT_EVAL_DIR="$EVAL_DIR"
    export MPP_CHECKPOINT_ROOT="$CHECKPOINT_ROOT" MPP_VISUAL_CHECKPOINT_ROOT="$VISUAL_ROOT"
    export MPP_EVA_CACHE_DIR="$CACHE_DIR" MPP_BRIDGE_INIT="$BRIDGE_INIT"
    exec "$PYTHON_BIN" -u scripts/diagnose_feedback_opd_weight.py --action "$1"
    ;;
  *) echo 'usage: bash scripts/run_feedback_opd_weight.sh check|train|status|eval MODEL SPLIT|eval_dev|eval_selected|report|report_eval|diagnose|diagnostic_report|diagnostic_status' ;;
esac
