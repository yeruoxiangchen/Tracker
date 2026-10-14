#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export FEEDBACK_DATA_DIR="${FEEDBACK_DATA_DIR:-$PROJECT_ROOT/datasets/processed/lingoqa_feedback_opd_40k_v2}"
export FEEDBACK_INITIAL_DIR="${FEEDBACK_INITIAL_DIR:-$PROJECT_ROOT/outputs/mpp_lingoqa_qformer_all_1epoch_v1}"
export FEEDBACK_PROBE_DIR="${FEEDBACK_PROBE_DIR:-$PROJECT_ROOT/outputs/feedback_teacher_compare_v2}"
export FEEDBACK_TRAIN_DIR="${FEEDBACK_TRAIN_DIR:-$PROJECT_ROOT/outputs/mpp_feedback_opd_40k_2epoch_v2}"
export FEEDBACK_CONTROL_DIR="${FEEDBACK_CONTROL_DIR:-$PROJECT_ROOT/outputs/mpp_feedback_sft_control_40k_2epoch_v2}"
export FEEDBACK_EVAL_DIR="${FEEDBACK_EVAL_DIR:-$PROJECT_ROOT/outputs/feedback_opd_eval_v2}"
export FEEDBACK_TRAIN_SIZE=40000 FEEDBACK_EPOCHS=2 FEEDBACK_MAX_STEPS=0
export FEEDBACK_MAX_QA_PER_SEGMENT=0 FEEDBACK_SAVE_EVERY=1000
export FEEDBACK_CHECKPOINT_STEPS="1000 2500"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}" GRAD_ACCUMULATION="${GRAD_ACCUMULATION:-4}"
export FEEDBACK_TEACHER_MODE="${FEEDBACK_TEACHER_MODE:-reference_only}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MASTER_PORT="${MASTER_PORT:-29697}"
export OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHON_BIN="${PYTHON_BIN:-/home/zjr/anaconda3/envs/mpp_llava/bin/python}"
UPSTREAM="${MPP_UPSTREAM_ROOT:-$PROJECT_ROOT/../mpp_llava_video_workspace/third_party/MPP-LLaVA}"
PERSISTENT="$PROJECT_ROOT/../mpp_llava_video_workspace/checkpoints"
CHECKPOINT_ROOT="${MPP_CHECKPOINT_ROOT:-/tmp/zjr_mpp_fast/checkpoints}"
if [[ ! -f "$CHECKPOINT_ROOT/mpp-qwen-next-sft/sharded/model.safetensors.index.json" ]]; then CHECKPOINT_ROOT="$PERSISTENT"; fi
VISUAL_ROOT="${MPP_VISUAL_CHECKPOINT_ROOT:-$PERSISTENT}"
BRIDGE_INIT="${MPP_BRIDGE_INIT:-$PROJECT_ROOT/checkpoints/visual_bridge_init.pt}"
CACHE_DIR="${MPP_EVA_CACHE_DIR:-/tmp/zjr_mpp_fast/cache/lingoqa_5f_eva_v1}"
export PYTHONPATH="$PROJECT_ROOT:$UPSTREAM${PYTHONPATH:+:$PYTHONPATH}"
report_args=(--data-dir "$FEEDBACK_DATA_DIR" --initial-dir "$FEEDBACK_INITIAL_DIR"
             --probe-dir "$FEEDBACK_PROBE_DIR" --train-dir "$FEEDBACK_TRAIN_DIR"
             --control-dir "$FEEDBACK_CONTROL_DIR" --eval-dir "$FEEDBACK_EVAL_DIR")
report() { "$PYTHON_BIN" scripts/report_feedback_opd_v2.py "${report_args[@]}" --action "$1" "${@:2}"; }
evaluate() {
  local branch="$1" label="$2" split="$3" data_path destination bundle key
  case "$split" in
    action|scenery) data_path="$FEEDBACK_DATA_DIR/dev_${split}.jsonl" ;;
    eval) data_path="$FEEDBACK_DATA_DIR/eval.jsonl" ;;
    *) echo "Unknown split: $split" >&2; exit 2 ;;
  esac
  local weights=()
  if [[ "$branch" == baseline ]]; then
    key=baseline
    weights=(--adapter "$FEEDBACK_INITIAL_DIR/final_adapter" --projector "$FEEDBACK_INITIAL_DIR/final_projector.pt"
             --qformer-adapter "$FEEDBACK_INITIAL_DIR/final_qformer_adapter.pt")
  else
    case "$branch" in
      sft_control) bundle="$FEEDBACK_CONTROL_DIR/$label" ;;
      feedback_opd) bundle="$FEEDBACK_TRAIN_DIR/$label" ;;
      *) echo "Unknown branch: $branch" >&2; exit 2 ;;
    esac
    case "$label" in checkpoint-001000|checkpoint-002500|final) ;; *) echo "Unknown checkpoint: $label" >&2; exit 2 ;; esac
    key="${branch}_${label}"
    weights=(--adapter "$bundle/adapter" --projector "$bundle/projector.pt" --qformer-adapter "$bundle/qformer_adapter.pt")
  fi
  destination="$FEEDBACK_EVAL_DIR/${key}_${split}"
  "$PYTHON_BIN" -u scripts/generate_answers.py --data "$data_path" --output-dir "$destination" \
    --cache-dir "$CACHE_DIR" --upstream-root "$UPSTREAM" --checkpoint-root "$CHECKPOINT_ROOT" \
    --visual-checkpoint-root "$VISUAL_ROOT" --bridge-init "$BRIDGE_INIT" --feature-stage eva \
    --precision-protocol fp32_trainables_v1 --qformer-mode auto --task-instruction-version legacy_v1 \
    --max-new-tokens "${FEEDBACK_MAX_NEW_TOKENS:-96}" --batch-size 1 "${weights[@]}"
  "$PYTHON_BIN" -u scripts/evaluate_answers.py --predictions "$destination/predictions.jsonl" \
    --output-dir "$destination/scored" --lingo-judge
}
case "${1:-help}" in
  prepare) bash scripts/run_feedback_opd.sh prepare ;;
  compare_teacher|validate_teacher)
    extra=()
    if [[ "$1" == validate_teacher ]]; then extra+=(--validate-only); fi
    "$PYTHON_BIN" -u scripts/compare_feedback_teachers.py --data "$FEEDBACK_DATA_DIR/train.jsonl" \
      --output-dir "$FEEDBACK_PROBE_DIR" --source-probe-dir "${FEEDBACK_SOURCE_PROBE_DIR:-$PROJECT_ROOT/outputs/feedback_opd_probe_v1}" \
      --initial-dir "$FEEDBACK_INITIAL_DIR" --cache-dir "$CACHE_DIR" --upstream-root "$UPSTREAM" \
      --checkpoint-root "$CHECKPOINT_ROOT" --visual-checkpoint-root "$VISUAL_ROOT" --bridge-init "$BRIDGE_INIT" \
      --seed "${FEEDBACK_SEED:-44}" --max-new-tokens "${FEEDBACK_MAX_NEW_TOKENS:-96}" \
      --max-txt-len "${FEEDBACK_MAX_TXT_LEN:-1536}" --judge-device "${FEEDBACK_JUDGE_DEVICE:-cpu}" \
      --cases "${FEEDBACK_COMPARE_CASES:-200}" "${extra[@]}"
    ;;
  train_control|train_opd) bash scripts/run_feedback_opd.sh "$1" ;;
  train_pair)
    if [[ -n "${FEEDBACK_RESUME_FROM:-}" ]]; then echo "Resume one branch at a time using train_control or train_opd." >&2; exit 2; fi
    if [[ ! -f "$FEEDBACK_CONTROL_DIR/summary.json" ]]; then bash scripts/run_feedback_opd.sh train_control; fi
    if [[ ! -f "$FEEDBACK_TRAIN_DIR/summary.json" ]]; then bash scripts/run_feedback_opd.sh train_opd; fi
    ;;
  eval) evaluate "${2:?branch required}" "${3:?checkpoint label required}" "${4:?split required}" ;;
  eval_dev)
    for split in action scenery; do evaluate baseline final "$split"; done
    for branch in sft_control feedback_opd; do
      for label in checkpoint-001000 checkpoint-002500 final; do
        for split in action scenery; do evaluate "$branch" "$label" "$split"; done
      done
    done
    report report
    ;;
  eval_selected)
    report report
    evaluate baseline final eval
    for branch in sft_control feedback_opd; do
      label="$(report selected --branch "$branch")"
      evaluate "$branch" "$label" eval
    done
    report report_eval
    ;;
  report) report report ;;
  report_eval) report report_eval ;;
  status) report status ;;
  *) echo "usage: bash scripts/run_feedback_opd_v2.sh prepare|compare_teacher|train_control|train_opd|train_pair|eval BRANCH LABEL SPLIT|eval_dev|eval_selected|report|report_eval|status" ;;
esac
