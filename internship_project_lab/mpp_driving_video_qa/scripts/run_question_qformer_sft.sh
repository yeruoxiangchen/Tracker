#!/usr/bin/env bash
# InstructBLIP-style question-input bridge; matched ordinary answer CE.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zjr/anaconda3/envs/mpp_llava/bin/python}"
UPSTREAM="${MPP_UPSTREAM_ROOT:-$PROJECT_ROOT/../mpp_llava_video_workspace/third_party/MPP-LLaVA}"
PERSISTENT="$PROJECT_ROOT/../mpp_llava_video_workspace/checkpoints"
CHECKPOINT_ROOT="${MPP_CHECKPOINT_ROOT:-/tmp/zjr_mpp_fast/checkpoints}"
if [[ ! -f "$CHECKPOINT_ROOT/mpp-qwen-next-sft/sharded/model.safetensors.index.json" ]]; then CHECKPOINT_ROOT="$PERSISTENT"; fi
VISUAL_ROOT="${MPP_VISUAL_CHECKPOINT_ROOT:-$PERSISTENT}"
BRIDGE_INIT="${MPP_BRIDGE_INIT:-$PROJECT_ROOT/checkpoints/visual_bridge_init.pt}"
CACHE_DIR="${MPP_EVA_CACHE_DIR:-/tmp/zjr_mpp_fast/cache/lingoqa_5f_eva_v1}"
if [[ ! -d "$CACHE_DIR" ]]; then CACHE_DIR="$PROJECT_ROOT/datasets/cache/lingoqa_5f_eva_v1"; fi
DATA_DIR="${QUESTION_QFORMER_DATA_DIR:-$PROJECT_ROOT/datasets/processed/lingoqa_feedback_opd_40k_v2}"
INITIAL_DIR="${QUESTION_QFORMER_INITIAL_DIR:-$PROJECT_ROOT/outputs/mpp_lingoqa_qformer_all_1epoch_v1}"
CONTROL_DIR="${QUESTION_QFORMER_CONTROL_DIR:-$PROJECT_ROOT/outputs/mpp_feedback_sft_control_40k_2epoch_v2}"
CONTROL_EVAL_DIR="${QUESTION_QFORMER_CONTROL_EVAL_DIR:-$PROJECT_ROOT/outputs/feedback_opd_eval_v2}"
TRAIN_DIR="${QUESTION_QFORMER_TRAIN_DIR:-$PROJECT_ROOT/outputs/mpp_question_qformer_40k_2epoch_v1}"
EVAL_DIR="${QUESTION_QFORMER_EVAL_DIR:-$PROJECT_ROOT/outputs/question_qformer_eval_v1}"
QUESTION_MAX_LEN="${QUESTION_QFORMER_MAX_LEN:-128}"
export PYTHONPATH="$PROJECT_ROOT:$UPSTREAM${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export MASTER_PORT="${MASTER_PORT:-29731}"
report_args=(--data-dir "$DATA_DIR" --initial-dir "$INITIAL_DIR" --control-dir "$CONTROL_DIR"
             --control-eval-dir "$CONTROL_EVAL_DIR" --train-dir "$TRAIN_DIR" --eval-dir "$EVAL_DIR"
             --bert-dir "$VISUAL_ROOT/ckpt/bert-base-uncased" --cache-dir "$CACHE_DIR" --question-max-len "$QUESTION_MAX_LEN")
report() { "$PYTHON_BIN" -u scripts/report_question_qformer_sft.py "${report_args[@]}" --action "$1" "${@:2}"; }

evaluate() {
  local key="$1" split="$2" data_path bundle destination selected
  local weights=()
  case "$split" in
    action|scenery) data_path="$DATA_DIR/dev_${split}.jsonl" ;;
    eval)
      if [[ ! -f "$EVAL_DIR/report.json" ]]; then echo "Complete eval_dev before official evaluation." >&2; exit 2; fi
      if [[ "$key" != parent ]]; then
        selected="$(report selected --model "${key%%_*}")"
        if [[ "$key" != "$selected" ]]; then echo "Official evaluation must follow development selection." >&2; exit 2; fi
      fi
      data_path="$DATA_DIR/eval.jsonl" ;;
    *) echo "Unknown split: $split" >&2; exit 2 ;;
  esac
  case "$key" in
    parent) weights=(--adapter "$INITIAL_DIR/final_adapter" --projector "$INITIAL_DIR/final_projector.pt"
                     --qformer-adapter "$INITIAL_DIR/final_qformer_adapter.pt") ;;
    visual_*|question_*)
      label="${key#*_}"
      case "$label" in checkpoint-001000|checkpoint-002500|final) ;; *) echo "Unknown checkpoint: $label" >&2; exit 2 ;; esac
      if [[ "$key" == visual_* ]]; then bundle="$CONTROL_DIR/$label"; else bundle="$TRAIN_DIR/$label"; fi
      weights=(--adapter "$bundle/adapter" --projector "$bundle/projector.pt" --qformer-adapter "$bundle/qformer_adapter.pt")
      ;;
    *) echo "Unknown model: $key" >&2; exit 2 ;;
  esac
  destination="$(report eval_path --model "$key" --split "$split")"
  if [[ -f "$destination/scored/summary.json" ]]; then printf 'Reusing scored answers: %s\n' "$destination"; return; fi
  "$PYTHON_BIN" -u scripts/generate_answers.py --data "$data_path" --output-dir "$destination" \
    --cache-dir "$CACHE_DIR" --upstream-root "$UPSTREAM" --checkpoint-root "$CHECKPOINT_ROOT" \
    --visual-checkpoint-root "$VISUAL_ROOT" --bridge-init "$BRIDGE_INIT" --feature-stage eva \
    --precision-protocol fp32_trainables_v1 --qformer-mode auto --qformer-text-input auto \
    --task-instruction-version legacy_v1 --max-new-tokens 96 --batch-size 1 "${weights[@]}"
  "$PYTHON_BIN" -u scripts/evaluate_answers.py --predictions "$destination/predictions.jsonl" \
    --output-dir "$destination/scored" --lingo-judge
}

case "${1:-help}" in
  check) report check ;;
  smoke)
    "$PYTHON_BIN" -u scripts/smoke_question_qformer.py --data "$DATA_DIR/train.jsonl" \
      --output-dir "${QUESTION_QFORMER_SMOKE_DIR:-$PROJECT_ROOT/outputs/question_qformer_real_smoke_v1}" \
      --initial-dir "$INITIAL_DIR" --cache-dir "$CACHE_DIR" --upstream-root "$UPSTREAM" \
      --checkpoint-root "$CHECKPOINT_ROOT" --visual-checkpoint-root "$VISUAL_ROOT" --bridge-init "$BRIDGE_INIT" \
      --qformer-question-max-len "$QUESTION_MAX_LEN" --max-txt-len 1536 --seed 44
    ;;
  train)
    report check
    extra=()
    if [[ -n "${QUESTION_QFORMER_RESUME_FROM:-}" ]]; then extra+=(--resume-from "$QUESTION_QFORMER_RESUME_FROM"); fi
    "$PYTHON_BIN" -u -m torch.distributed.run --nproc_per_node=4 --master_port="$MASTER_PORT" \
      scripts/train_feedback_opd.py --training-mode sft_control --language-lora-scope qkv \
      --qformer-text-input question --qformer-question-max-len "$QUESTION_MAX_LEN" \
      --data "$DATA_DIR/train.jsonl" --output-dir "$TRAIN_DIR" --initial-dir "$INITIAL_DIR" \
      --cache-dir "$CACHE_DIR" --upstream-root "$UPSTREAM" --checkpoint-root "$CHECKPOINT_ROOT" \
      --visual-checkpoint-root "$VISUAL_ROOT" --bridge-init "$BRIDGE_INIT" \
      --task-instruction-version legacy_v1 --teacher-mode reference_only --seed 44 --cpu-threads 1 \
      --max-txt-len 1536 --max-new-tokens 96 --epochs 2 --max-steps 0 --grad-accumulation 4 \
      --save-every 1000 --checkpoint-steps 1000 2500 --lr 2.5e-6 --projector-lr 2.5e-6 \
      --qformer-lr 5e-6 --weight-decay 0.01 --warmup-ratio 0.03 --ce-weight 1 "${extra[@]}"
    ;;
  status|eval_status) report status ;;
  eval) evaluate "${2:?model required}" "${3:?split required}" ;;
  eval_dev)
    for split in action scenery; do evaluate parent "$split"; done
    for branch in visual question; do
      for label in checkpoint-001000 checkpoint-002500 final; do
        for split in action scenery; do evaluate "${branch}_${label}" "$split"; done
      done
    done
    report report
    ;;
  report) report report ;;
  eval_selected)
    report report
    evaluate parent eval
    for branch in visual question; do evaluate "$(report selected --model "$branch")" eval; done
    report report_eval
    ;;
  report_eval) report report_eval ;;
  *) echo "usage: bash scripts/run_question_qformer_sft.sh check|smoke|train|status|eval MODEL SPLIT|eval_dev|report|eval_selected|report_eval|eval_status" ;;
esac
