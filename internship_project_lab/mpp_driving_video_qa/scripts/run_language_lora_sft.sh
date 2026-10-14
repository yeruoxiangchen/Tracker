#!/usr/bin/env bash
# Matched pure-CE SFT with QKV + attention output + MLP LoRA.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zjr/anaconda3/envs/mpp_llava/bin/python}"
UPSTREAM="${MPP_UPSTREAM_ROOT:-$PROJECT_ROOT/../mpp_llava_video_workspace/third_party/MPP-LLaVA}"
PERSISTENT="$PROJECT_ROOT/../mpp_llava_video_workspace/checkpoints"
CHECKPOINT_ROOT="${MPP_CHECKPOINT_ROOT:-/tmp/zjr_mpp_fast/checkpoints}"
if [[ ! -f "$CHECKPOINT_ROOT/mpp-qwen-next-sft/sharded/model.safetensors.index.json" ]]; then
  CHECKPOINT_ROOT="$PERSISTENT"
fi
VISUAL_ROOT="${MPP_VISUAL_CHECKPOINT_ROOT:-$PERSISTENT}"
BRIDGE_INIT="${MPP_BRIDGE_INIT:-$PROJECT_ROOT/checkpoints/visual_bridge_init.pt}"
CACHE_DIR="${MPP_EVA_CACHE_DIR:-/tmp/zjr_mpp_fast/cache/lingoqa_5f_eva_v1}"
if [[ ! -d "$CACHE_DIR" ]]; then CACHE_DIR="$PROJECT_ROOT/datasets/cache/lingoqa_5f_eva_v1"; fi
DATA_DIR="${LANGUAGE_LORA_DATA_DIR:-$PROJECT_ROOT/datasets/processed/lingoqa_feedback_opd_40k_v2}"
INITIAL_DIR="${LANGUAGE_LORA_INITIAL_DIR:-$PROJECT_ROOT/outputs/mpp_lingoqa_qformer_all_1epoch_v1}"
CONTROL_DIR="${LANGUAGE_LORA_CONTROL_DIR:-$PROJECT_ROOT/outputs/mpp_feedback_sft_control_40k_2epoch_v2}"
CONTROL_EVAL_DIR="${LANGUAGE_LORA_CONTROL_EVAL_DIR:-$PROJECT_ROOT/outputs/feedback_opd_eval_v2}"
TRAIN_DIR="${LANGUAGE_LORA_TRAIN_DIR:-$PROJECT_ROOT/outputs/mpp_language_lora_decoder_40k_2epoch_v1}"
EVAL_DIR="${LANGUAGE_LORA_EVAL_DIR:-$PROJECT_ROOT/outputs/language_lora_eval_v1}"
export PYTHONPATH="$PROJECT_ROOT:$UPSTREAM${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MASTER_PORT="${MASTER_PORT:-29711}"
report_args=(--data-dir "$DATA_DIR" --initial-dir "$INITIAL_DIR" --control-dir "$CONTROL_DIR"
             --control-eval-dir "$CONTROL_EVAL_DIR" --train-dir "$TRAIN_DIR" --eval-dir "$EVAL_DIR")
report() { "$PYTHON_BIN" -u scripts/report_language_lora_sft.py "${report_args[@]}" --action "$1" "${@:2}"; }

evaluate() {
  local key="$1" split="$2" destination data_path bundle label selected
  local weights=()
  case "$split" in
    action|scenery) data_path="$DATA_DIR/dev_${split}.jsonl" ;;
    eval)
      if [[ ! -f "$EVAL_DIR/report.json" ]]; then
        echo "Run eval_dev to select checkpoints before the official evaluation." >&2; exit 2
      fi
      if [[ "$key" != parent ]]; then
        selected="$(report selected --model "${key%%_*}")"
        if [[ "$key" != "$selected" ]]; then echo "Official evaluation uses the dev-selected checkpoint." >&2; exit 2; fi
      fi
      data_path="$DATA_DIR/eval.jsonl"
      ;;
    *) echo "Unknown split: $split" >&2; exit 2 ;;
  esac
  case "$key" in
    parent)
      weights=(--adapter "$INITIAL_DIR/final_adapter" --projector "$INITIAL_DIR/final_projector.pt"
               --qformer-adapter "$INITIAL_DIR/final_qformer_adapter.pt")
      ;;
    qkv_*|decoder_*)
      label="${key#*_}"
      case "$label" in checkpoint-001000|checkpoint-002500|final) ;;
        *) echo "Unknown checkpoint: $label" >&2; exit 2 ;; esac
      if [[ "$key" == qkv_* ]]; then bundle="$CONTROL_DIR/$label"; else bundle="$TRAIN_DIR/$label"; fi
      weights=(--adapter "$bundle/adapter" --projector "$bundle/projector.pt"
               --qformer-adapter "$bundle/qformer_adapter.pt")
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
  check) report check ;;
  train)
    report check
    extra=()
    if [[ -n "${LANGUAGE_LORA_RESUME_FROM:-}" ]]; then extra+=(--resume-from "$LANGUAGE_LORA_RESUME_FROM"); fi
    "$PYTHON_BIN" -u -m torch.distributed.run --nproc_per_node=4 --master_port="$MASTER_PORT" \
      scripts/train_feedback_opd.py --training-mode sft_control --language-lora-scope decoder_linear \
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
    for branch in qkv decoder; do
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
    for branch in qkv decoder; do
      key="$(report selected --model "$branch")"
      evaluate "$key" eval
    done
    report report_eval
    ;;
  report_eval) report report_eval ;;
  *) echo "usage: bash scripts/run_language_lora_sft.sh check|train|status|eval MODEL action/scenery|eval_dev|report|eval_selected|report_eval|eval_status" ;;
esac
