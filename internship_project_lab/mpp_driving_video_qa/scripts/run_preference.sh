#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zjr/anaconda3/envs/mpp_llava/bin/python}"
UPSTREAM="${MPP_UPSTREAM_ROOT:-$PROJECT_ROOT/../mpp_llava_video_workspace/third_party/MPP-LLaVA}"
CHECKPOINT_ROOT="${MPP_CHECKPOINT_ROOT:-/tmp/zjr_mpp_fast/checkpoints}"
if [[ ! -f "$CHECKPOINT_ROOT/mpp-qwen-next-sft/sharded/model.safetensors.index.json" ]]; then
  CHECKPOINT_ROOT="$PROJECT_ROOT/../mpp_llava_video_workspace/checkpoints"
fi
CACHE_DIR="${MPP_QFORMER_CACHE_DIR:-/tmp/zjr_mpp_fast/cache/lingoqa_5f_qformer_v1}"
if [[ ! -d "$CACHE_DIR" ]]; then CACHE_DIR="$PROJECT_ROOT/datasets/cache/lingoqa_5f_qformer_v1"; fi
SOURCE_DIR="${PREFERENCE_SOURCE_DIR:-$PROJECT_ROOT/datasets/processed/lingoqa_full_v2}"
PROBE_DATA="${PREFERENCE_PROBE_DATA:-$PROJECT_ROOT/datasets/processed/lingoqa_answer_opsd_v1/probe.jsonl}"
PROBE_SCORED="${PREFERENCE_PROBE_SCORED:-$PROJECT_ROOT/outputs/answer_opsd_probe_v1/student/scored/scored_predictions.jsonl}"
DATA_DIR="${PREFERENCE_DATA_DIR:-$PROJECT_ROOT/datasets/processed/lingoqa_preference_v1}"
INIT_DIR="${PREFERENCE_INIT_DIR:-$PROJECT_ROOT/outputs/mpp_lingoqa_projector_lora_1epoch_v1}"
REFERENCE_DIR="${PREFERENCE_REFERENCE_DIR:-$PROJECT_ROOT/outputs/preference_reference_v1}"
TRAIN_DIR="${PREFERENCE_TRAIN_DIR:-$PROJECT_ROOT/outputs/mpp_lingoqa_dpo_v1}"
CONTROL_DIR="${PREFERENCE_CONTROL_DIR:-$PROJECT_ROOT/outputs/mpp_lingoqa_preference_sft_v1}"
EVAL_DIR="${PREFERENCE_EVAL_DIR:-$PROJECT_ROOT/outputs/preference_eval_v1}"
export PYTHONPATH="$PROJECT_ROOT:$UPSTREAM${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
common=(--data "$DATA_DIR/train.jsonl" --cache-dir "$CACHE_DIR" --upstream-root "$UPSTREAM"
        --checkpoint-root "$CHECKPOINT_ROOT" --init-adapter "$INIT_DIR/final_adapter"
        --projector "$INIT_DIR/final_projector.pt")
case "${1:-help}" in
  prepare)
    extra=()
    if [[ -n "${PREFERENCE_REVIEW_JSONL:-}" ]]; then extra+=(--review-jsonl "$PREFERENCE_REVIEW_JSONL"); fi
    "$PYTHON_BIN" -u scripts/prepare_preference_data.py --probe "$PROBE_DATA" --student-scored "$PROBE_SCORED" \
      --source-dir "$SOURCE_DIR" --output-dir "$DATA_DIR" --max-pairs "${PREFERENCE_MAX_PAIRS:-1000}" \
      --judge-margin "${PREFERENCE_JUDGE_MARGIN:-1}" --replay-fraction "${PREFERENCE_REPLAY_FRACTION:-0.25}" \
      --seed "${SEED:-42}" "${extra[@]}"
    ;;
  reference)
    "$PYTHON_BIN" -u scripts/cache_preference_reference.py "${common[@]}" --output-dir "$REFERENCE_DIR"
    ;;
  train_dpo|train_control)
    mode=dpo; destination="$TRAIN_DIR"
    if [[ "$1" == train_control ]]; then mode=sft_control; destination="$CONTROL_DIR"; fi
    extra=()
    if [[ -n "${PREFERENCE_RESUME_FROM:-}" ]]; then extra+=(--resume-from "$PREFERENCE_RESUME_FROM"); fi
    if [[ "${PREFERENCE_VERIFY_REFERENCE:-0}" == 1 ]]; then extra+=(--verify-reference); fi
    "$PYTHON_BIN" -u -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE:-4}" \
      --master_port="${MASTER_PORT:-29681}" scripts/train_preference.py "${common[@]}" \
      --reference-dir "$REFERENCE_DIR" --output-dir "$destination" --mode "$mode" \
      --epochs "${PREFERENCE_EPOCHS:-3}" --max-steps "${PREFERENCE_MAX_STEPS:-0}" \
      --grad-accumulation "${GRAD_ACCUMULATION:-4}" --lr "${PREFERENCE_LR:-1e-6}" \
      --projector-lr "${PREFERENCE_PROJECTOR_LR:-1e-6}" --beta "${PREFERENCE_BETA:-0.1}" \
      --chosen-ce-weight "${PREFERENCE_CHOSEN_CE_WEIGHT:-0.2}" --replay-ce-weight "${PREFERENCE_REPLAY_CE_WEIGHT:-0.2}" \
      --save-every "${PREFERENCE_SAVE_EVERY:-50}" --seed "${SEED:-42}" "${extra[@]}"
    ;;
  eval)
    kind="${2:?specify joint|sft_control|dpo}"; split="${3:?specify action|scenery|eval}"
    case "$split" in
      action|scenery) data_path="$DATA_DIR/dev_${split}.jsonl" ;;
      eval) data_path="$DATA_DIR/eval.jsonl" ;;
      *) echo "Unknown split: $split" >&2; exit 2 ;;
    esac
    bundle="${PREFERENCE_CHECKPOINT:-final}"
    case "$kind" in
      joint) adapter="$INIT_DIR/final_adapter"; projector="$INIT_DIR/final_projector.pt" ;;
      sft_control) adapter="$CONTROL_DIR/$bundle/adapter"; projector="$CONTROL_DIR/$bundle/projector.pt" ;;
      dpo) adapter="$TRAIN_DIR/$bundle/adapter"; projector="$TRAIN_DIR/$bundle/projector.pt" ;;
      *) echo "Unknown model: $kind" >&2; exit 2 ;;
    esac
    destination="$EVAL_DIR/${kind}_${split}"
    "$PYTHON_BIN" -u scripts/generate_answers.py --data "$data_path" --cache-dir "$CACHE_DIR" \
      --output-dir "$destination" --upstream-root "$UPSTREAM" --checkpoint-root "$CHECKPOINT_ROOT" \
      --feature-stage qformer --adapter "$adapter" --projector "$projector" \
      --task-instruction-version legacy_v1 --batch-size 1 --max-new-tokens 96
    "$PYTHON_BIN" -u scripts/evaluate_answers.py --predictions "$destination/predictions.jsonl" \
      --output-dir "$destination/scored" --lingo-judge
    ;;
  eval_dev)
    for kind in joint sft_control dpo; do
      for split in action scenery; do bash "$0" eval "$kind" "$split"; done
    done
    bash "$0" report
    ;;
  status|report)
    "$PYTHON_BIN" scripts/report_preference.py --action "$1" --data-dir "$DATA_DIR" \
      --reference-dir "$REFERENCE_DIR" --train-dir "$TRAIN_DIR" --control-dir "$CONTROL_DIR" \
      --init-dir "$INIT_DIR" --eval-dir "$EVAL_DIR" --checkpoint "${PREFERENCE_CHECKPOINT:-final}"
    ;;
  *) echo "Usage: bash scripts/run_preference.sh prepare|reference|train_dpo|train_control|eval MODEL SPLIT|eval_dev|status|report" ;;
esac
