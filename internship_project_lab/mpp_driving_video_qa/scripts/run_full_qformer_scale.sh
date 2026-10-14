#!/usr/bin/env bash
# Full original-QA scaling experiment; reuse the verified full-QFormer trainer.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zjr/anaconda3/envs/mpp_llava/bin/python}"
export FULL_QFORMER_DATA_DIR="${SCALE_DATA_DIR:-$PROJECT_ROOT/datasets/processed/lingoqa_qformer_all_v1}"
export FULL_QFORMER_TRAIN_DIR="${SCALE_TRAIN_DIR:-$PROJECT_ROOT/outputs/mpp_lingoqa_qformer_all_1epoch_v1}"
export FULL_QFORMER_EVAL_DIR="${SCALE_EVAL_DIR:-$PROJECT_ROOT/outputs/full_qformer_scale_eval_v1}"
export JOINT_DIR="${JOINT_DIR:-$PROJECT_ROOT/outputs/mpp_lingoqa_projector_lora_1epoch_v1}"
REFERENCE_TRAIN_DIR="${REFERENCE_TRAIN_DIR:-$PROJECT_ROOT/outputs/mpp_lingoqa_qformer_full100k_2epoch_v1}"
REFERENCE_EVAL_DIR="${REFERENCE_EVAL_DIR:-$PROJECT_ROOT/outputs/full_qformer_eval_v1}"
export SFT_EPOCHS=1 QFORMER_MODE=full
export MASTER_PORT="${MASTER_PORT:-29686}"
report_args=(--data-dir "$FULL_QFORMER_DATA_DIR" --train-dir "$FULL_QFORMER_TRAIN_DIR"
  --eval-dir "$FULL_QFORMER_EVAL_DIR" --joint-dir "$JOINT_DIR"
  --reference-train-dir "$REFERENCE_TRAIN_DIR" --reference-eval-dir "$REFERENCE_EVAL_DIR")
report() { "$PYTHON_BIN" scripts/report_full_qformer_scale.py "${report_args[@]}" --action "$1"; }
case "${1:-help}" in
  prepare)
    "$PYTHON_BIN" scripts/prepare_full_qformer_sft.py --all-train \
      --source-dir "${SCALE_SOURCE_DIR:-$PROJECT_ROOT/datasets/processed/lingoqa_full_v2}" \
      --output-dir "$FULL_QFORMER_DATA_DIR"
    ;;
  check_references) report check_references ;;
  cache|cache_status|status|train)
    if [[ "$1" == train ]]; then report check_references; fi
    bash scripts/run_full_qformer_sft.sh "$1"
    ;;
  eval_dev)
    report check_references
    for split in action scenery; do bash scripts/run_full_qformer_sft.sh eval epoch-01 "$split"; done
    report report
    ;;
  report) report report ;;
  eval_selected)
    # Refresh/validate development selection before touching official evaluation.
    report report
    selected="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_model"])' "$FULL_QFORMER_EVAL_DIR/report.json")"
    if [[ "$selected" == full ]]; then bash scripts/run_full_qformer_sft.sh eval epoch-01 eval; fi
    report report_eval
    ;;
  eval_status) report status ;;
  *) echo "usage: $0 prepare|check_references|cache|cache_status|train|status|eval_dev|report|eval_selected|eval_status" >&2; exit 2 ;;
esac
