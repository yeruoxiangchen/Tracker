#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/zjr/anaconda3/envs/mpp_llava/bin/python}"
EXTERNAL_PYTHON="${EVIDENCE_PYTHON:-/home/zjr/anaconda3/envs/chartvlm/bin/python}"
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
args=(--python "$PYTHON_BIN" --external-python "$EXTERNAL_PYTHON"
      --model "${EVIDENCE_MODEL:-/data/zjr/models/Qwen3-VL-8B-Instruct}"
      --run-dir "${EVIDENCE_RUN_DIR:-$PROJECT_ROOT/outputs/vlm_evidence_search_v1}"
      --data-dir "${EVIDENCE_DATA_DIR:-$PROJECT_ROOT/datasets/processed/lingoqa_vlm_evidence_v1}"
      --cache-root "${EVIDENCE_CACHE_ROOT:-$PROJECT_ROOT/datasets/cache/lingoqa_vlm_evidence_eva_v1}"
      --workers "${EVIDENCE_WORKERS:-4}" --cache-workers "${NPROC_PER_NODE:-4}"
      --cache-batch-size "${EVIDENCE_CACHE_BATCH_SIZE:-4}" --master-port "${MASTER_PORT:-29712}"
      --min-pixels "${EVIDENCE_MIN_PIXELS:-50176}" --max-pixels "${EVIDENCE_MAX_PIXELS:-200704}"
      --total-pixels "${EVIDENCE_TOTAL_PIXELS:-2007040}"
      --max-input-tokens "${EVIDENCE_MAX_INPUT_TOKENS:-6144}"
      --search-max-new-tokens "${EVIDENCE_MAX_NEW_TOKENS:-512}")
if [[ -n "${MPP_CHECKPOINT_ROOT:-}" ]]; then args+=(--checkpoints "$MPP_CHECKPOINT_ROOT"); fi
if [[ -n "${MPP_EVA_CACHE_DIR:-}" ]]; then args+=(--full-cache "$MPP_EVA_CACHE_DIR"); fi
if [[ -n "${MPP_VISUAL_CHECKPOINT_ROOT:-}" ]]; then args+=(--visual-checkpoints "$MPP_VISUAL_CHECKPOINT_ROOT"); fi
if [[ -n "${EVIDENCE_SOURCE_DIR:-}" ]]; then args+=(--source-dir "$EVIDENCE_SOURCE_DIR"); fi
if [[ -n "${EVIDENCE_MPP_INIT:-}" ]]; then args+=(--mpp-init "$EVIDENCE_MPP_INIT"); fi
if [[ "${EVIDENCE_INCLUDE_DUPLICATE:-0}" == 1 ]]; then args+=(--include-duplicate); fi
if [[ "${EVIDENCE_RETRY_FAILED:-0}" == 1 ]]; then args+=(--retry-failed); fi
case "${1:-help}" in
  dev) action=run; split=dev ;;
  official) action=run; split=eval ;;
  search_dev|assemble_dev|cache_dev|eval_dev) action="${1%_dev}"; split=dev ;;
  search_official|assemble_official|cache_official|eval_official) action="${1%_official}"; split=eval ;;
  prepare|freeze|status|report|preview) action="$1"; split="${2:-dev}" ;;
  *) echo "Usage: bash $0 prepare|dev|freeze|official|status|report [dev/eval]|preview [dev/eval]|search_dev|assemble_dev|cache_dev|eval_dev|search_official|assemble_official|cache_official|eval_official" >&2; exit 2 ;;
esac
if [[ "$action" == eval ]]; then action=evaluate; fi
exec "$PYTHON_BIN" -u "$PROJECT_ROOT/scripts/evidence_workflow.py" "$action" --split "$split" "${args[@]}"
