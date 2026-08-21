#!/usr/bin/env bash
set -uo pipefail

# Batch-refine every timestamped phone reconstruction in the inclusive range.
# A successful existing refinement is skipped before optimize_o2w is launched.

PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
RECONSTRUCTION_ROOT=${RECONSTRUCTION_ROOT:-${PROJECT_ROOT}/pose_point_depth_mv/outputs2/手机AR第一阶段_全视图统一O_训练一致球面最远8帧/reconstructions}
START_SESSION=${START_SESSION:-20260821_025755_843}
END_SESSION=${END_SESSION:-20260821_093543_644}
BRANCH_NAME=${BRANCH_NAME:-01_training_spherical_farthest8}
GPU=${GPU:-4}
PYTHON_BIN=${PYTHON_BIN:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
LOG_DIR=${LOG_DIR:-${RECONSTRUCTION_ROOT}/../batch_optimize_o2w_logs_20260821}
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: [GPU=4] bash manual_mesh_reconstruction/batch_optimize_o2w_20260821.sh [--dry-run]

Environment overrides:
  PROJECT_ROOT, RECONSTRUCTION_ROOT, START_SESSION, END_SESSION,
  BRANCH_NAME, GPU, PYTHON_BIN, LOG_DIR
EOF
}

while (($#)); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ ! -d "${RECONSTRUCTION_ROOT}" ]]; then
  printf 'Reconstruction root does not exist: %s\n' "${RECONSTRUCTION_ROOT}" >&2
  exit 2
fi
if [[ ! -d "${RECONSTRUCTION_ROOT}/${START_SESSION}" ]]; then
  printf 'Start session does not exist: %s\n' "${START_SESSION}" >&2
  exit 2
fi
if [[ ! -d "${RECONSTRUCTION_ROOT}/${END_SESSION}" ]]; then
  printf 'End session does not exist: %s\n' "${END_SESSION}" >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  printf 'Python is not executable: %s\n' "${PYTHON_BIN}" >&2
  exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
  printf 'jq is required to validate completed reports.\n' >&2
  exit 2
fi

mapfile -t RECONSTRUCTIONS < <(
  find "${RECONSTRUCTION_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
    | LC_ALL=C sort \
    | while IFS= read -r session; do
        if [[ "${session}" =~ ^[0-9]{8}_[0-9]{6}_[0-9]{3}$ ]] \
          && [[ "${session}" == "${START_SESSION}" || "${session}" > "${START_SESSION}" ]] \
          && [[ "${session}" == "${END_SESSION}" || "${session}" < "${END_SESSION}" ]]; then
          printf '%s\n' "${RECONSTRUCTION_ROOT}/${session}"
        fi
      done
)

if ((${#RECONSTRUCTIONS[@]} == 0)); then
  printf 'No timestamped reconstruction sessions found in [%s, %s].\n' \
    "${START_SESSION}" "${END_SESSION}" >&2
  exit 2
fi

is_complete() {
  local report=$1
  local selected_npz
  [[ -s "${report}" ]] || return 1
  selected_npz=$(jq -er '
    select(.format == "manual_mesh_reconstruction.input_mask_o2w_refinement.v1")
    | select(.passed == true)
    | .selected_T_O2W_npz
    | select(type == "string" and length > 0)
  ' "${report}" 2>/dev/null) || return 1
  [[ -s "${selected_npz}" ]]
}

printf 'Found %d sessions in [%s, %s]; GPU=%s\n' \
  "${#RECONSTRUCTIONS[@]}" "${START_SESSION}" "${END_SESSION}" "${GPU}"

if ((DRY_RUN == 0)); then
  mkdir -p "${LOG_DIR}"
fi

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/ReconViaGen:${PROJECT_ROOT}/ReconViaGen/wheels/vggt${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}
export ATTN_BACKEND=${ATTN_BACKEND:-flash_attn}
export SPCONV_ALGO=${SPCONV_ALGO:-native}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba_cache}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

skipped=0
succeeded=0
failed=0
failed_sessions=()

cd "${PROJECT_ROOT}" || exit 2
for recon in "${RECONSTRUCTIONS[@]}"; do
  session=${recon##*/}
  report="${recon}/branches/${BRANCH_NAME}/04b_input_o2w_refinement/report.json"
  if is_complete "${report}"; then
    printf '[SKIP] %s (passed report and selected transform already exist)\n' "${session}"
    ((skipped += 1))
    continue
  fi

  if ((DRY_RUN == 1)); then
    printf '[WOULD RUN] %s\n' "${session}"
    continue
  fi

  log="${LOG_DIR}/${session}.log"
  printf '[RUN]  %s -> %s\n' "${session}" "${log}"
  "${PYTHON_BIN}" -u -m manual_mesh_reconstruction.optimize_o2w \
    --reconstruction_dir "${recon}" \
    --gpu "${GPU}" \
    --resume \
    2>&1 | tee "${log}"
  rc=${PIPESTATUS[0]}
  if ((rc == 0)); then
    printf '[OK]   %s\n' "${session}"
    ((succeeded += 1))
  else
    printf '[FAIL] %s (exit=%d, log=%s)\n' "${session}" "${rc}" "${log}" >&2
    ((failed += 1))
    failed_sessions+=("${session}")
  fi
done

if ((DRY_RUN == 1)); then
  printf 'Dry run complete: total=%d skipped=%d would_run=%d\n' \
    "${#RECONSTRUCTIONS[@]}" "${skipped}" "$(( ${#RECONSTRUCTIONS[@]} - skipped ))"
  exit 0
fi

printf 'Batch complete: total=%d skipped=%d succeeded=%d failed=%d\n' \
  "${#RECONSTRUCTIONS[@]}" "${skipped}" "${succeeded}" "${failed}"
if ((failed > 0)); then
  printf 'Failed sessions: %s\n' "${failed_sessions[*]}" >&2
  exit 1
fi
