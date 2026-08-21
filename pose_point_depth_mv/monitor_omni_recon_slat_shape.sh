#!/usr/bin/env bash

# A quote-safe monitor for the Omni Holdout64 SLat15k/SLat25k comparison.
# Keep all shell syntax in this file so `watch` does not re-parse nested quotes
# or Bash-only here-strings through /bin/sh.

set -u

MODE=${1:-inference}
ROOT=${OMNI_SHAPE_ROOT:-/data/zjr/omni_holdout64_recon_vs_official_ss2k_slat15k25k_shape_20260816_v1}

service_state() {
  local unit=$1
  local state
  state=$(systemctl --user is-active "${unit}.service" 2>/dev/null || true)
  if [ -z "${state}" ]; then
    state=not-found-or-collected
  fi
  printf '%s' "${state}"
}

count_files() {
  local root=$1
  local pattern=$2
  find "${root}" -type f -name "${pattern}" 2>/dev/null | wc -l
}

date -u

case "${MODE}" in
  inference)
    for STEP in 15 25; do
      case "${STEP}" in
        15)
          PREFIX=01
          UNIT=tracker-omni64-official-slat15k-v1
          ;;
        25)
          PREFIX=02
          UNIT=tracker-omni64-official-slat25k-v1
          ;;
      esac

      OUT=${ROOT}/${PREFIX}_official_ss2k_slat${STEP}k_seed42
      LOG=${ROOT}/logs/O1_slat${STEP}k.log
      printf 'SLat%sk service: %s\n' "${STEP}" "$(service_state "${UNIT}")"
      printf '  completed SS coordinates: %s/64\n' "$(count_files "${OUT}/ss_coords" 'seed_42.npz')"
      printf '  completed Mesh: %s/64\n' "$(count_files "${OUT}/meshes" 'mesh_o.obj')"
      if [ -s "${OUT}/inference_manifest.json" ]; then
        echo '  manifest: COMPLETE'
      else
        echo '  manifest: pending'
      fi
      tail -n 1 "${LOG}" 2>/dev/null || true
    done
    ;;

  evaluation)
    UNIT=tracker-omni64-recon-slat15k25k-shape-v1
    EVAL=${ROOT}/03_normalized_and_proper_sim3_shape_v1
    LOG=${ROOT}/logs/O4_shape_eval.log
    printf 'service: %s\n' "$(service_state "${UNIT}")"
    printf 'completed objects: %s/64\n' "$(count_files "${EVAL}/records" '*.json')"
    if [ -s "${EVAL}/report.json" ]; then
      echo 'report: COMPLETE'
    else
      echo 'report: pending'
    fi
    tail -n 2 "${LOG}" 2>/dev/null || true
    ;;

  *)
    echo "usage: bash $0 {inference|evaluation}" >&2
    exit 2
    ;;
esac
