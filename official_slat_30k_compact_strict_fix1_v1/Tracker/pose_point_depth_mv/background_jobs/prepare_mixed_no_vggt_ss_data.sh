#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
RUN=/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1
REAL_FULL=/data/zjr/native_v2_real500_domain_adapt_20260806_v2/cache_train_real_runtime_o_v2/lifting_manifest.json
REAL_DINO=${RUN}/lifting_real376_dino_only_v1
SYNTH=/data/zjr/native_ss_no_vggt_mixed1k_20260807_v1/lifting_train868_dino_only_v1/lifting_manifest.json
MIXED=${RUN}/manifests/mixed_ss_lifting_synth868_real376_v1.json
SS_PARENT_RUN=/data/zjr/native_v2_real500_domain_adapt_20260806_v2/ss_real_step1000_seed42_2gpu_v2
SS_PARENT=${SS_PARENT_RUN}/checkpoints/last.pt
SS_PARENT_REPORT=${SS_PARENT_RUN}/report.json
SS_CONTRACT=${RUN}/contracts/ss_real_full_ema_v1.json
STATE=${RUN}/logs/M1_prepare_ss_data.state
EXIT_CODE=${RUN}/logs/M1_prepare_ss_data.exit_code
LOCK=${RUN}/logs/M1_prepare_ss_data.lock

mkdir -p "${RUN}/logs" "${RUN}/manifests" "${RUN}/contracts"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "M1 refused: another SS data preparation holds ${LOCK}" >&2
  exit 99
fi

finish() {
  RC=$?
  trap - EXIT
  printf '%s\n' "${RC}" > "${EXIT_CODE}"
  printf 'finished_at=%s rc=%s\n' "$(date --iso-8601=seconds)" "${RC}" > "${STATE}"
  exit "${RC}"
}
trap finish EXIT
printf 'started_at=%s state=running\n' "$(date --iso-8601=seconds)" > "${STATE}"
rm -f "${EXIT_CODE}"

for REQUIRED in "${PY}" "${REAL_FULL}" "${SYNTH}" "${SS_PARENT}" "${SS_PARENT_REPORT}"; do
  test -s "${REQUIRED}"
done

if [ ! -s "${REAL_DINO}/lifting_manifest.json" ]; then
  RESUME=()
  if [ -e "${REAL_DINO}" ]; then RESUME=(--resume); fi
  "${PY}" -u -m pose_point_depth_mv.dataset_tools.derive_dino_only_lifting_cache \
    --source_manifest "${REAL_FULL}" \
    --output_dir "${REAL_DINO}" \
    --indices all \
    --ss_context_tokens 4096 \
    "${RESUME[@]}"
fi

"${PY}" -u -m pose_point_depth_mv.dataset_tools.build_mixed_no_vggt_manifest \
  lifting \
  --synthetic_manifest "${SYNTH}" \
  --real_manifest "${REAL_DINO}/lifting_manifest.json" \
  --output "${MIXED}" \
  --expected_synthetic_objects 868 \
  --expected_real_objects 376

"${PY}" -u -m pose_point_depth_mv.dataset_tools.build_real_full_no_vggt_migration_contract \
  --stage ss \
  --parent_checkpoint "${SS_PARENT}" \
  --parent_report "${SS_PARENT_REPORT}" \
  --output "${SS_CONTRACT}" \
  --min_real_objects 350

"${PY}" - "${MIXED}" <<'PY'
from pose_point_depth_mv.mixed_no_vggt_data import (
    DomainBalancedDistributedSampler,
    MixedPoseLiftingCacheDataset,
    validate_mixed_no_vggt_cache_contract,
)
import sys

dataset = MixedPoseLiftingCacheDataset(sys.argv[1])
contract = validate_mixed_no_vggt_cache_contract(dataset)
sampler = DomainBalancedDistributedSampler(
    dataset.rows, num_replicas=1, rank=0, seed=42
)
domains = [dataset.rows[index]["_mixed_domain"] for index in sampler]
assert domains.count("synthetic") == domains.count("real")
assert len(dataset.domain_datasets["synthetic"].rows) == 1417
assert len(dataset.domain_datasets["real"].rows) == 376
print({"passed": True, "mixed_samples": len(dataset), "domain_counts": {
    name: len(value.rows) for name, value in dataset.domain_datasets.items()
}, "sampler_epoch": {name: domains.count(name) for name in set(domains)},
"contract": contract["no_vggt"]})
PY
