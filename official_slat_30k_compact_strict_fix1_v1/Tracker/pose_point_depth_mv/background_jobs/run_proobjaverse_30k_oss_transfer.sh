#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/zjr/Tracker}
PY=${PY:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
OSSUTIL=${OSSUTIL:-ossutil}

DATA_ROOT=${DATA_ROOT:-/data/zjr/ProObjaverse-300K-ReconViaGen-30K}
SOURCE_STATE=${SOURCE_STATE:-/data/zjr/ProObjaverse-300K-ReconViaGen-30K-state}
SELECTION=${SELECTION:-${SOURCE_STATE}/combined_audit/combined_selection_30k.json}
SOURCE_AUDIT=${SOURCE_AUDIT:-${SOURCE_STATE}/combined_audit/audit_report.json}
TRANSFER_STATE=${TRANSFER_STATE:-${SOURCE_STATE}/oss_transfer_c9175c52_20260816_v1}

OSS_ROOT=${OSS_ROOT:-oss://yxc814/proobjaverse-transfer/proobjaverse_30k_raw_c9175c52_20260816_v1}
HASH_WORKERS=${HASH_WORKERS:-4}
OSS_JOBS=${OSS_JOBS:-16}
OSS_PARALLEL=${OSS_PARALLEL:-4}
FINALIZE_ONLY=${FINALIZE_ONLY:-0}

TOOL=${PROJECT_ROOT}/pose_point_depth_mv/tools/proobjaverse_30k_oss_transfer.py
LOG=${TRANSFER_STATE}/transfer.log

mkdir -p "${TRANSFER_STATE}" "${TRANSFER_STATE}/ossutil_output"
exec 9>"${TRANSFER_STATE}/transfer.lock"
if ! flock -n 9; then
    echo "ERROR: another 30K OSS transfer process owns ${TRANSFER_STATE}/transfer.lock" >&2
    exit 88
fi

exec > >(tee -a "${LOG}") 2>&1

echo "============================================================"
echo "ProObjaverse frozen 30K raw -> OSS"
echo "============================================================"
echo "started       : $(date -Is)"
echo "host          : $(hostname)"
echo "data_root     : ${DATA_ROOT}"
echo "selection     : ${SELECTION}"
echo "source_audit  : ${SOURCE_AUDIT}"
echo "transfer_state: ${TRANSFER_STATE}"
echo "oss_root      : ${OSS_ROOT}"
echo

test -x "${PY}"
test -s "${TOOL}"
test -s "${SELECTION}"
test -s "${SOURCE_AUDIT}"
command -v "${OSSUTIL}" >/dev/null

LOCAL_INVENTORY=${TRANSFER_STATE}/local_inventory.jsonl
LOCAL_REPORT=${TRANSFER_STATE}/local_inventory_report.json
REMOTE_LISTING=${TRANSFER_STATE}/remote_payload_ls.txt
REMOTE_LISTING_ERR=${TRANSFER_STATE}/remote_payload_ls.stderr
REMOTE_AUDIT=${TRANSFER_STATE}/remote_inventory_audit.json
COMPLETION=${TRANSFER_STATE}/COMPLETE.json

case "${FINALIZE_ONLY}" in
    0|1) ;;
    *) echo "ERROR: FINALIZE_ONLY must be 0 or 1" >&2; exit 87 ;;
esac

if (( FINALIZE_ONLY == 0 )); then
    echo "===== P0: local immutable inventory (resume-safe) ====="
    "${PY}" -u "${TOOL}" prepare \
        --data_root "${DATA_ROOT}" \
        --selection "${SELECTION}" \
        --source_audit "${SOURCE_AUDIT}" \
        --state_dir "${TRANSFER_STATE}" \
        --hash_workers "${HASH_WORKERS}"
else
    echo "===== P0: reuse frozen local inventory (FINALIZE_ONLY=1) ====="
    test -s "${LOCAL_INVENTORY}"
    test -s "${LOCAL_REPORT}"
fi

readarray -t IDENTITIES < <("${PY}" - "${LOCAL_REPORT}" <<'PY'
import json
import sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
assert r["passed"] is True
print(r["dataset_content_sha256"])
print(r["selection_sha256"])
PY
)
DATASET_CONTENT_SHA256=${IDENTITIES[0]}
SELECTION_SHA256=${IDENTITIES[1]}

META="X-Oss-Meta-Dataset-Content-Sha256:${DATASET_CONTENT_SHA256}#X-Oss-Meta-Selection-Sha256:${SELECTION_SHA256}"

ALREADY_COMPLETED=0
if "${OSSUTIL}" stat "${OSS_ROOT}/COMPLETE.json" \
    >"${TRANSFER_STATE}/remote_completion.stat" 2>&1
then
    ALREADY_COMPLETED=1
    echo "Remote completion marker already exists; payload upload is skipped and full audit is rerun."
elif (( FINALIZE_ONLY == 0 )); then
    echo
    echo "===== P1: upload renders (recursive/update/snapshot/CRC64) ====="
    mkdir -p \
        "${TRANSFER_STATE}/snapshot_renders" \
        "${TRANSFER_STATE}/checkpoint_renders"
    "${OSSUTIL}" cp \
        "${DATA_ROOT}/renders_random_env" \
        "${OSS_ROOT}/payload/renders_random_env/" \
        -r -u \
        --disable-all-symlink \
        --disable-ignore-error \
        --snapshot-path "${TRANSFER_STATE}/snapshot_renders" \
        --checkpoint-dir "${TRANSFER_STATE}/checkpoint_renders" \
        --output-dir "${TRANSFER_STATE}/ossutil_output" \
        --bigfile-threshold 104857600 \
        --part-size 67108864 \
        --jobs "${OSS_JOBS}" \
        --parallel "${OSS_PARALLEL}" \
        --retry-times 20 \
        --meta "${META}"

    echo
    echo "===== P2: upload official lh-slats (recursive/update/snapshot/CRC64) ====="
    mkdir -p \
        "${TRANSFER_STATE}/snapshot_slats" \
        "${TRANSFER_STATE}/checkpoint_slats"
    "${OSSUTIL}" cp \
        "${DATA_ROOT}/lh-slats" \
        "${OSS_ROOT}/payload/lh-slats/" \
        -r -u \
        --disable-all-symlink \
        --disable-ignore-error \
        --snapshot-path "${TRANSFER_STATE}/snapshot_slats" \
        --checkpoint-dir "${TRANSFER_STATE}/checkpoint_slats" \
        --output-dir "${TRANSFER_STATE}/ossutil_output" \
        --bigfile-threshold 104857600 \
        --part-size 67108864 \
        --jobs "${OSS_JOBS}" \
        --parallel "${OSS_PARALLEL}" \
        --retry-times 20 \
        --meta "${META}"

    echo
    echo "===== P3: upload frozen source metadata ====="
    upload_metadata() {
        local SOURCE=$1
        local REMOTE_NAME=$2
        local FILE_SHA
        FILE_SHA=$(sha256sum "${SOURCE}" | awk '{print $1}')
        "${OSSUTIL}" cp \
            "${SOURCE}" \
            "${OSS_ROOT}/metadata/${REMOTE_NAME}" \
            -u \
            --retry-times 20 \
            --meta "X-Oss-Meta-Sha256:${FILE_SHA}#X-Oss-Meta-Dataset-Content-Sha256:${DATASET_CONTENT_SHA256}"
    }
    upload_metadata "${SELECTION}" combined_selection_30k.json
    upload_metadata "${SOURCE_AUDIT}" source_audit_report.json
    upload_metadata "${LOCAL_INVENTORY}" local_inventory.jsonl
    upload_metadata "${LOCAL_REPORT}" local_inventory_report.json
else
    echo
    echo "FINALIZE_ONLY=1: P1/P2/P3 payload and metadata uploads are skipped."
fi

echo
echo "===== P4: list and audit all remote payload objects ====="
"${OSSUTIL}" ls "${OSS_ROOT}/payload/" \
    >"${REMOTE_LISTING}.tmp" \
    2>"${REMOTE_LISTING_ERR}.tmp"
mv "${REMOTE_LISTING}.tmp" "${REMOTE_LISTING}"
mv "${REMOTE_LISTING_ERR}.tmp" "${REMOTE_LISTING_ERR}"

"${PY}" -u "${TOOL}" audit-remote \
    --inventory "${LOCAL_INVENTORY}" \
    --remote_listing "${REMOTE_LISTING}" \
    --oss_root "${OSS_ROOT}" \
    --output "${REMOTE_AUDIT}"

if (( ALREADY_COMPLETED == 0 )); then
    echo
    echo "===== P5: upload remote audit, then completion marker LAST ====="
    REMOTE_AUDIT_SHA=$(sha256sum "${REMOTE_AUDIT}" | awk '{print $1}')
    "${OSSUTIL}" cp \
        "${REMOTE_AUDIT}" \
        "${OSS_ROOT}/metadata/remote_inventory_audit.json" \
        -u \
        --retry-times 20 \
        --meta "X-Oss-Meta-Sha256:${REMOTE_AUDIT_SHA}#X-Oss-Meta-Dataset-Content-Sha256:${DATASET_CONTENT_SHA256}"

    "${PY}" -u "${TOOL}" complete \
        --local_report "${LOCAL_REPORT}" \
        --remote_report "${REMOTE_AUDIT}" \
        --oss_root "${OSS_ROOT}" \
        --output "${COMPLETION}"

    COMPLETION_SHA=$(sha256sum "${COMPLETION}" | awk '{print $1}')
    "${OSSUTIL}" cp \
        "${COMPLETION}" \
        "${OSS_ROOT}/COMPLETE.json" \
        -u \
        --retry-times 20 \
        --meta "X-Oss-Meta-Sha256:${COMPLETION_SHA}#X-Oss-Meta-Dataset-Content-Sha256:${DATASET_CONTENT_SHA256}"
fi

echo
echo "===== P6: final remote marker ====="
"${OSSUTIL}" stat "${OSS_ROOT}/COMPLETE.json"

echo
echo "============================================================"
echo "PROOBJAVERSE 30K OSS TRANSFER + AUDIT COMPLETE"
echo "finished: $(date -Is)"
echo "local report : ${LOCAL_REPORT}"
echo "remote audit : ${REMOTE_AUDIT}"
echo "completion   : ${COMPLETION}"
echo "log          : ${LOG}"
echo "============================================================"
