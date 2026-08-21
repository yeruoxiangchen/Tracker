#!/usr/bin/env bash
set -euo pipefail

DEST=${DEST:-/data/zjr/Dora-Bench-256_20260821_v1}
REPO_URL=https://huggingface.co/datasets/aruichen/Dora-bench-256/resolve/main
ZIP_NAME=dora-bench-256.zip
ZIP_SHA256=bfbdadb1a99ddb6067d3b781c6f8e6bb01455bc24f3effb0240fe21e8f607ba2

mkdir -p "${DEST}"

download_one() {
    local name=$1
    local final="${DEST}/${name}"
    local partial="${final}.part"
    local url="${REPO_URL}/${name}?download=true"
    local attempt=0
    local before=0
    local after=0
    local rc=0

    if [ -s "${final}" ]; then
        echo "reuse completed file: ${final}"
        return 0
    fi

    echo "download: ${name}"
    while true; do
        attempt=$((attempt + 1))
        before=$(stat -c %s "${partial}" 2>/dev/null || echo 0)
        echo "download attempt=${attempt} resume_bytes=${before} file=${name}"

        # Do not use curl's internal --retry here.  With a long Xet-backed
        # transfer, an internal retry may reopen -o before recalculating the
        # resume offset and truncate a valid partial file.  Each outer-loop
        # invocation instead reads the current file size afresh.
        set +e
        curl \
            --fail \
            --location \
            --continue-at - \
            --connect-timeout 30 \
            --max-time 900 \
            --speed-time 180 \
            --speed-limit 1024 \
            --output "${partial}" \
            "${url}"
        rc=$?
        set -e

        after=$(stat -c %s "${partial}" 2>/dev/null || echo 0)
        if (( after < before )); then
            echo "ERROR: partial file shrank: before=${before} after=${after} file=${partial}" >&2
            exit 93
        fi
        if (( rc == 0 )); then
            break
        fi

        echo "download connection ended rc=${rc}; preserved_bytes=${after}; retrying in 10s"
        sleep 10
    done
    mv "${partial}" "${final}"
}

echo "============================================================"
echo "Dora-Bench-256 official download"
echo "destination: ${DEST}"
echo "started: $(date -Is)"
echo "============================================================"

for name in README.md Level1.json Level2.json Level3.json Level4.json Level_all.json; do
    download_one "${name}"
done
download_one "${ZIP_NAME}"

echo "===== verify official ZIP SHA256 ====="
printf '%s  %s\n' "${ZIP_SHA256}" "${ZIP_NAME}" > "${DEST}/SHA256SUMS.official.txt"
(
    cd "${DEST}"
    sha256sum -c SHA256SUMS.official.txt
)

echo "===== verify ZIP central directory and payload CRCs ====="
unzip -tq "${DEST}/${ZIP_NAME}"

echo "===== freeze downloaded metadata SHA256 ====="
(
    cd "${DEST}"
    sha256sum README.md Level1.json Level2.json Level3.json Level4.json Level_all.json \
        > SHA256SUMS.metadata.txt
)

printf '%s\n' \
    "format=dora_bench_256_download.v1" \
    "source=https://huggingface.co/datasets/aruichen/Dora-bench-256" \
    "zip_sha256=${ZIP_SHA256}" \
    "completed_at=$(date -Is)" \
    > "${DEST}/DOWNLOAD_COMPLETE.txt"

echo "============================================================"
echo "DORA-BENCH-256 DOWNLOAD PASS"
echo "completed: $(date -Is)"
echo "destination: ${DEST}"
du -sh "${DEST}"
echo "============================================================"
