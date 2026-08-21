#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

export EVAL_GPUS=${EVAL_GPUS:-4,5,7}
source pose_point_depth_mv/background_jobs/source_proobjaverse_30k_dev64_abc_r_env.sh

if [[ ${EVAL_GPU_COUNT} -ne 3 ]]; then
  echo "ERROR: P2 three-GPU launcher requires exactly three GPUs; got ${EVAL_GPUS}" >&2
  exit 90
fi

test -s "${F39_PROTOCOL_DIR}/protocol.json"
test -s "${RELOCATED_PROTOCOL_DIR}/predicted_support_bridge.json"
test -s "${RELOCATED_PROTOCOL_DIR}/dev.json"
test -s "${TRAINING_SS_EVIDENCE}"
test -s "${STOCK_SLAT_FREEZE}"
test ! -e "${BRIDGE_COMPACT}"

"${PY}" - "${F39_PROTOCOL_DIR}/protocol.json" \
  "${RELOCATED_PROTOCOL_DIR}/predicted_support_bridge.json" "${PROTOCOL_SHA}" <<'PY'
import json
import sys

protocol_path, split_path, expected_sha = sys.argv[1:]
protocol = json.load(open(protocol_path, encoding="utf-8"))
split = json.load(open(split_path, encoding="utf-8"))
assert protocol["protocol_sha256"] == expected_sha
assert split["protocol_sha256"] == expected_sha
assert split["name"] == "predicted_support_bridge"
assert split["count"] == 32 and len(split["rows"]) == 32
assert len({row["uid"] for row in split["rows"]}) == 32
print({"passed": True, "split": "predicted_support_bridge", "objects": 32})
PY

mkdir -p "${BRIDGE_COMPACT}/logs"
IFS=, read -r -a GPUS <<<"${EVAL_GPUS}"

for i in 0 1 2; do
  session=ss30k_bridge_cache_${i}
  log=${BRIDGE_COMPACT}/logs/worker_${i}_gpu${GPUS[$i]}.log
  ! tmux has-session -t "${session}" 2>/dev/null
  tmux new-session -d -s "${session}" \
    "bash -lc 'cd ${PROJECT_ROOT} && CUDA_VISIBLE_DEVICES=${GPUS[$i]} ${PY} -u -m pose_point_depth_mv.prepare_proobjaverse_official_slat_compact_cache --split_manifest ${RELOCATED_PROTOCOL_DIR}/predicted_support_bridge.json --native_ss_report ${TRAINING_SS_EVIDENCE} --stock_slat_freeze ${STOCK_SLAT_FREEZE} --output_dir ${BRIDGE_COMPACT} --selected_views 8 --worker_index ${i} --worker_count 3 --materialize_only >> ${log} 2>&1'"
  printf 'worker=%d gpu=%s session=%s log=%s\n' \
    "${i}" "${GPUS[$i]}" "${session}" "${log}"
done

echo "P2 THREE-GPU WORKERS LAUNCHED"
echo "monitor: tail -F ${BRIDGE_COMPACT}/logs/*.log"
