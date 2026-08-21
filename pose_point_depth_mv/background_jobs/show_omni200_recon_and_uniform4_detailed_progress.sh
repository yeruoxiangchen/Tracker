#!/usr/bin/env bash
set -u

cd /home/zjr/Tracker

PY=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
BENCH=/data/zjr/omniobject3d_reconviagen_style_omni200_20cat_render4_20260821_v1/manifest.json
RECON=/data/zjr/omniobject3d_omni200_strict_reconviagen_seed42_8gpu_20260821_v1
UDATA=/data/zjr/omniobject3d_reconviagen_style_omni200_20cat_uniform4_idx0_6_12_18_20260821_v1
UOUT=/data/zjr/omniobject3d_omni200_uniform4_ss30k_slat30k_step30k_metrics_seed42_4gpu1235_20260821_v1

date -u -Is
echo "================ Strict ReconViaGen ================"
"${PY}" - "${BENCH}" "${RECON}" <<'PY'
import json
import sys
from pathlib import Path

benchmark = json.load(open(sys.argv[1], encoding="utf-8"))
root = Path(sys.argv[2])
expected = [f"{row['category']}:{row['uid']}" for row in benchmark["objects"]]
found = {}
for path in root.glob("**/seed_42/result.json"):
    try:
        row = json.load(open(path, encoding="utf-8"))
        if row.get("format") == "pose_point_depth_mv.omni_real_reconviagen_inference.v1":
            found[str(row["object_key"])] = str(path)
    except Exception:
        pass
missing = [key for key in expected if key not in found]
aggregate = root / "inference_aggregate_v1/report.json"
print(f"valid_unique_results={len(found)}/200 ({len(found)/2:.1f}%)")
print(f"remaining={len(missing)}")
for key in missing:
    print(f"  missing: {key}")
print(f"aggregate={'COMPLETE' if aggregate.is_file() else 'pending'}")
PY

for log in "${RECON}"/repair_remaining_per_object_2gpu67_v1/logs/*.log; do
  [[ -f "${log}" ]] || continue
  printf '%s: ' "$(basename "${log}")"
  line=$(tail -c 65536 "${log}" | tr '\r' '\n' | grep -E '\[real_reconviagen\]|Sampling:|ERROR:|attempted=' | tail -n 1 || true)
  [[ -n "${line}" ]] && echo "${line}" || echo "loading model"
done
if tmux has-session -t omni200reconfinal 2>/dev/null; then
  echo "recon_tmux=RUNNING"
else
  echo "recon_tmux=EXITED_OR_COMPLETE"
fi

echo
echo "================ Uniform4 [0,6,12,18] ================"
rendered=0
[[ -d "${UDATA}/objects" ]] && rendered=$(find "${UDATA}/objects" -name report.json -type f | wc -l)
runtime=0
dino=0
meshes=0
metrics=0
[[ -d "${UOUT}/00_exact_model_o_runtime/objects" ]] && runtime=$(find "${UOUT}/00_exact_model_o_runtime/objects" -name report.json -type f | wc -l)
[[ -d "${UOUT}/workers" ]] && dino=$(find "${UOUT}/workers" -path '*/01_model_inputs/objects/*/*/report.json' -type f | wc -l)
[[ -d "${UOUT}/workers" ]] && meshes=$(find "${UOUT}/workers" -path '*/02_current_ss30k_slat30k/meshes/*/*.obj' -type f | wc -l)
[[ -d "${UOUT}/workers" ]] && metrics=$(find "${UOUT}/workers" -path '*/03_metrics/objects/*/metric.json' -type f | wc -l)

if [[ ! -s "${UDATA}/report.json" ]]; then
  echo "stage=render_uniform4"
  echo "rendered_objects=${rendered}/200 ($((rendered*100/200))%)"
  for log in "${UDATA}"/logs/worker_0[0-3]_gpu*.log; do
    [[ -f "${log}" ]] || continue
    # Ignore the stale first-attempt GPU4 log; the active fourth worker is GPU5.
    [[ "$(basename "${log}")" == "worker_03_gpu4.log" ]] && continue
    printf '%s: ' "$(basename "${log}")"
    grep -E '^\[omni200\]' "${log}" | tail -n 1 || echo "rendering first object"
  done
elif [[ ! -s "${UOUT}/aggregate_v1/report.json" ]]; then
  echo "stage=model_inference_or_metrics"
  echo "runtime=${runtime}/200 dino=${dino}/200 meshes=${meshes}/200 metrics=${metrics}/200"
  for log in "${UOUT}"/logs/worker_*.log; do
    [[ -f "${log}" ]] || continue
    printf '%s: ' "$(basename "${log}")"
    line=$(tail -c 65536 "${log}" | tr '\r' '\n' | grep -E '\[real_|\[official_|\[omni200:metric\]|COMPLETE|ERROR|Traceback' | tail -n 1 || true)
    [[ -n "${line}" ]] && echo "${line}" || echo "loading model"
  done
else
  echo "stage=COMPLETE"
  cat "${UOUT}/aggregate_v1/summary.txt"
fi
if tmux has-session -t omni200uniform4 2>/dev/null; then
  echo "uniform4_tmux=RUNNING"
else
  echo "uniform4_tmux=EXITED_OR_COMPLETE"
fi
