#!/usr/bin/env bash
set -u

cd /home/zjr/Tracker

PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
OUT=/data/zjr/dorabench_dora299_strict_reconviagen_seed42_trellis40_input0_9_19_29_8gpu_20260821_v1

date -u -Is
echo "============================================================"
"${PY}" - "${OUT}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
subset_path = root / "protocol/dora299_current_valid_subset.json"
expected = set()
if subset_path.is_file():
    subset = json.load(open(subset_path, encoding="utf-8"))
    expected = {f"{r['category']}:{r['uid']}" for r in subset.get("objects", [])}
found = {}
for path in root.glob("**/seed_42/result.json"):
    try:
        row = json.load(open(path, encoding="utf-8"))
    except Exception:
        continue
    if row.get("format") != "pose_point_depth_mv.omni_real_reconviagen_inference.v1":
        continue
    key = str(row.get("object_key", ""))
    if key in expected and row.get("passed") is True:
        found.setdefault(key, []).append(str(path))
duplicates = {k:v for k,v in found.items() if len(v) > 1}
complete = len(found)
print(f"strict_reconviagen_valid_unique={complete}/299 ({100.0*complete/299:.1f}%)")
print(f"remaining={299-complete}")
print(f"duplicate_keys={len(duplicates)}")
inference = root / "inference_aggregate_v1/report.json"
metric_reports = list(root.glob("metric_workers_model_o_v2/worker_*/metrics_report.json"))
metric_objects = 0
for path in metric_reports:
    try:
        metric_objects += int(json.load(open(path, encoding="utf-8")).get("object_count", 0))
    except Exception:
        pass
print(f"inference_aggregate={'COMPLETE' if inference.is_file() else 'pending'}")
print(f"metric_objects={metric_objects}/299")
print(f"metric_worker_reports={len(metric_reports)}/8")
legacy_v1 = root / "aggregate_v1/report.json"
final = root / "aggregate_model_o_v2/report.json"
print(f"final_aggregate={'COMPLETE' if final.is_file() else 'pending'}")
if final.is_file():
    report = json.load(open(final, encoding="utf-8"))
    print("metric_coordinate_contract=decoder-native/model-O identity v2")
    print(f"CD_mean={report['chamfer_distance']['mean']:.8f}")
    print(f"Fscore_0p1_mean={report['fscore']['mean']:.8f}")
elif legacy_v1.is_file():
    print("legacy_v1_metric=present_but_invalid_double_rotation; not reported")
PY

echo "------------------------------------------------------------"
for log in "${OUT}"/logs/worker_*.log; do
  [[ -f "${log}" ]] || continue
  name=$(basename "${log}")
  last=$(grep -aE '\[real_reconviagen\]|INFERENCE COMPLETE|Traceback|CUDA error|RuntimeError' "${log}" | tail -n 1)
  [[ -n "${last}" ]] || last=initializing
  echo "${name}: ${last}"
done
for log in "${OUT}"/repair/logs/*.log; do
  [[ -f "${log}" ]] || continue
  name=$(basename "${log}")
  last=$(grep -aE '\[real_reconviagen\]|attempted=|WARNING:|Traceback|CUDA error|RuntimeError' "${log}" | tail -n 1)
  [[ -n "${last}" ]] || last=initializing
  echo "repair/${name}: ${last}"
done

echo "------------------------------------------------------------"
if tmux has-session -t dora299recon 2>/dev/null; then
  echo "tmux=dora299recon RUNNING"
else
  echo "tmux=dora299recon EXITED_OR_COMPLETE"
fi
pgrep -af '[m]anual_mesh_reconstruction.reconviagen.*dorabench_dora300_ss30k' \
  || echo "no matching strict ReconViaGen process"
echo "------------------------------------------------------------"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits 2>/dev/null || true
