#!/usr/bin/env bash
set -u

cd /home/zjr/Tracker
PY=/home/zjr/anaconda3/envs/reconviagen/bin/python
ROOT=${OUTPUT_ROOT:-/data/zjr/dorabench_dora299_trellis_s_m_seed42_trellis40_input0_9_19_29_8gpu_20260821_v1}

date -u -Is
echo "============================================================"
"${PY}" - "${ROOT}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
fmt = "reconviagen.dorabench_dora299_trellis_baseline_result.v1"
for baseline in ("trellis_s", "trellis_m"):
    branch = root / baseline
    valid = {}
    for path in branch.glob("**/seed_42/result.json"):
        try:
            row = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        if row.get("format") == fmt and row.get("passed") is True and row.get("baseline") == baseline:
            valid[str(row.get("object_key"))] = path
    inference = branch / "inference_aggregate_v1/report.json"
    metric_reports = list((branch / "metric_workers").glob("worker_*/metrics_report.json"))
    metric_count = 0
    for path in metric_reports:
        try:
            metric_count += int(json.load(open(path, encoding="utf-8")).get("object_count", 0))
        except Exception:
            pass
    final = branch / "aggregate_v1/report.json"
    print(f"{baseline}: meshes={len(valid)}/299 ({100*len(valid)/299:.1f}%)")
    print(f"  inference_aggregate={'COMPLETE' if inference.is_file() else 'pending'}")
    print(f"  metrics={metric_count}/299 workers={len(metric_reports)}/8")
    print(f"  final={'COMPLETE' if final.is_file() else 'pending'}")
    if final.is_file():
        report = json.load(open(final, encoding="utf-8"))
        print(f"  CD_mean={report['chamfer_distance']['mean']:.8f}")
        print(f"  Fscore_0p1_mean={report['fscore']['mean']:.8f}")
PY

echo "------------------------------------------------------------"
for baseline in trellis_s trellis_m; do
  branch=${ROOT}/${baseline}
  [[ -d "${branch}/logs" ]] || continue
  echo "${baseline} workers:"
  for log in "${branch}"/logs/primary_worker_*.log "${branch}"/logs/attempt*_worker*.log; do
    [[ -f "${log}" ]] || continue
    last=$(grep -aE '\[trellis_[sm]\]|COMPLETE|Traceback|CUDA error|RuntimeError|attempted=' "${log}" | tail -n 1)
    [[ -n "${last}" ]] || last=initializing
    echo "  $(basename "${log}"): ${last}"
  done
done

echo "------------------------------------------------------------"
if tmux has-session -t dora299trellis 2>/dev/null; then
  echo "tmux=dora299trellis RUNNING"
else
  echo "tmux=dora299trellis EXITED_OR_COMPLETE"
fi
pgrep -af '[e]valuate_dorabench299_trellis_baselines.*inference-worker' \
  || echo "no matching TRELLIS baseline inference process"
if [[ -f "${ROOT}/master.log" ]]; then
  echo "------------------------------------------------------------"
  echo "master.log latest:"
  tail -n 8 "${ROOT}/master.log"
fi
echo "------------------------------------------------------------"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits 2>/dev/null || true
