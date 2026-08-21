#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
BENCHMARK=${BENCHMARK_MANIFEST:-/data/zjr/omniobject3d_reconviagen_style_omni200_20cat_render4_20260821_v1/manifest.json}
RUNTIME=${RUNTIME_INPUT_MANIFEST:-/data/zjr/omniobject3d_omni200_ss30k_slat30k_step30k_metrics_seed42_20260821_v1/00_exact_model_o_runtime/runtime_input_manifest.json}
OUT=${OUTPUT_ROOT:-/data/zjr/omniobject3d_omni200_strict_reconviagen_seed42_8gpu_20260821_v1}
FINAL_REPAIR=${OUT}/repair_remaining_per_object_2gpu67_v1
GPUS_CSV=${REPAIR_GPUS:-6,7}
SEED=${EVAL_SEED:-42}
PRETRAINED=${PRETRAINED:-Stable-X/trellis-vggt-v0-2}

IFS=, read -r -a GPUS <<<"${GPUS_CSV}"
if (( ${#GPUS[@]} != 2 )); then
  echo "ERROR: final repair requires exactly two GPUs" >&2
  exit 90
fi
mkdir -p "${FINAL_REPAIR}/logs" "${FINAL_REPAIR}/plan"

export PYTHONPATH="${PWD}:${PWD}/ReconViaGen:${PWD}/ReconViaGen/wheels/vggt"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export ATTN_BACKEND=flash_attn
export SPCONV_ALGO=native
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

echo "[$(date -u -Is)] waiting for first worker0 repair wave to finish"
while tmux has-session -t omni200reconrepair 2>/dev/null; do
  sleep 10
done

make_missing_plan() {
  local stage=$1
  "${PY}" - "${BENCHMARK}" "${OUT}" "${FINAL_REPAIR}/plan" "${stage}" "${SEED}" <<'PY'
import json
import sys
from pathlib import Path

benchmark, out, plan_dir = map(Path, sys.argv[1:4])
stage = str(sys.argv[4])
seed = int(sys.argv[5])
b = json.load(open(benchmark, encoding="utf-8"))
expected = [f"{row['category']}:{row['uid']}" for row in b["objects"]]
found = {}
for path in out.glob("**/seed_42/result.json"):
    row = json.load(open(path, encoding="utf-8"))
    if row.get("format") != "pose_point_depth_mv.omni_real_reconviagen_inference.v1":
        continue
    if int(row.get("seed", -1)) != seed:
        continue
    key = str(row["object_key"])
    if key in found:
        raise RuntimeError(f"duplicate successful ReconViaGen result: {key}")
    found[key] = str(path)
missing = [key for key in expected if key not in found]
plan_dir.mkdir(parents=True, exist_ok=True)
for worker in range(2):
    values = missing[worker::2]
    (plan_dir / f"{stage}_worker{worker}.txt").write_text(
        "\n".join(values) + ("\n" if values else ""), encoding="utf-8"
    )
print(json.dumps({"stage": stage, "complete": len(found), "missing": len(missing), "missing_keys": missing}, ensure_ascii=False))
PY
}

run_object_list() {
  local stage=$1
  local worker=$2
  local gpu=$3
  local list=${FINAL_REPAIR}/plan/${stage}_worker${worker}.txt
  local position=0
  local failures=0
  [[ -f "${list}" ]] || return 91
  while IFS= read -r key; do
    [[ -n "${key}" ]] || continue
    slug=$(printf '%03d' "${position}")
    destination=${FINAL_REPAIR}/${stage}_worker${worker}_${slug}
    echo "[${stage}] worker=${worker} gpu=${gpu} key=${key} output=${destination}"
    set +e
    CUDA_VISIBLE_DEVICES="${gpu}" \
      "${PY}" -u -m manual_mesh_reconstruction.reconviagen \
        --runtime_input_manifest "${RUNTIME}" \
        --output_dir "${destination}" \
        --pretrained "${PRETRAINED}" \
        --seeds "${SEED}" \
        --device cuda \
        --object "${key}"
    rc=$?
    set -e
    if (( rc != 0 )); then
      echo "ERROR: ${stage} key=${key} rc=${rc}; continuing with a fresh process" >&2
      failures=$((failures + 1))
    fi
    position=$((position + 1))
  done < "${list}"
  echo "[${stage}] worker=${worker} attempted=${position} failures=${failures}"
}

echo "===== P0 discover remaining objects after first repair wave ====="
make_missing_plan attempt1

echo "===== P1 per-object fresh-process repair on GPUs ${GPUS_CSV} ====="
pids=()
for worker in 0 1; do
  run_object_list attempt1 "${worker}" "${GPUS[$worker]}" \
    >"${FINAL_REPAIR}/logs/attempt1_worker${worker}_gpu${GPUS[$worker]}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "${pid}"; done

echo "===== P2 retry any still-missing object once on the opposite fresh GPU ====="
make_missing_plan attempt2
pids=()
for worker in 0 1; do
  opposite=$((1-worker))
  run_object_list attempt2 "${worker}" "${GPUS[$opposite]}" \
    >"${FINAL_REPAIR}/logs/attempt2_worker${worker}_gpu${GPUS[$opposite]}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "${pid}"; done

echo "===== P3 validate and aggregate exact 200 objects ====="
"${PY}" - "${OUT}" "${RUNTIME}" "${BENCHMARK}" "${FINAL_REPAIR}" "${SEED}" <<'PY'
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

out, runtime, benchmark, final_repair = map(Path, sys.argv[1:5])
seed = int(sys.argv[5])

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

b = json.load(open(benchmark, encoding="utf-8"))
expected_order = [f"{row['category']}:{row['uid']}" for row in b["objects"]]
runtime_sha = sha(runtime)
records = {}
bindings = []
for path in sorted(out.glob("**/seed_42/result.json")):
    row = json.load(open(path, encoding="utf-8"))
    if row.get("format") != "pose_point_depth_mv.omni_real_reconviagen_inference.v1":
        continue
    key = str(row["object_key"])
    if key in records:
        raise RuntimeError(f"duplicate successful result: {key}")
    assert row["seed"] == seed and row["view_count"] == 4
    assert row["runtime_input_manifest_sha256"] == runtime_sha
    mesh = Path(row["mesh"])
    assert mesh.is_file() and sha(mesh) == row["mesh_sha256"]
    records[key] = row
    bindings.append({"path": str(path), "sha256": sha(path)})
missing = [key for key in expected_order if key not in records]
extra = sorted(set(records) - set(expected_order))
if missing or extra or len(records) != 200:
    raise RuntimeError(f"strict ReconViaGen coverage differs: count={len(records)} missing={missing} extra={extra}")
ordered = [records[key] for key in expected_order]
manifest = {
    "format": "reconviagen.omniobject3d_omni200_strict_reconviagen_inference_aggregate.v1",
    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "passed": True,
    "method": "strict original ReconViaGen release: VGGT -> Stock SS -> Stock SLat -> Stock Mesh decoder",
    "pretrained": "Stable-X/trellis-vggt-v0-2",
    "seed": seed,
    "object_count": 200,
    "record_count": 200,
    "input_views_per_object": 4,
    "runtime_input_manifest": {"path": str(runtime), "sha256": runtime_sha},
    "benchmark_manifest": {"path": str(benchmark), "sha256": sha(benchmark)},
    "atomic_result_bindings": bindings,
    "repair": {
        "original_worker0_failure": "CUDA illegal memory access",
        "policy": "preserve valid atomic results; retry only missing keys in fresh per-object processes",
        "final_repair_root": str(final_repair),
    },
    "objects": ordered,
    "metric_or_target_consumed_during_inference": False,
    "output_frame": "reference-view canonical O diagnostic; transform_pose=False",
}
target = out / "inference_aggregate_v1/report.json"
target.parent.mkdir(parents=True, exist_ok=True)
temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
os.replace(temporary, target)
print(json.dumps({"passed": True, "objects": 200, "report": str(target)}, indent=2))
PY

echo "OMNI200 STRICT RECONVIAGEN FINAL REPAIR COMPLETE: ${OUT}/inference_aggregate_v1/report.json"
