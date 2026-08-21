#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
BENCHMARK=${BENCHMARK_MANIFEST:-/data/zjr/omniobject3d_reconviagen_style_omni200_20cat_render4_20260821_v1/manifest.json}
RUNTIME=${RUNTIME_INPUT_MANIFEST:-/data/zjr/omniobject3d_omni200_ss30k_slat30k_step30k_metrics_seed42_20260821_v1/00_exact_model_o_runtime/runtime_input_manifest.json}
OUT=${OUTPUT_ROOT:-/data/zjr/omniobject3d_omni200_strict_reconviagen_seed42_8gpu_20260821_v1}
REPAIR=${OUT}/repair_worker00_3gpu067_v1
GPUS_CSV=${REPAIR_GPUS:-6,7,0}
SEED=${EVAL_SEED:-42}
PRETRAINED=${PRETRAINED:-Stable-X/trellis-vggt-v0-2}

test -x "${PY}"
test -s "${BENCHMARK}"
test -s "${RUNTIME}"
IFS=, read -r -a GPUS <<<"${GPUS_CSV}"
if (( ${#GPUS[@]} != 3 )); then
  echo "ERROR: repair requires exactly three GPUs; got ${GPUS_CSV}" >&2
  exit 90
fi
mkdir -p "${REPAIR}/logs" "${REPAIR}/plan"

export PYTHONPATH="${PWD}:${PWD}/ReconViaGen:${PWD}/ReconViaGen/wheels/vggt"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export ATTN_BACKEND=flash_attn
export SPCONV_ALGO=native
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false

echo "===== P0 freeze worker0 repair plan: reuse 6, repair 19 ====="
"${PY}" - "${BENCHMARK}" "${RUNTIME}" "${OUT}" "${REPAIR}/plan" "${SEED}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

benchmark_path, runtime_path, out, plan_dir = map(Path, sys.argv[1:5])
seed = int(sys.argv[5])
benchmark = json.load(open(benchmark_path, encoding="utf-8"))
runtime = json.load(open(runtime_path, encoding="utf-8"))
assert benchmark["format"] == "reconviagen.omniobject3d_omni200_render_manifest.v1"
assert len(benchmark["objects"]) == 200
assert runtime["passed"] is True and len(runtime["objects"]) == 200

assigned = [
    f"{row['category']}:{row['uid']}"
    for index, row in enumerate(benchmark["objects"])
    if index % 8 == 0
]
assert len(assigned) == 25
partial_records = []
for path in sorted((out / "workers/worker_00/01_strict_reconviagen/meshes").glob("*/*/seed_42/result.json")):
    row = json.load(open(path, encoding="utf-8"))
    assert row["format"] == "pose_point_depth_mv.omni_real_reconviagen_inference.v1"
    assert row["seed"] == seed and row["object_key"] in assigned
    mesh = Path(row["mesh"])
    assert mesh.is_file()
    digest = hashlib.sha256(mesh.read_bytes()).hexdigest()
    assert digest == row["mesh_sha256"]
    partial_records.append(row)
partial_keys = [row["object_key"] for row in partial_records]
assert len(partial_keys) == len(set(partial_keys)) == 6, partial_keys
missing = [key for key in assigned if key not in set(partial_keys)]
assert len(missing) == 19, missing
assert missing[0] == "fig:omni_fig_041", missing[0]

bulk = missing[1:]
shards = [bulk[index::3] for index in range(3)]
assert [len(values) for values in shards] == [6, 6, 6]
files = {
    "problem_object.txt": [missing[0]],
    "bulk_0.txt": shards[0],
    "bulk_1.txt": shards[1],
    "bulk_2.txt": shards[2],
}
plan_dir.mkdir(parents=True, exist_ok=True)
for name, values in files.items():
    target = plan_dir / name
    content = "\n".join(values) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") != content:
        raise RuntimeError(f"existing repair plan differs: {target}")
    target.write_text(content, encoding="utf-8")
plan = {
    "format": "reconviagen.omniobject3d_worker0_repair_plan.v1",
    "benchmark_manifest": str(benchmark_path),
    "runtime_input_manifest": str(runtime_path),
    "seed": seed,
    "original_worker0_objects": assigned,
    "reused_partial_objects": partial_keys,
    "isolated_problem_object": missing[0],
    "bulk_shards": shards,
    "repair_object_count": len(missing),
}
target = plan_dir / "plan.json"
encoded = json.dumps(plan, indent=2, ensure_ascii=False) + "\n"
if target.exists() and target.read_text(encoding="utf-8") != encoded:
    raise RuntimeError(f"existing repair plan JSON differs: {target}")
target.write_text(encoded, encoding="utf-8")
print(json.dumps({"passed": True, "reused": 6, "repair": 19, "bulk_shards": [6, 6, 6], "isolated": missing[0]}, indent=2))
PY

run_keys() {
  local gpu=$1
  local key_file=$2
  local output=$3
  local label=$4
  local -a args=()
  while IFS= read -r key; do
    [[ -n "${key}" ]] && args+=(--object "${key}")
  done < "${key_file}"
  if (( ${#args[@]} == 0 )); then
    echo "ERROR: empty repair object list: ${key_file}" >&2
    return 90
  fi
  echo "[${label}] gpu=${gpu} objects=$((${#args[@]}/2)) output=${output}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PY}" -u -m manual_mesh_reconstruction.reconviagen \
      --runtime_input_manifest "${RUNTIME}" \
      --output_dir "${output}" \
      --pretrained "${PRETRAINED}" \
      --seeds "${SEED}" \
      --device cuda \
      "${args[@]}"
}

echo "===== P1 isolated former crash object + three balanced bulk shards ====="
(
  set +e
  run_keys "${GPUS[0]}" "${REPAIR}/plan/problem_object.txt" \
    "${REPAIR}/problem_object" problem_object
  problem_rc=$?
  set -e
  run_keys "${GPUS[0]}" "${REPAIR}/plan/bulk_0.txt" \
    "${REPAIR}/bulk_0" bulk_0
  if (( problem_rc != 0 )); then
    echo "ERROR: isolated problem object failed rc=${problem_rc}" >&2
    exit "${problem_rc}"
  fi
) >"${REPAIR}/logs/gpu${GPUS[0]}_problem_then_bulk0.log" 2>&1 &
pids=("$!")

for shard in 1 2; do
  gpu=${GPUS[$shard]}
  (
    run_keys "${gpu}" "${REPAIR}/plan/bulk_${shard}.txt" \
      "${REPAIR}/bulk_${shard}" "bulk_${shard}"
  ) >"${REPAIR}/logs/gpu${gpu}_bulk${shard}.log" 2>&1 &
  pids+=("$!")
done

echo "repair_pid0=${pids[0]} gpu=${GPUS[0]} isolated+6"
echo "repair_pid1=${pids[1]} gpu=${GPUS[1]} objects=6"
echo "repair_pid2=${pids[2]} gpu=${GPUS[2]} objects=6"

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "ERROR: repair process ${index} failed; other completed repairs are preserved" >&2
    failed=1
  fi
done
(( failed == 0 )) || exit 95

echo "===== P2 strict 200-object aggregate: original workers1-7 + partial6 + repaired19 ====="
"${PY}" - "${OUT}" "${RUNTIME}" "${BENCHMARK}" "${REPAIR}" "${SEED}" <<'PY'
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

out, runtime, benchmark, repair = map(Path, sys.argv[1:5])
seed = int(sys.argv[5])

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

b = json.load(open(benchmark, encoding="utf-8"))
expected_order = [f"{row['category']}:{row['uid']}" for row in b["objects"]]
expected = set(expected_order)
runtime_sha = sha(runtime)
records = []
bindings = []

for worker in range(1, 8):
    path = out / f"workers/worker_{worker:02d}/01_strict_reconviagen/inference_manifest.json"
    report = json.load(open(path, encoding="utf-8"))
    assert report["format"] == "pose_point_depth_mv.omni_real_reconviagen_inference_manifest.v1"
    assert report["passed"] is True and report["seeds"] == [seed]
    assert report["object_count"] == report["record_count"] == 25
    assert report["runtime_input_manifest_sha256"] == runtime_sha
    records.extend(report["objects"])
    bindings.append({"role": f"original_worker_{worker:02d}", "path": str(path), "sha256": sha(path)})

partial_paths = sorted((out / "workers/worker_00/01_strict_reconviagen/meshes").glob("*/*/seed_42/result.json"))
assert len(partial_paths) == 6
for path in partial_paths:
    records.append(json.load(open(path, encoding="utf-8")))
    bindings.append({"role": "original_worker_00_atomic_partial", "path": str(path), "sha256": sha(path)})

repair_manifests = [
    repair / "problem_object/inference_manifest.json",
    repair / "bulk_0/inference_manifest.json",
    repair / "bulk_1/inference_manifest.json",
    repair / "bulk_2/inference_manifest.json",
]
repair_counts = []
for path in repair_manifests:
    report = json.load(open(path, encoding="utf-8"))
    assert report["format"] == "pose_point_depth_mv.omni_real_reconviagen_inference_manifest.v1"
    assert report["passed"] is True and report["seeds"] == [seed]
    assert report["runtime_input_manifest_sha256"] == runtime_sha
    assert report["object_count"] == report["record_count"]
    repair_counts.append(report["record_count"])
    records.extend(report["objects"])
    bindings.append({"role": "worker_00_repair", "path": str(path), "sha256": sha(path)})
assert repair_counts == [1, 6, 6, 6], repair_counts

keys = [row["object_key"] for row in records]
assert len(records) == 200 and len(set(keys)) == 200 and set(keys) == expected
by_key = {row["object_key"]: row for row in records}
ordered = [by_key[key] for key in expected_order]
for row in ordered:
    assert row["seed"] == seed and row["view_count"] == 4
    assert row["runtime_input_manifest_sha256"] == runtime_sha
    mesh = Path(row["mesh"])
    assert mesh.is_file() and sha(mesh) == row["mesh_sha256"]

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
    "source_bindings": bindings,
    "repair": {
        "reason": "original worker0 CUDA illegal memory access after 6 atomic objects",
        "reused_original_worker0_atomic_records": 6,
        "repaired_records": 19,
        "repair_plan": str(repair / "plan/plan.json"),
        "repair_plan_sha256": sha(repair / "plan/plan.json"),
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
print(json.dumps({"passed": True, "objects": 200, "reused_partial": 6, "repaired": 19, "report": str(target)}, indent=2))
PY

echo "OMNI200 STRICT RECONVIAGEN REPAIR COMPLETE: ${OUT}/inference_aggregate_v1/report.json"
