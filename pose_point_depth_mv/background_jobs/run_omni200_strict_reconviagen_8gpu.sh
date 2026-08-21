#!/usr/bin/env bash
set -euo pipefail

cd /home/zjr/Tracker

PY=${PYTHON:-/home/zjr/anaconda3/envs/reconviagen/bin/python}
BENCHMARK=${BENCHMARK_MANIFEST:-/data/zjr/omniobject3d_reconviagen_style_omni200_20cat_render4_20260821_v1/manifest.json}
RUNTIME=${RUNTIME_INPUT_MANIFEST:-/data/zjr/omniobject3d_omni200_ss30k_slat30k_step30k_metrics_seed42_20260821_v1/00_exact_model_o_runtime/runtime_input_manifest.json}
OUT=${OUTPUT_ROOT:-/data/zjr/omniobject3d_omni200_strict_reconviagen_seed42_8gpu_20260821_v1}
# GPU4 hosts the phone/SAM2 service, so it is launched last and uses the
# CPU-offload path.  The other seven workers keep their models GPU-resident.
# Sequential startup prevents eight simultaneous 13+ GiB model loads from
# exhausting the 125 GiB host RAM before the weights move to CUDA.
GPUS_CSV=${EVAL_GPUS:-0,1,2,3,5,6,7,4}
LOW_VRAM_GPUS=${LOW_VRAM_GPUS:-4}
STARTUP_TIMEOUT_SECONDS=${STARTUP_TIMEOUT_SECONDS:-360}
SEED=${EVAL_SEED:-42}
PRETRAINED=${PRETRAINED:-Stable-X/trellis-vggt-v0-2}

test -x "${PY}"
test -s "${BENCHMARK}"
test -s "${RUNTIME}"

IFS=, read -r -a GPU_ARRAY <<<"${GPUS_CSV}"
WORKERS=${#GPU_ARRAY[@]}
if (( WORKERS != 8 )); then
  echo "ERROR: this frozen run requires exactly 8 workers; got ${WORKERS}" >&2
  exit 90
fi
if [[ "$(printf '%s\n' "${GPU_ARRAY[@]}" | sort -u | wc -l)" -ne 8 ]]; then
  echo "ERROR: EVAL_GPUS contains duplicate GPU indices" >&2
  exit 91
fi

mkdir -p "${OUT}/logs" "${OUT}/workers"
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

"${PY}" - "${BENCHMARK}" "${RUNTIME}" <<'PY'
import json, sys
benchmark = json.load(open(sys.argv[1], encoding="utf-8"))
runtime = json.load(open(sys.argv[2], encoding="utf-8"))
assert benchmark["format"] == "reconviagen.omniobject3d_omni200_render_manifest.v1"
assert benchmark["object_count"] == 200
assert len(benchmark["objects"]) == 200
assert runtime["format"] == "pose_point_depth_mv.omni_real_runtime_input_manifest.v3"
assert runtime["passed"] is True
assert runtime["selected_object_count"] == 200
assert runtime["completed_object_count"] == 200
assert len(runtime["objects"]) == 200
benchmark_keys = [f"{row['category']}:{row['uid']}" for row in benchmark["objects"]]
runtime_keys = [str(row["object_key"]) for row in runtime["objects"]]
assert set(benchmark_keys) == set(runtime_keys)
assert len(set(benchmark_keys)) == 200
assert all(int(row["selected_view_count"]) == 4 for row in runtime["objects"])
print({"passed": True, "objects": 200, "views_per_object": 4})
PY

echo "===== strict original ReconViaGen: 200 objects / 4 frozen views / seed ${SEED} ====="
echo "workers=${WORKERS} GPUs=${GPUS_CSV} output=${OUT}"
echo "low_vram_physical_gpus=${LOW_VRAM_GPUS:-none} startup=sequential-until-first-sampling"

pids=()
for worker in "${!GPU_ARRAY[@]}"; do
  gpu=${GPU_ARRAY[$worker]}
  worker_name=$(printf 'worker_%02d' "${worker}")
  worker_root=${OUT}/workers/${worker_name}
  log=${OUT}/logs/${worker_name}_gpu${gpu}.log
  mkdir -p "${worker_root}"

  mapfile -t object_keys < <(
    "${PY}" - "${BENCHMARK}" "${worker}" "${WORKERS}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
worker, workers = map(int, sys.argv[2:])
for index, row in enumerate(p["objects"]):
    if index % workers == worker:
        print(f"{row['category']}:{row['uid']}")
PY
  )
  if (( ${#object_keys[@]} != 25 )); then
    echo "ERROR: worker ${worker} object count differs: ${#object_keys[@]} != 25" >&2
    exit 92
  fi
  object_args=()
  for key in "${object_keys[@]}"; do
    object_args+=(--object "${key}")
  done
  low_vram_args=()
  case ",${LOW_VRAM_GPUS}," in
    *,"${gpu}",*) low_vram_args+=(--low_vram) ;;
  esac

  (
    set -euo pipefail
    echo "[worker ${worker}] objects=25 physical_gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
      "${PY}" -u -m manual_mesh_reconstruction.reconviagen \
        --runtime_input_manifest "${RUNTIME}" \
        --output_dir "${worker_root}/01_strict_reconviagen" \
        --pretrained "${PRETRAINED}" \
        --seeds "${SEED}" \
        --device cuda \
        "${low_vram_args[@]}" \
        "${object_args[@]}"
    echo "[worker ${worker}] INFERENCE COMPLETE"
  ) >"${log}" 2>&1 &
  pids+=("$!")
  pid=$!
  mode=gpu_resident
  (( ${#low_vram_args[@]} )) && mode=low_vram
  echo "worker=${worker} gpu=${gpu} pid=${pid} objects=25 mode=${mode} log=${log}"

  startup_begin=$SECONDS
  while ! grep -q 'Sampling:' "${log}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "ERROR: worker ${worker} exited during model startup; log=${log}" >&2
      wait "${pid}" || true
      exit 93
    fi
    if (( SECONDS - startup_begin > STARTUP_TIMEOUT_SECONDS )); then
      echo "ERROR: worker ${worker} did not reach sampling within ${STARTUP_TIMEOUT_SECONDS}s" >&2
      exit 94
    fi
    sleep 2
  done
  echo "worker=${worker} gpu=${gpu} startup_gate=PASS elapsed=$((SECONDS-startup_begin))s"
done

failed=0
for worker in "${!pids[@]}"; do
  if ! wait "${pids[$worker]}"; then
    echo "ERROR: worker ${worker} failed; completed object outputs are preserved" >&2
    failed=1
  fi
done
if (( failed != 0 )); then
  exit 95
fi

"${PY}" - "${OUT}" "${RUNTIME}" "${BENCHMARK}" "${SEED}" <<'PY'
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

out = Path(sys.argv[1]).resolve()
runtime = Path(sys.argv[2]).resolve()
benchmark = Path(sys.argv[3]).resolve()
seed = int(sys.argv[4])

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

worker_paths = sorted(out.glob("workers/worker_*/01_strict_reconviagen/inference_manifest.json"))
assert len(worker_paths) == 8, len(worker_paths)
records = []
bindings = []
for path in worker_paths:
    report = json.load(open(path, encoding="utf-8"))
    assert report["format"] == "pose_point_depth_mv.omni_real_reconviagen_inference_manifest.v1"
    assert report["passed"] is True
    assert report["seeds"] == [seed]
    assert report["object_count"] == 25 and report["record_count"] == 25
    records.extend(report["objects"])
    bindings.append({"path": str(path), "sha256": sha(path)})
keys = [row["object_key"] for row in records]
assert len(records) == 200 and len(set(keys)) == 200
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
    "runtime_input_manifest": {"path": str(runtime), "sha256": sha(runtime)},
    "benchmark_manifest": {"path": str(benchmark), "sha256": sha(benchmark)},
    "worker_manifests": bindings,
    "objects": records,
    "metric_or_target_consumed_during_inference": False,
    "output_frame": "reference-view canonical O diagnostic; transform_pose=False",
}
target = out / "inference_aggregate_v1" / "report.json"
target.parent.mkdir(parents=True, exist_ok=True)
temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
os.replace(temporary, target)
print(json.dumps({"passed": True, "objects": 200, "report": str(target)}, indent=2))
PY

echo "OMNI200 STRICT RECONVIAGEN INFERENCE COMPLETE: ${OUT}/inference_aggregate_v1/report.json"
