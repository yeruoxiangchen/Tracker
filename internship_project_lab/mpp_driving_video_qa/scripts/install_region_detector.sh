#!/usr/bin/env bash
# Keep the detector's newer Transformers/Python outside the Python-3.8 MPP env.
set -euo pipefail
CONDA_BIN="${REGION_CONDA_BIN:-/home/zjr/anaconda3/bin/conda}"
DETECTOR_PYTHON="${DETECTOR_PYTHON:-/home/zjr/anaconda3/envs/mpp_regions/bin/python}"
if [[ ! -x "$DETECTOR_PYTHON" ]]; then
  "$CONDA_BIN" create -n mpp_regions python=3.11 pip -y
fi
"$DETECTOR_PYTHON" -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu118
"$DETECTOR_PYTHON" -m pip install transformers==4.57.6 pillow==11.3.0 numpy==1.26.4 safetensors
"$DETECTOR_PYTHON" -c 'from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection; import torch; print({"torch": torch.__version__, "cuda": torch.cuda.is_available(), "detector_import": "ok"})'
