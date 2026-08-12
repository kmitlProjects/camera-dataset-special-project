#!/usr/bin/env bash
set -euo pipefail

# Remove heavy ML packages installed during ultralytics setup from current venv.
# Usage:
#   source .venv/bin/activate
#   bash scripts/cleanup_ultralytics_venv.sh

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[ERROR] Activate your virtualenv first (source .venv/bin/activate)"
  exit 1
fi

echo "[INFO] Using venv: $VIRTUAL_ENV"
python -m pip uninstall -y \
  ultralytics ultralytics-thop torch torchvision triton \
  nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime \
  nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand nvidia-cusolver \
  nvidia-cusparse nvidia-cusparselt-cu13 nvidia-ml-py nvidia-nccl-cu13 \
  nvidia-nvjitlink nvidia-nvshmem-cu13 nvidia-nvtx cuda-bindings cuda-pathfinder cuda-toolkit \
  polars polars-runtime-32 filelock fsspec networkx sympy mpmath || true

echo "[INFO] Reinstall minimal packages for NCNN capture path"
python -m pip install --upgrade pip
python -m pip install -r scripts/requirements-pi.txt

echo "[DONE] Cleanup complete"
python -m pip list | grep -E "ncnn|opencv|numpy|picamera2|ultralytics|torch" || true
