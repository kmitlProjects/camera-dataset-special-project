#!/usr/bin/env bash
set -e

MODEL="${1:-yolov8n.pt}"
IMG_SIZE="${2:-640}"
OUT_DIR="${3:-./exported_models}"

mkdir -p "$OUT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found"
  exit 1
fi

if [ ! -d .venv-yolo-export ]; then
  python3 -m venv .venv-yolo-export
fi

source .venv-yolo-export/bin/activate
python -m pip install --upgrade pip >/dev/null 2>&1 || true
python -m pip install ultralytics >/dev/null 2>&1 || true

echo "Exporting ${MODEL} to NCNN..."
yolo export model="$MODEL" format=ncnn imgsz="$IMG_SIZE" project="$OUT_DIR"

echo "Done. Exported model directory: $OUT_DIR"
