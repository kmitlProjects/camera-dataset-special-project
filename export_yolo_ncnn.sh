#!/usr/bin/env bash
set -e

# Export YOLO model to NCNN format.
# Run this on a PC or a machine with better GPU support.

if ! command -v yolo >/dev/null 2>&1; then
  echo "Ultralytics CLI 'yolo' is not available."
  echo "Install dependencies first:"
  echo "  python3 -m venv venv"
  echo "  source venv/bin/activate"
  echo "  pip install ultralytics"
  exit 1
fi

MODEL_PATH="${1:-yolov8n.pt}"
OUTPUT_DIR="${2:-./models}"
mkdir -p "$OUTPUT_DIR"

# Export to NCNN
# This may generate yolov8n.param and yolov8n.bin in the output directory.
yolo export model="$MODEL_PATH" format=ncnn imgsz=640 project="$OUTPUT_DIR"

echo "Export complete. Files are in: $OUTPUT_DIR"
