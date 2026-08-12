#!/usr/bin/env python3
"""Real-time YOLO dataset capture app for Raspberry Pi.

Features:
- live camera preview
- YOLO-based object detection
- square 1:1 crop around each detected object
- ensure detected object remains inside crop
- save final images at 224x244 pixels
- persistent sequence numbering by prefix
- save directory selection

Keyboard controls:
  q         quit
  space     capture currently selected object
  n         select next detected object
  p         change prefix (console prompt)
  d         choose output directory
  l         toggle lock on current crop (visual only)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import List, Tuple

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
except Exception:  # pragma: no cover
    Picamera2 = None

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None

STATE_FILE = Path.home() / ".dataset_capture_state.json"
DEFAULT_PREFIX = "pill"
OUTPUT_WIDTH = 224
OUTPUT_HEIGHT = 244


def ensure_model_ready() -> bool:
    """Install ultralytics if it is missing so YOLO can load."""
    global YOLO
    if YOLO is not None:
        return True

    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "ultralytics"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        from ultralytics import YOLO as _YOLO

        YOLO = _YOLO
        return True
    except Exception:
        print("[WARN] ultralytics is not installed and could not be installed automatically.")
        print("[WARN] Please run: python3 -m pip install ultralytics")
        return False


def load_last_sequence() -> int:
    if not STATE_FILE.exists():
        return 0
    try:
        with STATE_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return int(data.get("last_sequence", 0))
    except Exception:
        return 0


def save_last_sequence(value: int) -> None:
    STATE_FILE.write_text(json.dumps({"last_sequence": int(value)}), encoding="utf-8")


def open_directory_dialog() -> str:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    directory = filedialog.askdirectory(title="Choose dataset folder")
    root.destroy()
    return directory or str(Path.home() / "dataset")


def ensure_save_dir(path: str) -> Path:
    save_dir = Path(path)
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def open_camera():
    try:
        if Picamera2 is None:
            raise RuntimeError("Picamera2 unavailable")
        cam = Picamera2()
        config = cam.create_preview_configuration(main={"size": (640, 480)})
        cam.configure(config)
        cam.start()
        return cam, True
    except Exception:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Failed to open camera device.")
        return cap, False


def detect_objects(frame: np.ndarray):
    if YOLO is None:
        return []

    results = YOLO(frame, conf=0.45, verbose=False)
    detections = []
    for result in results:
        for box in getattr(result, "boxes", []):
            if getattr(box, "xyxy", None) is None:
                continue
            coords = box.xyxy[0].cpu().tolist()
            x1, y1, x2, y2 = [int(v) for v in coords]
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({"bbox": (x1, y1, x2, y2)})
    return detections


def square_box_from_detection(frame_shape: Tuple[int, int, int], bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    h, w, _ = frame_shape
    x1, y1, x2, y2 = bbox
    object_w = max(1, x2 - x1)
    object_h = max(1, y2 - y1)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    side = max(object_w, object_h)
    half = side / 2.0

    sq_x1 = int(round(center_x - half))
    sq_y1 = int(round(center_y - half))
    sq_x2 = sq_x1 + int(round(side))
    sq_y2 = sq_y1 + int(round(side))

    # ensure the entire object remains inside the crop if possible
    sq_x1 = max(0, min(sq_x1, w - int(round(side))))
    sq_y1 = max(0, min(sq_y1, h - int(round(side))))
    sq_x2 = min(w, sq_x1 + int(round(side)))
    sq_y2 = min(h, sq_y1 + int(round(side)))

    # maintain a valid square region
    side_final = max(1, min(sq_x2 - sq_x1, sq_y2 - sq_y1))
    sq_x2 = sq_x1 + side_final
    sq_y2 = sq_y1 + side_final
    return sq_x1, sq_y1, sq_x2, sq_y2


def ensure_square_crop(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = square_box_from_detection(frame.shape, bbox)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
    return crop


def maybe_set_prefix() -> str:
    prefix = input("Enter image prefix (default pill): ").strip()
    return prefix if prefix else DEFAULT_PREFIX


def main() -> None:
    if ensure_model_ready() is False:
        print("YOLO model is unavailable. Exiting.")
        return

    prefix = maybe_set_prefix()
    output_dir = ensure_save_dir(open_directory_dialog())
    last_sequence = load_last_sequence()
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Current sequence number: {last_sequence}")

    try:
        camera, is_picam = open_camera()
    except Exception as exc:
        print(f"[ERROR] Could not open camera: {exc}")
        return

    print("[INFO] Camera started. Controls: q=quit, space=capture, n=next object, p=prefix, d=change folder")

    selected_index = 0
    lock_box = False
    current_prefix = prefix

    try:
        while True:
            if is_picam:
                frame = camera.capture_array()
                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                elif frame.ndim == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                ok, frame = camera.read()
                if not ok:
                    break

            frame_vis = frame.copy()
            detections = detect_objects(frame)

            if detections:
                selected_box = detections[selected_index % len(detections)]["bbox"]
                sq = square_box_from_detection(frame_vis.shape, selected_box)
                x1, y1, x2, y2 = sq
                cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_vis, f"Object {selected_index + 1}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                for idx, det in enumerate(detections):
                    box = det["bbox"]
                    x1d, y1d, x2d, y2d = box
                    color = (0, 200, 255) if idx == selected_index % len(detections) else (255, 255, 255)
                    cv2.rectangle(frame_vis, (x1d, y1d), (x2d, y2d), color, 1)
                    cv2.putText(frame_vis, str(idx + 1), (x1d, max(0, y1d - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                cv2.putText(frame_vis, "No object detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.putText(frame_vis, f"Prefix: {current_prefix}", (20, frame_vis.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_vis, f"Output: {output_dir}", (20, frame_vis.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if lock_box:
                cv2.putText(frame_vis, "LOCKED", (frame_vis.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Dataset Capture - YOLO Square Crop", frame_vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('n'):
                if detections:
                    selected_index = (selected_index + 1) % len(detections)
                else:
                    selected_index = 0
            elif key == ord('p'):
                current_prefix = maybe_set_prefix()
            elif key == ord('d'):
                output_dir = ensure_save_dir(open_directory_dialog())
            elif key == ord('l'):
                lock_box = not lock_box
            elif key == 32:
                if not detections:
                    print("[WARN] No object detected, cannot capture.")
                    continue

                selected_box = detections[selected_index % len(detections)]["bbox"]
                square_crop = ensure_square_crop(frame, selected_box)
                if square_crop.size == 0:
                    print("[WARN] Crop area was empty.")
                    continue

                resized = cv2.resize(square_crop, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
                last_sequence += 1
                save_last_sequence(last_sequence)
                filename = f"{current_prefix}_{last_sequence:04d}.jpg"
                save_path = output_dir / filename
                ok = cv2.imwrite(str(save_path), resized)
                if ok:
                    print(f"[INFO] Saved {save_path}")
                else:
                    print(f"[ERROR] Could not save {save_path}")

    finally:
        if is_picam:
            camera.stop()
        else:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Program interrupted by user.")
    finally:
        print("[INFO] Exit dataset capture app.")
