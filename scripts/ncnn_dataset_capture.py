#!/usr/bin/env python3
"""Dataset capture app for Raspberry Pi using NCNN runtime directly.

Usage:
    python3 ncnn_dataset_capture.py --model ./exported_models/yolov8n_ncnn_model --output-dir ./dataset --prefix pill

Features:
- live camera preview
- object detection from NCNN exported YOLO model
- convert detected box to square 1:1 crop that fully contains the object
- expand crop slightly so object edges are less likely to be cut off
- save final crop as 224x244 pixels
- persistent sequence counter saved to disk
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import tkinter as tk
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import yaml

try:
    import ncnn
except Exception:  # pragma: no cover
    ncnn = None

try:
    from picamera2 import Picamera2
except Exception:  # pragma: no cover
    Picamera2 = None

STATE_FILE = Path.home() / ".dataset_capture_state.json"
DEFAULT_PREFIX = "pill"
OUTPUT_WIDTH = 224
OUTPUT_HEIGHT = 244
INPUT_SIZE = 640


class NcnnYoloDetector:
    def __init__(self, model_dir: str, conf_thres: float = 0.05, iou_thres: float = 0.45) -> None:
        if ncnn is None:
            raise RuntimeError("ncnn python package is not installed. Run: python3 -m pip install ncnn")

        self.model_dir = Path(model_dir)
        self.param_path = self.model_dir / "model.ncnn.param"
        self.bin_path = self.model_dir / "model.ncnn.bin"
        self.meta_path = self.model_dir / "metadata.yaml"
        if not self.param_path.exists() or not self.bin_path.exists():
            raise FileNotFoundError(f"NCNN model files not found in: {self.model_dir}")

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.class_names = self._load_class_names()

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.load_param(str(self.param_path))
        self.net.load_model(str(self.bin_path))

    def _load_class_names(self) -> List[str]:
        if not self.meta_path.exists():
            return []
        try:
            data = yaml.safe_load(self.meta_path.read_text(encoding="utf-8"))
            raw = data.get("names", {}) if isinstance(data, dict) else {}
            if isinstance(raw, dict):
                pairs = sorted(((int(k), str(v)) for k, v in raw.items()), key=lambda x: x[0])
                return [name for _, name in pairs]
        except Exception:
            pass
        return []

    def _letterbox(self, image: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        h, w = image.shape[:2]
        scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
        pad_w = (INPUT_SIZE - new_w) / 2.0
        pad_h = (INPUT_SIZE - new_h) / 2.0
        left = int(np.floor(pad_w))
        top = int(np.floor(pad_h))
        canvas[top:top + new_h, left:left + new_w] = resized
        return canvas, scale, pad_w, pad_h

    @staticmethod
    def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
        out = np.empty_like(xywh)
        out[:, 0] = xywh[:, 0] - xywh[:, 2] / 2.0
        out[:, 1] = xywh[:, 1] - xywh[:, 3] / 2.0
        out[:, 2] = xywh[:, 0] + xywh[:, 2] / 2.0
        out[:, 3] = xywh[:, 1] + xywh[:, 3] / 2.0
        return out

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]
        keep: List[int] = []

        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter + 1e-6
            iou = inter / union

            inds = np.where(iou <= iou_thres)[0]
            order = order[inds + 1]
        return keep

    def infer(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float, int]]:
        img, scale, pad_w, pad_h = self._letterbox(frame_bgr)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mat = ncnn.Mat.from_pixels(img_rgb, ncnn.Mat.PixelType.PIXEL_RGB, INPUT_SIZE, INPUT_SIZE)
        mean_vals = []
        norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]
        mat.substract_mean_normalize(mean_vals, norm_vals)

        ex = self.net.create_extractor()
        ex.input("in0", mat)
        _, out = ex.extract("out0")

        pred = np.array(out)
        pred = np.squeeze(pred)

        if pred.ndim != 2:
            return []

        # YOLOv8 typical output: [84, N] or [N, 84]
        if pred.shape[0] <= pred.shape[1] and pred.shape[0] >= 6:
            pred = pred.T

        if pred.shape[1] < 6:
            return []

        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
        mask = scores >= self.conf_thres

        if not np.any(mask):
            return []

        boxes_xywh = boxes_xywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        boxes = self._xywh_to_xyxy(boxes_xywh)
        boxes[:, [0, 2]] -= pad_w
        boxes[:, [1, 3]] -= pad_h
        boxes /= max(scale, 1e-6)

        h, w = frame_bgr.shape[:2]
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)

        keep = self._nms(boxes, scores, self.iou_thres)
        detections: List[Tuple[int, int, int, int, float, int]] = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append((int(x1), int(y1), int(x2), int(y2), float(scores[i]), int(class_ids[i])))
        return detections


def expand_bbox(frame_shape: Tuple[int, int, int], bbox: Tuple[int, int, int, int], extra_pixels: int = 10) -> Tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    return (
        max(0, x1 - extra_pixels),
        max(0, y1 - extra_pixels),
        min(w, x2 + extra_pixels),
        min(h, y2 + extra_pixels),
    )


def load_last_sequence() -> int:
    if not STATE_FILE.exists():
        return 0
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return int(data.get("last_sequence", 0))
    except Exception:
        return 0


def load_last_prefix() -> str:
    if not STATE_FILE.exists():
        return DEFAULT_PREFIX
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        prefix = str(data.get("last_prefix", DEFAULT_PREFIX)).strip()
        return prefix if prefix else DEFAULT_PREFIX
    except Exception:
        return DEFAULT_PREFIX


def save_state(last_sequence: int, last_prefix: str) -> None:
    STATE_FILE.write_text(
        json.dumps({"last_sequence": int(last_sequence), "last_prefix": sanitize_prefix(last_prefix)}),
        encoding="utf-8",
    )


def sanitize_prefix(value: str) -> str:
    cleaned = value.strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
    return cleaned if cleaned else DEFAULT_PREFIX


def get_existing_sequence_numbers(output_dir: Path, prefix: str) -> List[int]:
    safe_prefix = sanitize_prefix(prefix)
    numbers: List[int] = []
    pattern = f"{safe_prefix}_*.jpg"
    for path in output_dir.glob(pattern):
        stem = path.stem
        if not stem.startswith(f"{safe_prefix}_"):
            continue
        suffix = stem[len(safe_prefix) + 1 :]
        if suffix.isdigit():
            numbers.append(int(suffix))
    return sorted(set(numbers))


def next_available_sequence(output_dir: Path, prefix: str) -> int:
    existing = get_existing_sequence_numbers(output_dir, prefix)
    next_number = 1
    for number in existing:
        if number == next_number:
            next_number += 1
        elif number > next_number:
            break
    return next_number


class ControlPanel:
    def __init__(self, initial_prefix: str) -> None:
        self.root = tk.Tk()
        self.root.title("Dataset Capture Controls")
        self.root.resizable(False, False)
        self.lock = threading.Lock()
        self._capture_requested = False
        self._quit_requested = False
        self._prefix = sanitize_prefix(initial_prefix)

        tk.Label(self.root, text="Filename prefix:").grid(row=0, column=0, padx=8, pady=(8, 4), sticky="w")
        self.prefix_var = tk.StringVar(value=self._prefix)
        self.prefix_entry = tk.Entry(self.root, textvariable=self.prefix_var, width=24)
        self.prefix_entry.grid(row=0, column=1, padx=8, pady=(8, 4), sticky="we")

        tk.Button(self.root, text="Update Name", command=self._update_prefix).grid(row=1, column=0, padx=8, pady=6, sticky="we")
        tk.Button(self.root, text="Capture", command=self._request_capture).grid(row=1, column=1, padx=8, pady=6, sticky="we")
        tk.Button(self.root, text="Quit", command=self._request_quit).grid(row=2, column=0, columnspan=2, padx=8, pady=(4, 8), sticky="we")

        self.root.protocol("WM_DELETE_WINDOW", self._request_quit)

    def _update_prefix(self) -> None:
        with self.lock:
            self._prefix = sanitize_prefix(self.prefix_var.get())
            self.prefix_var.set(self._prefix)

    def _request_capture(self) -> None:
        with self.lock:
            self._capture_requested = True

    def _request_quit(self) -> None:
        with self.lock:
            self._quit_requested = True

    def current_prefix(self) -> str:
        with self.lock:
            return self._prefix

    def consume_capture_request(self) -> bool:
        with self.lock:
            requested = self._capture_requested
            self._capture_requested = False
            return requested

    def quit_requested(self) -> bool:
        with self.lock:
            return self._quit_requested

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self.root.mainloop, daemon=True)
        thread.start()
        return thread


def open_camera():
    try:
        if Picamera2 is not None:
            cam = Picamera2()
            config = cam.create_preview_configuration(main={"size": (640, 480)})
            cam.configure(config)
            cam.start()
            return cam, True
    except Exception:
        pass

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera could not be opened")
    return cap, False


def ensure_save_dir(path: str) -> Path:
    dp = Path(path)
    dp.mkdir(parents=True, exist_ok=True)
    return dp


def make_square_crop(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    obj_w = max(1, x2 - x1)
    obj_h = max(1, y2 - y1)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    side = max(obj_w, obj_h)
    half = side / 2.0

    square_x1 = int(round(center_x - half))
    square_y1 = int(round(center_y - half))
    square_x2 = square_x1 + int(round(side))
    square_y2 = square_y1 + int(round(side))

    square_x1 = max(0, min(square_x1, w - 1))
    square_y1 = max(0, min(square_y1, h - 1))
    square_x2 = min(w, square_x1 + int(round(side)))
    square_y2 = min(h, square_y1 + int(round(side)))

    # ensure final crop stays valid
    side_final = max(1, min(square_x2 - square_x1, square_y2 - square_y1))
    square_x2 = square_x1 + side_final
    square_y2 = square_y1 + side_final
    return square_x1, square_y1, square_x2, square_y2


def save_capture(frame: np.ndarray, output_dir: Path, prefix: str, last_sequence: int) -> int:
    filename = f"{sanitize_prefix(prefix)}_{last_sequence}.jpg"
    dst = output_dir / filename
    cv2.imwrite(str(dst), frame)
    print(f"[INFO] Saved {dst}")
    return last_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset capture app using YOLO NCNN model")
    parser.add_argument("--model", default="./exported_models/yolov8n_ncnn_model", help="Path to exported NCNN model folder")
    parser.add_argument("--output-dir", default="./dataset", help="Directory for captured images")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="Filename prefix")
    parser.add_argument("--conf", type=float, default=0.05, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--pad", type=int, default=10, help="Extra pixels added around detected object before crop")
    args = parser.parse_args()

    model_path = str(Path(args.model).expanduser())
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    output_dir = ensure_save_dir(args.output_dir)
    current_prefix = sanitize_prefix(load_last_prefix())
    detector = NcnnYoloDetector(model_path, conf_thres=args.conf, iou_thres=args.iou)
    print(f"[INFO] Model: {model_path}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Current prefix: {current_prefix}")
    if not os.environ.get("DISPLAY"):
        print("[ERROR] DISPLAY is not set. Run this on Raspberry Pi desktop, VNC, or with X11 forwarding.")
        return

    panel = ControlPanel(current_prefix)
    panel.start()

    camera, is_picam = open_camera()
    selected_index = 0
    backend_name = "Picamera2" if is_picam else "OpenCV VideoCapture"
    print(f"[INFO] Camera backend: {backend_name}")
    print("[INFO] Controls: capture button, update name button, quit button, n=next object, space=save image")

    try:
        while True:
            if panel.quit_requested():
                break

            current_prefix = panel.current_prefix()
            next_sequence = next_available_sequence(output_dir, current_prefix)

            if is_picam:
                frame = camera.capture_array()
                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                elif frame.ndim == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                ok, frame = camera.read()
                if not ok:
                    print("[ERROR] Camera read failed. Check the camera connection or permissions.")
                    break

            frame_vis = frame.copy()
            detections = detector.infer(frame)

            if detections:
                selected_bbox = detections[selected_index % len(detections)][:4]
                selected_bbox = expand_bbox(frame_vis.shape, selected_bbox, args.pad)
                x1, y1, x2, y2 = make_square_crop(frame, selected_bbox)
                cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_vis, f"Object {selected_index + 1}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                for idx, bbox in enumerate(detections):
                    bx1, by1, bx2, by2, score, cls_id = bbox
                    color = (0, 200, 255) if idx == selected_index % len(detections) else (255, 255, 255)
                    cv2.rectangle(frame_vis, (bx1, by1), (bx2, by2), color, 1)
                    label = f"{idx + 1} {score:.2f}"
                    cv2.putText(frame_vis, label, (bx1, max(0, by1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            else:
                cv2.putText(frame_vis, "No object detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Dataset Capture", frame_vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('n') and detections:
                selected_index = (selected_index + 1) % len(detections)
            elif key == 32:
                if not detections:
                    print("[WARN] No object detected. Move the object into frame first.")
                    continue
                sel = detections[selected_index % len(detections)][:4]
                sel = expand_bbox(frame.shape, sel, args.pad)
                x1, y1, x2, y2 = make_square_crop(frame, sel)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    print("[WARN] Empty crop. Try another detection.")
                    continue
                resize_crop = cv2.resize(crop, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
                current_prefix = panel.current_prefix()
                next_sequence = next_available_sequence(output_dir, current_prefix)
                save_capture(resize_crop, output_dir, current_prefix, next_sequence)
                save_state(next_sequence, current_prefix)

            if panel.consume_capture_request():
                if not detections:
                    print("[WARN] No object detected. Move the object into frame first.")
                    continue
                sel = detections[selected_index % len(detections)][:4]
                sel = expand_bbox(frame.shape, sel, args.pad)
                x1, y1, x2, y2 = make_square_crop(frame, sel)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    print("[WARN] Empty crop. Try another detection.")
                    continue
                resize_crop = cv2.resize(crop, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
                current_prefix = panel.current_prefix()
                next_sequence = next_available_sequence(output_dir, current_prefix)
                save_capture(resize_crop, output_dir, current_prefix, next_sequence)
                save_state(next_sequence, current_prefix)

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
        print("\n[INFO] Interrupted by user")
    except Exception as exc:
        print(f"[ERROR] {exc}")
        raise
