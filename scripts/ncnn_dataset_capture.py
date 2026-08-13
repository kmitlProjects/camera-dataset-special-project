#!/usr/bin/env python3
"""Dataset capture app for Raspberry Pi using NCNN runtime directly.

Usage:
    python3 ncnn_dataset_capture.py --model ./exported_models/yolov8n_ncnn_model --output-dir ./storage --prefix image

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
import shutil
import tkinter as tk
import uuid
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, simpledialog
from typing import List, Tuple

import cv2
import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

try:
    import ncnn
except Exception:  # pragma: no cover
    ncnn = None

try:
    from picamera2 import Picamera2
except Exception:  # pragma: no cover
    Picamera2 = None

STATE_FILE = Path.home() / ".dataset_capture_state.json"
DEFAULT_PREFIX = "image"
DEFAULT_FOLDER = "default"
OUTPUT_WIDTH = 224
OUTPUT_HEIGHT = 244
INPUT_SIZE = 640
PREVIEW_WINDOW = "Dataset Capture"
PREVIEW_HEADER_HEIGHT = 82
MANUAL_CROP_PRESETS = {
    "1:1  |  224x224 px": (1, 1, 224, 224),
    "4:3  |  224x168 px": (4, 3, 224, 168),
    "3:4  |  168x224 px": (3, 4, 168, 224),
    "Custom  |  224x244 px": (224, 244, 224, 244),
}


def log_action(message: str) -> None:
    """Print a timestamped log for every user-visible action."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [ACTION] {message}", flush=True)


def fit_preview_to_window(image: np.ndarray) -> np.ndarray:
    """Scale to the current window while filling unused space with black."""
    try:
        _, _, window_width, window_height = cv2.getWindowImageRect(PREVIEW_WINDOW)
    except cv2.error:
        return image
    if window_width <= 1 or window_height <= 1:
        return image

    image_height, image_width = image.shape[:2]
    scale = min(window_width / image_width, window_height / image_height)
    scaled_width = max(1, int(round(image_width * scale)))
    scaled_height = max(1, int(round(image_height * scale)))
    resized = cv2.resize(image, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((window_height, window_width, 3), dtype=np.uint8)
    offset_x = (window_width - scaled_width) // 2
    offset_y = (window_height - scaled_height) // 2
    canvas[offset_y:offset_y + scaled_height, offset_x:offset_x + scaled_width] = resized
    return canvas


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
        if not self.meta_path.exists() or yaml is None:
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


def load_last_folder() -> str:
    if not STATE_FILE.exists():
        return DEFAULT_FOLDER
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return sanitize_folder_name(str(data.get("last_folder", DEFAULT_FOLDER)))
    except Exception:
        return DEFAULT_FOLDER


def save_state(last_sequence: int, last_prefix: str, last_folder: str) -> None:
    STATE_FILE.write_text(
        json.dumps({
            "last_sequence": int(last_sequence),
            "last_prefix": sanitize_prefix(last_prefix),
            "last_folder": sanitize_folder_name(last_folder),
        }),
        encoding="utf-8",
    )


def sanitize_prefix(value: str) -> str:
    cleaned = value.strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
    return cleaned if cleaned else DEFAULT_PREFIX


def sanitize_folder_name(value: str) -> str:
    cleaned = value.strip().replace("/", "_").replace("\\", "_")
    return cleaned if cleaned not in {"", ".", ".."} else DEFAULT_FOLDER


def get_existing_sequence_numbers(output_dir: Path, prefix: str) -> List[int]:
    safe_prefix = sanitize_prefix(prefix)
    numbers: List[int] = []
    pattern = f"{safe_prefix}*.jpg"
    for path in output_dir.glob(pattern):
        stem = path.stem
        if not stem.startswith(safe_prefix):
            continue
        suffix = stem[len(safe_prefix) :]
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
    def __init__(self, initial_prefix: str, storage_root: Path, initial_folder: str) -> None:
        self.root = tk.Tk()
        self.root.title("Dataset Capture Controls")
        self.root.resizable(False, False)
        self.storage_root = storage_root
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self._capture_requested = False
        self._quit_requested = False
        self._yolo_enabled = True
        self._full_frame_enabled = False
        self.crop_preset_var = tk.StringVar(value=next(iter(MANUAL_CROP_PRESETS)))
        self._prefix = sanitize_prefix(initial_prefix)
        self._folder = sanitize_folder_name(initial_folder)
        (self.storage_root / self._folder).mkdir(parents=True, exist_ok=True)
        self.manager = None
        self.folder_list = None
        self.image_list = None
        self.preview_label = None
        self.preview_info_var = None
        self.preview_photo = None
        self._storage_snapshot = ()
        self._active_pane = "image"

        tk.Label(self.root, text="Filename prefix:").grid(row=0, column=0, padx=8, pady=(8, 4), sticky="w")
        self.prefix_var = tk.StringVar(value=self._prefix)
        tk.Entry(self.root, textvariable=self.prefix_var, width=24).grid(row=0, column=1, padx=8, pady=(8, 4), sticky="we")
        tk.Button(self.root, text="Update Name", command=self._update_prefix).grid(row=1, column=0, padx=8, pady=2, sticky="we")
        tk.Button(self.root, text="Capture", command=self._request_capture).grid(row=1, column=1, padx=8, pady=2, sticky="we")
        self.yolo_button = tk.Button(self.root, text="YOLO: ON", command=self._toggle_yolo)
        self.yolo_button.grid(row=2, column=0, padx=(8, 4), pady=2, sticky="we")
        self.full_frame_button = tk.Button(self.root, text="Full Frame: OFF", command=self._toggle_full_frame)
        self.full_frame_button.grid(row=2, column=1, padx=(4, 8), pady=2, sticky="we")
        tk.Label(self.root, text="Manual crop:").grid(row=3, column=0, padx=8, pady=2, sticky="w")
        tk.OptionMenu(
            self.root,
            self.crop_preset_var,
            *MANUAL_CROP_PRESETS.keys(),
            command=self._crop_preset_changed,
        ).grid(row=3, column=1, padx=8, pady=2, sticky="we")
        self.crop_size_label_var = tk.StringVar(value="Crop size: 78%")
        tk.Label(self.root, textvariable=self.crop_size_label_var).grid(row=4, column=0, padx=8, pady=0, sticky="w")
        self.crop_scale_var = tk.IntVar(value=78)
        tk.Scale(
            self.root,
            from_=30,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.crop_scale_var,
            command=self._crop_scale_changed,
            showvalue=False,
            resolution=1,
        ).grid(row=4, column=1, padx=8, pady=0, sticky="we")
        self.crop_x_label_var = tk.StringVar(value="Crop X: 0")
        tk.Label(self.root, textvariable=self.crop_x_label_var).grid(row=5, column=0, padx=8, pady=0, sticky="w")
        self.crop_x_var = tk.IntVar(value=0)
        tk.Scale(self.root, from_=-100, to=100, orient=tk.HORIZONTAL, variable=self.crop_x_var, showvalue=False, command=self._crop_x_changed).grid(
            row=5, column=1, padx=8, pady=0, sticky="we"
        )
        self.crop_y_label_var = tk.StringVar(value="Crop Y: 0")
        tk.Label(self.root, textvariable=self.crop_y_label_var).grid(row=6, column=0, padx=8, pady=0, sticky="w")
        self.crop_y_var = tk.IntVar(value=0)
        tk.Scale(self.root, from_=-100, to=100, orient=tk.HORIZONTAL, variable=self.crop_y_var, showvalue=False, command=self._crop_y_changed).grid(
            row=6, column=1, padx=8, pady=0, sticky="we"
        )
        tk.Button(self.root, text="Center Crop", command=self._center_manual_crop).grid(row=7, column=0, columnspan=2, padx=8, pady=2, sticky="we")
        tk.Button(self.root, text="File Manager", command=self._open_file_manager).grid(row=8, column=0, columnspan=2, padx=8, pady=2, sticky="we")
        self.folder_var = tk.StringVar(value=f"Save folder: {self._folder}")
        tk.Label(self.root, textvariable=self.folder_var).grid(row=9, column=0, columnspan=2, padx=8, pady=2, sticky="w")
        tk.Button(self.root, text="Quit", command=self._request_quit).grid(row=10, column=0, columnspan=2, padx=8, pady=(2, 6), sticky="we")
        self.root.protocol("WM_DELETE_WINDOW", self._request_quit)

    def _update_prefix(self) -> None:
        previous_prefix = self._prefix
        self._prefix = sanitize_prefix(self.prefix_var.get())
        self.prefix_var.set(self._prefix)
        save_state(load_last_sequence(), self._prefix, self._folder)
        log_action(f"Updated filename prefix: {previous_prefix} -> {self._prefix}")

    def _request_capture(self) -> None:
        self._capture_requested = True
        log_action("Capture button pressed")

    def _toggle_yolo(self) -> None:
        self._yolo_enabled = not self._yolo_enabled
        if self._yolo_enabled:
            self._full_frame_enabled = False
            self.full_frame_button.configure(text="Full Frame: OFF")
        self.yolo_button.configure(text=f"YOLO: {'ON' if self._yolo_enabled else 'OFF'}")
        log_action(f"YOLO detection: {'ON' if self._yolo_enabled else 'OFF'}")

    def yolo_enabled(self) -> bool:
        return self._yolo_enabled

    def _toggle_full_frame(self) -> None:
        self._full_frame_enabled = not self._full_frame_enabled
        if self._full_frame_enabled:
            self._yolo_enabled = False
            self.yolo_button.configure(text="YOLO: OFF")
        self.full_frame_button.configure(text=f"Full Frame: {'ON' if self._full_frame_enabled else 'OFF'}")
        log_action(f"Full frame capture: {'ON' if self._full_frame_enabled else 'OFF'}")

    def full_frame_enabled(self) -> bool:
        return self._full_frame_enabled

    def _crop_preset_changed(self, value: str) -> None:
        log_action(f"Manual crop preset: {value}")

    def manual_crop_preset(self) -> Tuple[str, Tuple[int, int, int, int]]:
        name = self.crop_preset_var.get()
        return name, MANUAL_CROP_PRESETS[name]

    def _crop_scale_changed(self, value: str) -> None:
        # Tk calls this continuously while dragging; preview updates immediately.
        current = int(float(value))
        self.crop_scale_var.set(current)
        self.crop_size_label_var.set(f"Crop size: {current}%")

    def _crop_x_changed(self, value: str) -> None:
        self.crop_x_label_var.set(f"Crop X: {int(float(value))}")

    def _crop_y_changed(self, value: str) -> None:
        self.crop_y_label_var.set(f"Crop Y: {int(float(value))}")

    def manual_crop_scale(self) -> float:
        return self.crop_scale_var.get() / 100.0

    def manual_crop_offset(self) -> Tuple[float, float]:
        return self.crop_x_var.get() / 100.0, self.crop_y_var.get() / 100.0

    def _center_manual_crop(self) -> None:
        self.crop_x_var.set(0)
        self.crop_y_var.set(0)
        self.crop_x_label_var.set("Crop X: 0")
        self.crop_y_label_var.set("Crop Y: 0")
        log_action("Manual crop position centered")

    def _request_quit(self) -> None:
        self._quit_requested = True
        log_action("Quit requested")

    def current_prefix(self) -> str:
        return self._prefix

    def current_folder(self) -> str:
        return self._folder

    def current_output_dir(self) -> Path:
        path = self.storage_root / self._folder
        path.mkdir(parents=True, exist_ok=True)
        return path

    def consume_capture_request(self) -> bool:
        requested = self._capture_requested
        self._capture_requested = False
        return requested

    def quit_requested(self) -> bool:
        return self._quit_requested

    def _open_file_manager(self) -> None:
        if self.manager is not None and self.manager.winfo_exists():
            self.manager.lift()
            return
        self.manager = tk.Toplevel(self.root)
        self.manager.title("Storage File Manager")
        self.manager.geometry("700x390")
        self.manager.resizable(False, False)
        tk.Label(self.manager, text="Folders in storage").grid(row=0, column=0, padx=8, pady=6)
        tk.Label(self.manager, text="Images").grid(row=0, column=1, padx=8, pady=6)
        self.folder_list = tk.Listbox(
            self.manager, width=20, height=13, exportselection=False, selectmode=tk.EXTENDED
        )
        self.folder_list.grid(row=1, column=0, padx=8, pady=4, sticky="ns")
        self.folder_list.bind("<<ListboxSelect>>", self._folder_selected)
        self.folder_list.bind("<Button-3>", self._show_folder_menu)
        self.image_list = tk.Listbox(
            self.manager, width=25, height=13, exportselection=False, selectmode=tk.EXTENDED
        )
        self.image_list.grid(row=1, column=1, padx=8, pady=4, sticky="ns")
        self.image_list.bind("<<ListboxSelect>>", self._image_selected)
        self.image_list.bind("<Control-a>", self._select_all_images)
        self.image_list.bind("<Command-a>", self._select_all_images)
        self.image_list.bind("<Button-3>", self._show_image_menu)
        preview_frame = tk.Frame(self.manager)
        preview_frame.grid(row=1, column=2, padx=8, pady=4, sticky="n")
        self.preview_label = tk.Label(preview_frame, text="Select an image")
        self.preview_label.pack(anchor="center")
        self.preview_info_var = tk.StringVar(value="")
        tk.Label(
            preview_frame,
            textvariable=self.preview_info_var,
            justify=tk.LEFT,
            anchor="w",
        ).pack(anchor="w", pady=(2, 0))

        folder_buttons = tk.Frame(self.manager)
        folder_buttons.grid(row=2, column=0, padx=8, pady=6)
        tk.Button(folder_buttons, text="New Folder", command=self._create_folder).pack(fill="x")
        tk.Button(folder_buttons, text="Use for Capture", command=self._select_save_folder).pack(fill="x", pady=3)
        image_buttons = tk.Frame(self.manager)
        image_buttons.grid(row=2, column=1, padx=8, pady=6)
        tk.Button(image_buttons, text="Rename Selected", command=self._rename_selected).pack(fill="x")
        tk.Button(image_buttons, text="Delete Selected", command=self._delete_selected).pack(fill="x", pady=3)

        self.folder_menu = tk.Menu(self.manager, tearoff=False)
        self.folder_menu.add_command(label="New Folder", command=self._create_folder)
        self.folder_menu.add_command(label="Rename Folder", command=self._rename_folder)
        self.folder_menu.add_command(label="Delete Selected Folder(s)", command=self._delete_folder)
        self.folder_menu.add_separator()
        self.folder_menu.add_command(label="Use for Capture", command=self._select_save_folder)
        self.folder_menu.add_command(label="Select All", command=self._select_all_folders)

        self.image_menu = tk.Menu(self.manager, tearoff=False)
        self.image_menu.add_command(label="Select All", command=self._select_all_images)
        self.image_menu.add_command(label="Rename Selected", command=self._rename_images)
        self.image_menu.add_command(label="Delete Selected Image(s)", command=self._delete_image)
        self._refresh_manager(select_folder=self._folder)
        self._storage_snapshot = self._make_storage_snapshot()
        self.manager.after(500, self._auto_refresh_manager)
        log_action("Opened storage file manager")

    def _folders(self) -> List[str]:
        return sorted(path.name for path in self.storage_root.iterdir() if path.is_dir())

    def _selected_folder(self) -> str | None:
        if self.folder_list is None or not self.folder_list.curselection():
            return None
        return str(self.folder_list.get(self.folder_list.curselection()[0]))

    def _selected_folders(self) -> List[str]:
        if self.folder_list is None:
            return []
        return [str(self.folder_list.get(index)) for index in self.folder_list.curselection()]

    def _selected_image_path(self) -> Path | None:
        folder = self._selected_folder()
        if folder is None or self.image_list is None or not self.image_list.curselection():
            return None
        return self.storage_root / folder / str(self.image_list.get(self.image_list.curselection()[0]))

    def _selected_image_paths(self) -> List[Path]:
        folder = self._selected_folder()
        if folder is None or self.image_list is None:
            return []
        return [
            self.storage_root / folder / str(self.image_list.get(index))
            for index in self.image_list.curselection()
        ]

    def _select_all_images(self, _event=None) -> str:
        if self.image_list is not None:
            self.image_list.selection_set(0, tk.END)
            count = self.image_list.size()
            log_action(f"Selected all images: {count} file(s)")
        return "break"

    def _select_all_folders(self, _event=None) -> str:
        if self.folder_list is not None:
            self.folder_list.selection_set(0, tk.END)
            log_action(f"Selected all folders: {self.folder_list.size()} folder(s)")
        return "break"

    @staticmethod
    def _select_right_clicked_item(listbox: tk.Listbox, event) -> bool:
        if listbox.size() == 0:
            return False
        index = listbox.nearest(event.y)
        bounds = listbox.bbox(index)
        if bounds is None or not (bounds[1] <= event.y <= bounds[1] + bounds[3]):
            return False
        if index not in listbox.curselection():
            listbox.selection_clear(0, tk.END)
            listbox.selection_set(index)
            listbox.activate(index)
            listbox.event_generate("<<ListboxSelect>>")
        return True

    def _show_folder_menu(self, event) -> str:
        if self.folder_list is not None and self._select_right_clicked_item(self.folder_list, event):
            self._active_pane = "folder"
            self.folder_menu.tk_popup(event.x_root, event.y_root)
        return "break"

    def _show_image_menu(self, event) -> str:
        if self.image_list is not None and self._select_right_clicked_item(self.image_list, event):
            self._active_pane = "image"
            self.image_menu.tk_popup(event.x_root, event.y_root)
        return "break"

    def _rename_selected(self) -> None:
        if self._active_pane == "folder":
            self._rename_folder()
        else:
            self._rename_images()

    def _delete_selected(self) -> None:
        if self._active_pane == "folder":
            self._delete_folder()
        else:
            self._delete_image()

    @staticmethod
    def _apply_alternating_rows(listbox: tk.Listbox) -> None:
        for index in range(listbox.size()):
            listbox.itemconfig(index, background="#ffffff" if index % 2 == 0 else "#e5e5e5")

    def _refresh_manager(self, select_folder: str | None = None) -> None:
        if self.folder_list is None:
            return
        wanted = select_folder or self._selected_folder() or self._folder
        self.folder_list.delete(0, tk.END)
        folders = self._folders()
        for folder in folders:
            self.folder_list.insert(tk.END, folder)
        self._apply_alternating_rows(self.folder_list)
        if wanted in folders:
            index = folders.index(wanted)
            self.folder_list.selection_set(index)
            self.folder_list.see(index)
        self._refresh_images()

    def _make_storage_snapshot(self) -> tuple:
        items = []
        for folder in self.storage_root.iterdir():
            if not folder.is_dir():
                continue
            items.append((folder.name, tuple(
                sorted((path.name, path.stat().st_mtime_ns, path.stat().st_size)
                       for path in folder.iterdir() if path.is_file())
            )))
        return tuple(sorted(items))

    def _auto_refresh_manager(self) -> None:
        if self.manager is None or not self.manager.winfo_exists():
            return
        snapshot = self._make_storage_snapshot()
        if snapshot != self._storage_snapshot:
            selected_folder = self._selected_folder() or self._folder
            selected_image = self._selected_image_path()
            selected_image_name = selected_image.name if selected_image else None
            self._refresh_manager(select_folder=selected_folder)
            if selected_image_name and self.image_list is not None:
                names = list(self.image_list.get(0, tk.END))
                if selected_image_name in names:
                    index = names.index(selected_image_name)
                    self.image_list.selection_set(index)
                    self._image_selected()
            self._storage_snapshot = snapshot
        self.manager.after(500, self._auto_refresh_manager)

    def _refresh_images(self) -> None:
        if self.image_list is None:
            return
        self.image_list.delete(0, tk.END)
        folder = self._selected_folder()
        if folder is not None:
            for path in sorted((self.storage_root / folder).iterdir()):
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.image_list.insert(tk.END, path.name)
        self._apply_alternating_rows(self.image_list)
        self._clear_preview()

    def _folder_selected(self, _event=None) -> None:
        self._active_pane = "folder"
        self._refresh_images()

    def _image_selected(self, _event=None) -> None:
        self._active_pane = "image"
        path = self._selected_image_path()
        if path is None or self.preview_label is None:
            return
        image = cv2.imread(str(path))
        if image is None:
            self._clear_preview("Cannot read image")
            return
        height, width = image.shape[:2]
        # Saved dataset images are shown at their native pixel size when they fit.
        scale = min(280 / max(width, 1), 280 / max(height, 1), 1.0)
        resized = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        ppm = f"P6 {rgb.shape[1]} {rgb.shape[0]} 255\n".encode("ascii") + rgb.tobytes()
        self.preview_photo = tk.PhotoImage(data=ppm, format="PPM")
        self.preview_label.configure(image=self.preview_photo, text="", width=rgb.shape[1], height=rgb.shape[0])
        file_size_kb = path.stat().st_size / 1024.0
        if self.preview_info_var is not None:
            self.preview_info_var.set(
                f"File: {path.name}\n"
                f"Folder: {path.parent.name}\n"
                f"Image: {width} x {height} px\n"
                f"Size: {file_size_kb:.1f} KB"
            )
        log_action(f"Previewed image: {path}")

    def _clear_preview(self, text: str = "Select an image") -> None:
        self.preview_photo = None
        if self.preview_label is not None:
            self.preview_label.configure(image="", text=text, width=20, height=10)
        if self.preview_info_var is not None:
            self.preview_info_var.set("")

    def _create_folder(self) -> None:
        value = simpledialog.askstring("New Folder", "Folder name:", parent=self.manager)
        if value is None:
            return
        name = sanitize_folder_name(value)
        path = self.storage_root / name
        if path.exists():
            messagebox.showerror("Folder exists", f"Folder '{name}' already exists.", parent=self.manager)
            return
        path.mkdir()
        log_action(f"Created folder: {name}")
        self._refresh_manager(select_folder=name)

    def _rename_folder(self) -> None:
        selected = self._selected_folders()
        if len(selected) != 1:
            messagebox.showinfo("Select folder", "Select exactly one folder to rename.", parent=self.manager)
            return
        old_name = selected[0]
        value = simpledialog.askstring("Rename Folder", "New folder name:", initialvalue=old_name, parent=self.manager)
        if value is None:
            return
        new_name = sanitize_folder_name(value)
        destination = self.storage_root / new_name
        if destination.exists() and new_name != old_name:
            messagebox.showerror("Folder exists", f"Folder '{new_name}' already exists.", parent=self.manager)
            return
        (self.storage_root / old_name).rename(destination)
        if self._folder == old_name:
            self._folder = new_name
            self.folder_var.set(f"Save folder: {self._folder}")
            save_state(load_last_sequence(), self._prefix, self._folder)
        log_action(f"Renamed folder: {old_name} -> {new_name}")
        self._refresh_manager(select_folder=new_name)

    def _delete_folder(self) -> None:
        folders = self._selected_folders()
        if not folders:
            messagebox.showinfo("Select folders", "Select one or more folders first.", parent=self.manager)
            return
        paths = [self.storage_root / folder for folder in folders]
        image_count = sum(
            1 for path in paths for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        folder_names = ", ".join(folders[:5]) + ("..." if len(folders) > 5 else "")
        confirmed = messagebox.askyesno(
            "Delete Selected Folders",
            f"Delete {len(folders)} selected folder(s)?\n{folder_names}\n\n"
            f"All {image_count} image(s) inside will be permanently deleted.\nThis cannot be undone.",
            icon="warning",
            parent=self.manager,
        )
        if not confirmed:
            log_action(f"Cancelled deletion of {len(folders)} folder(s)")
            return
        for path in paths:
            shutil.rmtree(path)
        log_action(f"Deleted {len(folders)} folder(s) and {image_count} image(s): {folder_names}")
        if self._folder in folders:
            remaining = self._folders()
            self._folder = remaining[0] if remaining else DEFAULT_FOLDER
            (self.storage_root / self._folder).mkdir(parents=True, exist_ok=True)
            self.folder_var.set(f"Save folder: {self._folder}")
            save_state(load_last_sequence(), self._prefix, self._folder)
            log_action(f"Selected fallback capture folder: {self._folder}")
        self._refresh_manager(select_folder=self._folder)

    def _select_save_folder(self) -> None:
        selected = self._selected_folders()
        if len(selected) != 1:
            messagebox.showinfo("Select folder", "Select exactly one folder for capture.", parent=self.manager)
            return
        folder = selected[0]
        self._folder = folder
        self.folder_var.set(f"Save folder: {self._folder}")
        save_state(load_last_sequence(), self._prefix, self._folder)
        log_action(f"Selected capture folder: {self._folder}")

    def _rename_images(self) -> None:
        sources = self._selected_image_paths()
        if not sources:
            messagebox.showinfo("Select images", "Select one or more images first.", parent=self.manager)
            return
        value = simpledialog.askstring(
            "Rename Selected Images",
            "New format/prefix:\nExamples: image  -> image1.jpg\n              image_{n} -> image_1.jpg",
            initialvalue=self._prefix,
            parent=self.manager,
        )
        if value is None:
            return
        pattern = Path(value.strip()).name
        if not pattern:
            return

        destinations = []
        for number, source in enumerate(sources, start=1):
            stem_pattern = Path(pattern).stem if Path(pattern).suffix else pattern
            stem = stem_pattern.replace("{n}", str(number)) if "{n}" in stem_pattern else f"{stem_pattern}{number}"
            suffix = Path(pattern).suffix or source.suffix
            destinations.append(source.parent / f"{stem}{suffix}")

        if len(set(destinations)) != len(destinations):
            messagebox.showerror("Duplicate names", "The format creates duplicate filenames.", parent=self.manager)
            return
        selected_set = set(sources)
        conflicts = [path.name for path in destinations if path.exists() and path not in selected_set]
        if conflicts:
            messagebox.showerror(
                "File exists",
                "Cannot rename because these files already exist:\n" + "\n".join(conflicts[:8]),
                parent=self.manager,
            )
            return

        if not messagebox.askyesno(
            "Confirm Rename",
            f"Rename {len(sources)} selected image(s) using format '{pattern}'?",
            parent=self.manager,
        ):
            log_action(f"Cancelled bulk rename of {len(sources)} image(s)")
            return

        temporary_paths = []
        for source in sources:
            temporary = source.parent / f".rename-{uuid.uuid4().hex}{source.suffix}"
            source.rename(temporary)
            temporary_paths.append(temporary)
        for temporary, destination in zip(temporary_paths, destinations):
            temporary.rename(destination)
        log_action(f"Renamed {len(sources)} image(s) using format: {pattern}")
        self._refresh_images()

    def _delete_image(self) -> None:
        paths = self._selected_image_paths()
        if not paths:
            messagebox.showinfo("Select images", "Select one or more images first.", parent=self.manager)
            return
        names = ", ".join(path.name for path in paths[:5]) + ("..." if len(paths) > 5 else "")
        if not messagebox.askyesno(
            "Delete Selected Images",
            f"Delete {len(paths)} selected image(s) permanently?\n{names}\n\nThis cannot be undone.",
            icon="warning",
            parent=self.manager,
        ):
            log_action(f"Cancelled deletion of {len(paths)} image(s)")
            return
        for path in paths:
            path.unlink()
        log_action(f"Deleted {len(paths)} selected image(s): {names}")
        self._refresh_images()

    def process_events(self) -> None:
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self._quit_requested = True

    def close(self) -> None:
        try:
            self.root.destroy()
        except tk.TclError:
            pass


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


def centered_crop_box(
    frame: np.ndarray,
    ratio_width: int,
    ratio_height: int,
    coverage: float = 0.78,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> Tuple[int, int, int, int]:
    """Return a movable crop box while preserving the requested aspect ratio."""
    frame_height, frame_width = frame.shape[:2]
    max_width = max(1, int(frame_width * coverage))
    max_height = max(1, int(frame_height * coverage))
    ratio = ratio_width / ratio_height
    crop_width = max_width
    crop_height = int(round(crop_width / ratio))
    if crop_height > max_height:
        crop_height = max_height
        crop_width = int(round(crop_height * ratio))
    center_x1 = (frame_width - crop_width) // 2
    center_y1 = (frame_height - crop_height) // 2
    x1 = center_x1 + int(round(max(-1.0, min(1.0, offset_x)) * center_x1))
    y1 = center_y1 + int(round(max(-1.0, min(1.0, offset_y)) * center_y1))
    x1 = max(0, min(x1, frame_width - crop_width))
    y1 = max(0, min(y1, frame_height - crop_height))
    return x1, y1, x1 + crop_width, y1 + crop_height


def draw_manual_crop_preview(frame: np.ndarray, box: Tuple[int, int, int, int], label: str) -> np.ndarray:
    """Dim everything outside the selected manual crop and label its boundary."""
    x1, y1, x2, y2 = box
    result = (frame.astype(np.float32) * 0.28).astype(np.uint8)
    result[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 255), 2)
    label_y = y1 + 22 if y1 < 28 else y1 - 8
    cv2.putText(result, label, (x1 + 5, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    return result


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
    filename = f"{sanitize_prefix(prefix)}{last_sequence}.jpg"
    dst = output_dir / filename
    cv2.imwrite(str(dst), frame)
    log_action(f"Saved image: {dst}")
    return last_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset capture app using YOLO NCNN model")
    parser.add_argument("--model", default="./exported_models/yolov8n_ncnn_model", help="Path to exported NCNN model folder")
    parser.add_argument("--output-dir", default="./storage", help="Storage root for captured image folders")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="Filename prefix")
    parser.add_argument("--conf", type=float, default=0.05, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--pad", type=int, default=10, help="Extra pixels added around detected object before crop")
    args = parser.parse_args()

    model_path = str(Path(args.model).expanduser())
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    storage_root = ensure_save_dir(args.output_dir)
    current_prefix = sanitize_prefix(load_last_prefix())
    current_folder = sanitize_folder_name(load_last_folder())
    detector = NcnnYoloDetector(model_path, conf_thres=args.conf, iou_thres=args.iou)
    print(f"[INFO] Model: {model_path}")
    print(f"[INFO] Storage root: {storage_root}")
    print(f"[INFO] Current prefix: {current_prefix}")
    print(f"[INFO] Current folder: {current_folder}")
    if not os.environ.get("DISPLAY"):
        print("[ERROR] DISPLAY is not set. Run this on Raspberry Pi desktop, VNC, or with X11 forwarding.")
        return

    panel = ControlPanel(current_prefix, storage_root, current_folder)

    camera, is_picam = open_camera()
    selected_index = 0
    last_saved_filename = "-"
    backend_name = "Picamera2" if is_picam else "OpenCV VideoCapture"
    print(f"[INFO] Camera backend: {backend_name}")
    print("[INFO] Controls: capture button, update name button, quit button, n=next object, space=save image")

    # WINDOW_GUI_NORMAL removes Qt's pan/zoom/save toolbar from the preview.
    cv2.namedWindow(
        PREVIEW_WINDOW,
        cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_FREERATIO,
    )
    cv2.resizeWindow(PREVIEW_WINDOW, 640, 480 + PREVIEW_HEADER_HEIGHT)
    log_action(f"Application started with filename prefix: {current_prefix}")

    try:
        while True:
            panel.process_events()
            if panel.quit_requested():
                break

            current_prefix = panel.current_prefix()
            current_folder = panel.current_folder()
            yolo_enabled = panel.yolo_enabled()
            full_frame_enabled = panel.full_frame_enabled()
            crop_preset_name, crop_preset = panel.manual_crop_preset()
            manual_crop_scale = panel.manual_crop_scale()
            manual_crop_offset_x, manual_crop_offset_y = panel.manual_crop_offset()
            ratio_width, ratio_height, manual_output_width, manual_output_height = crop_preset
            output_dir = panel.current_output_dir()
            next_sequence = next_available_sequence(output_dir, current_prefix)
            current_filename = f"{current_prefix}{next_sequence}.jpg"

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

            detections = detector.infer(frame) if yolo_enabled else []
            manual_crop_box = centered_crop_box(
                frame,
                ratio_width,
                ratio_height,
                manual_crop_scale,
                manual_crop_offset_x,
                manual_crop_offset_y,
            )
            preview_frame = frame if (yolo_enabled or full_frame_enabled) else draw_manual_crop_preview(
                    frame,
                    manual_crop_box,
                    crop_preset_name.replace("  |  ", " size: "),
                )

            # Add a separate responsive header above the camera image so no preview pixels are covered.
            frame_vis = cv2.copyMakeBorder(
                preview_frame,
                PREVIEW_HEADER_HEIGHT,
                0,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
            cv2.putText(
                frame_vis,
                f"Last: {last_saved_filename}",
                (9, 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame_vis,
                f"Prefix: {current_prefix}",
                (9, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame_vis,
                f"Folder: {current_folder}",
                (9, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.line(
                frame_vis,
                (0, PREVIEW_HEADER_HEIGHT - 1),
                (frame_vis.shape[1], PREVIEW_HEADER_HEIGHT - 1),
                (190, 190, 190),
                1,
            )

            if yolo_enabled and detections:
                selected_bbox = detections[selected_index % len(detections)][:4]
                selected_bbox = expand_bbox(frame.shape, selected_bbox, args.pad)
                x1, y1, x2, y2 = make_square_crop(frame, selected_bbox)
                draw_y1 = y1 + PREVIEW_HEADER_HEIGHT
                draw_y2 = y2 + PREVIEW_HEADER_HEIGHT
                cv2.rectangle(frame_vis, (x1, draw_y1), (x2, draw_y2), (0, 255, 0), 2)
                cv2.putText(frame_vis, f"Object {selected_index + 1}", (x1, max(PREVIEW_HEADER_HEIGHT + 18, draw_y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                for idx, bbox in enumerate(detections):
                    bx1, by1, bx2, by2, score, cls_id = bbox
                    color = (0, 200, 255) if idx == selected_index % len(detections) else (255, 255, 255)
                    draw_by1 = by1 + PREVIEW_HEADER_HEIGHT
                    draw_by2 = by2 + PREVIEW_HEADER_HEIGHT
                    cv2.rectangle(frame_vis, (bx1, draw_by1), (bx2, draw_by2), color, 1)
                    label = f"{idx + 1} {score:.2f}"
                    cv2.putText(frame_vis, label, (bx1, max(PREVIEW_HEADER_HEIGHT + 15, draw_by1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            elif yolo_enabled:
                cv2.putText(frame_vis, "No object detected", (20, PREVIEW_HEADER_HEIGHT + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            elif full_frame_enabled:
                cv2.putText(frame_vis, "FULL FRAME 640x480", (20, PREVIEW_HEADER_HEIGHT + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            else:
                cv2.putText(frame_vis, "MANUAL CROP", (20, PREVIEW_HEADER_HEIGHT + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            cv2.imshow(PREVIEW_WINDOW, fit_preview_to_window(frame_vis))
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                log_action("Quit key pressed")
                break
            elif key == ord('n') and detections:
                selected_index = (selected_index + 1) % len(detections)
                log_action(f"Selected object: {selected_index + 1} of {len(detections)}")
            elif key == 32:
                log_action(f"Spacebar pressed; attempting to capture: {current_filename}")
                if yolo_enabled and not detections:
                    print("[WARN] No object detected. Move the object into frame first.")
                    log_action(f"Capture skipped (no object detected): {current_filename}")
                    continue
                if yolo_enabled:
                    sel = detections[selected_index % len(detections)][:4]
                    sel = expand_bbox(frame.shape, sel, args.pad)
                    x1, y1, x2, y2 = make_square_crop(frame, sel)
                    capture_frame = frame[y1:y2, x1:x2]
                    if capture_frame.size == 0:
                        print("[WARN] Empty crop. Try another detection.")
                        continue
                elif full_frame_enabled:
                    capture_frame = frame
                else:
                    mx1, my1, mx2, my2 = manual_crop_box
                    capture_frame = frame[my1:my2, mx1:mx2]
                if full_frame_enabled:
                    resize_crop = capture_frame.copy()
                else:
                    output_size = (OUTPUT_WIDTH, OUTPUT_HEIGHT) if yolo_enabled else (manual_output_width, manual_output_height)
                    resize_crop = cv2.resize(capture_frame, output_size, interpolation=cv2.INTER_LINEAR)
                current_prefix = panel.current_prefix()
                next_sequence = next_available_sequence(output_dir, current_prefix)
                save_capture(resize_crop, output_dir, current_prefix, next_sequence)
                last_saved_filename = f"{current_prefix}{next_sequence}.jpg"
                save_state(next_sequence, current_prefix, current_folder)

            if panel.consume_capture_request():
                current_filename = f"{current_prefix}{next_available_sequence(output_dir, current_prefix)}.jpg"
                log_action(f"Attempting to capture: {current_filename}")
                if yolo_enabled and not detections:
                    print("[WARN] No object detected. Move the object into frame first.")
                    log_action(f"Capture skipped (no object detected): {current_filename}")
                    continue
                if yolo_enabled:
                    sel = detections[selected_index % len(detections)][:4]
                    sel = expand_bbox(frame.shape, sel, args.pad)
                    x1, y1, x2, y2 = make_square_crop(frame, sel)
                    capture_frame = frame[y1:y2, x1:x2]
                    if capture_frame.size == 0:
                        print("[WARN] Empty crop. Try another detection.")
                        continue
                elif full_frame_enabled:
                    capture_frame = frame
                else:
                    mx1, my1, mx2, my2 = manual_crop_box
                    capture_frame = frame[my1:my2, mx1:mx2]
                if full_frame_enabled:
                    resize_crop = capture_frame.copy()
                else:
                    output_size = (OUTPUT_WIDTH, OUTPUT_HEIGHT) if yolo_enabled else (manual_output_width, manual_output_height)
                    resize_crop = cv2.resize(capture_frame, output_size, interpolation=cv2.INTER_LINEAR)
                current_prefix = panel.current_prefix()
                next_sequence = next_available_sequence(output_dir, current_prefix)
                save_capture(resize_crop, output_dir, current_prefix, next_sequence)
                last_saved_filename = f"{current_prefix}{next_sequence}.jpg"
                save_state(next_sequence, current_prefix, current_folder)

    finally:
        log_action("Application stopped")
        panel.close()
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
