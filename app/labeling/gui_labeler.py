from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import messagebox

from app.calibration.hoop_locator import _detect_circles

BASE = Path(__file__).resolve().parents[2]
LABELS_ROOT = BASE / "data" / "hoops" / "labels_v1"


@dataclass
class HoopLabel:
    x: float
    y: float
    radius: float
    confidence: float = 0.0
    enabled: bool = True

    def to_dict(self) -> Dict[str, float | bool]:
        return {
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "radius": round(self.radius, 2),
            "confidence": round(self.confidence, 3),
            "enabled": self.enabled,
        }


def detect_initial(frame: np.ndarray) -> Dict[str, HoopLabel]:
    h, w = frame.shape[:2]
    circles = _detect_circles(frame)
    default_radius = min(w, h) * 0.07
    labels = {
        "left": HoopLabel(w * 0.25, h * 0.5, default_radius, 0.1, False),
        "right": HoopLabel(w * 0.75, h * 0.5, default_radius, 0.1, False),
    }
    if not circles:
        return labels
    for x, y, r, fill in circles:
        side = "left" if x <= w / 2 else "right"
        labels[side] = HoopLabel(x, y, r, float(fill), True)
    return labels


def load_existing(video_stem: str, frame_idx: int) -> Optional[Dict[str, HoopLabel]]:
    path = LABELS_ROOT / video_stem / f"frame_{frame_idx:06d}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels: Dict[str, HoopLabel] = {}
    for name, vals in payload.get("labels", {}).items():
        labels[name] = HoopLabel(
            x=float(vals.get("x", 0.0)),
            y=float(vals.get("y", 0.0)),
            radius=float(vals.get("radius", 0.0)),
            confidence=float(vals.get("confidence", 0.0)),
            enabled=bool(vals.get("enabled", True)),
        )
    return labels


def save_label(video: Path, frame_idx: int, ts: float, width: int, height: int, labels: Dict[str, HoopLabel]) -> None:
    folder = LABELS_ROOT / video.stem
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "video": video.name,
        "frame": frame_idx,
        "ts": ts,
        "width": width,
        "height": height,
        "labels": {name: lab.to_dict() for name, lab in labels.items() if lab.enabled},
    }
    (folder / f"frame_{frame_idx:06d}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def cv_to_photo(frame: np.ndarray) -> ImageTk.PhotoImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    return ImageTk.PhotoImage(pil_img)


def draw_overlay(frame: np.ndarray, labels: Dict[str, HoopLabel], active: str, frame_idx: int, total_frames: int, ts: float, step: int) -> np.ndarray:
    canvas = frame.copy()
    h, w = frame.shape[:2]
    for name, lab in labels.items():
        color = (0, 255, 0) if name == active else (0, 128, 255)
        radius = max(5, int(lab.radius))
        if lab.enabled:
            cv2.circle(canvas, (int(lab.x), int(lab.y)), radius, color, 2)
        else:
            cv2.circle(canvas, (int(lab.x), int(lab.y)), radius, (150, 150, 150), 1)
        status = "ON" if lab.enabled else "OFF"
        cv2.putText(
            canvas,
            f"{name}: x={lab.x:.1f} y={lab.y:.1f} r={lab.radius:.1f} [{status}]",
            (int(lab.x - lab.radius), max(int(lab.y - lab.radius - 10), 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.rectangle(canvas, (0, h - 22), (w, h), (25, 25, 25), -1)
    progress = frame_idx / max(total_frames - 1, 1)
    cv2.rectangle(canvas, (0, h - 22), (int(w * progress), h), (0, 180, 255), -1)
    cv2.putText(
        canvas,
        f"frame {frame_idx}/{total_frames-1} ts={ts:.2f}s step={step}",
        (20, h - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


class HoopLabelerApp(ctk.CTk):
    def __init__(self, video_path: Path, scale: float, step: int) -> None:
        super().__init__()
        self.title("Hoop Labeler")
        self.geometry("1280x760")
        ctk.set_default_color_theme("dark-blue")

        self.video_path = video_path
        self.scale = scale
        self.step = step

        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Video could not be opened: {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.frame_idx = 0
        self.active = "left"
        self.cached_labels: Dict[int, Dict[str, HoopLabel]] = {}

        self.canvas = ctk.CTkCanvas(self, bg="#101010")
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_click)

        controls = ctk.CTkFrame(self)
        controls.pack(fill="x", padx=10, pady=(0, 10))

        btn_prev = ctk.CTkButton(controls, text="? Prev", command=self.go_prev)
        btn_prev.pack(side="left", padx=4)
        btn_next = ctk.CTkButton(controls, text="Next ?", command=self.go_next)
        btn_next.pack(side="left", padx=4)
        ctk.CTkButton(controls, text="Switch", command=self.toggle_active).pack(side="left", padx=4)
        ctk.CTkButton(controls, text="Toggle", command=self.toggle_enabled).pack(side="left", padx=4)
        ctk.CTkButton(controls, text="Radius+", command=lambda: self.adjust_radius(2)).pack(side="left", padx=4)
        ctk.CTkButton(controls, text="Radius-", command=lambda: self.adjust_radius(-2)).pack(side="left", padx=4)
        ctk.CTkButton(controls, text="Save", command=self.save_current).pack(side="left", padx=4)
        ctk.CTkButton(controls, text="Save+Next", command=self.save_and_next).pack(side="left", padx=4)
        ctk.CTkButton(controls, text="Fill Samples", command=self.fill_samples).pack(side="left", padx=4)

        self.frame_entry = ctk.CTkEntry(controls, width=80, placeholder_text="Frame")
        self.frame_entry.pack(side="left", padx=4)
        ctk.CTkButton(controls, text="Go", command=self.goto_frame).pack(side="left", padx=4)

        self.time_entry = ctk.CTkEntry(controls, width=80, placeholder_text="Sec")
        self.time_entry.pack(side="left", padx=4)
        ctk.CTkButton(controls, text="GoTime", command=self.goto_time).pack(side="left", padx=4)

        ctk.CTkButton(controls, text="Quit", command=self.destroy).pack(side="right", padx=4)

        self.slider = ctk.CTkSlider(self, from_=0, to=max(self.total_frames - 1, 1), command=self.on_slider)
        self.slider.pack(fill="x", padx=10, pady=(0, 10))

        self.frame, self.labels, self.ts = self.load_frame(self.frame_idx)
        self.refresh_canvas()

    def load_frame(self, index: int) -> tuple[np.ndarray, Dict[str, HoopLabel], float]:
        index = max(0, min(self.total_frames - 1, index))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Failed to read frame")
        if self.scale != 1.0:
            frame = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        if index not in self.cached_labels:
            existing = load_existing(self.video_path.stem, index)
            self.cached_labels[index] = existing or detect_initial(frame)
        return frame, self.cached_labels[index], index / self.fps

    def refresh_canvas(self) -> None:
        overlay = draw_overlay(self.frame, self.labels, self.active, self.frame_idx, self.total_frames, self.ts, self.step)
        self.photo = cv_to_photo(overlay)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.slider.set(self.frame_idx)

    def on_click(self, event) -> None:
        self.labels[self.active].x = float(event.x)
        self.labels[self.active].y = float(event.y)
        self.labels[self.active].enabled = True
        self.refresh_canvas()

    def go_next(self) -> None:
        self.frame_idx = min(self.total_frames - 1, self.frame_idx + self.step)
        self.frame, self.labels, self.ts = self.load_frame(self.frame_idx)
        self.refresh_canvas()

    def go_prev(self) -> None:
        self.frame_idx = max(0, self.frame_idx - self.step)
        self.frame, self.labels, self.ts = self.load_frame(self.frame_idx)
        self.refresh_canvas()

    def toggle_active(self) -> None:
        self.active = "right" if self.active == "left" else "left"
        self.refresh_canvas()

    def toggle_enabled(self) -> None:
        self.labels[self.active].enabled = not self.labels[self.active].enabled
        self.refresh_canvas()

    def adjust_radius(self, delta: float) -> None:
        self.labels[self.active].radius = max(5, self.labels[self.active].radius + delta)
        self.refresh_canvas()

    def fill_samples(self) -> None:
        meta_dir = BASE / 'configs' / 'calibrations' / f'{self.video_path.stem}_samples' / 'meta'
        if not meta_dir.exists():
            messagebox.showwarning('Fill Samples', f'Meta directory not found: {meta_dir}')
            return
        saved = 0
        for meta_path in sorted(meta_dir.glob('frame_*.json')):
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            frame = int(meta.get('frame', 0))
            ts = float(meta.get('ts', 0.0))
            width = int(meta.get('width', self.frame.shape[1]))
            height = int(meta.get('height', self.frame.shape[0]))
            labels_copy = {name: HoopLabel(lab.x, lab.y, lab.radius, lab.confidence, lab.enabled) for name, lab in self.labels.items()}
            save_label(self.video_path, frame, ts, width, height, labels_copy)
            self.cached_labels[frame] = labels_copy
            saved += 1
        messagebox.showinfo('Fill Samples', f'Copied labels to {saved} frames.')

    def save_current(self) -> None:
        self.labels = self.cached_labels[self.frame_idx]
        save_label(self.video_path, self.frame_idx, self.ts, self.frame.shape[1], self.frame.shape[0], self.labels)

    def save_and_next(self) -> None:
        self.save_current()
        self.go_next()

    def goto_frame(self) -> None:
        try:
            target = int(self.frame_entry.get())
        except ValueError:
            return
        self.frame_idx = max(0, min(self.total_frames - 1, target))
        self.frame, self.labels, self.ts = self.load_frame(self.frame_idx)
        self.refresh_canvas()

    def goto_time(self) -> None:
        try:
            target_sec = float(self.time_entry.get())
        except ValueError:
            return
        self.frame_idx = max(0, min(self.total_frames - 1, int(target_sec * self.fps)))
        self.frame, self.labels, self.ts = self.load_frame(self.frame_idx)
        self.refresh_canvas()

    def on_slider(self, value: float) -> None:
        idx = int(value)
        if idx != self.frame_idx:
            self.frame_idx = idx
            self.frame, self.labels, self.ts = self.load_frame(self.frame_idx)
            self.refresh_canvas()


def main() -> None:
    parser = argparse.ArgumentParser(description="Hoop labeling GUI")
    parser.add_argument("--video", required=True, help="Video file name or path")
    parser.add_argument("--scale", type=float, default=0.7)
    parser.add_argument("--step", type=int, default=30)
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        video_path = BASE / "input_videos" / video_path.name
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    app = HoopLabelerApp(video_path, args.scale, args.step)
    app.mainloop()


if __name__ == "__main__":
    main()
