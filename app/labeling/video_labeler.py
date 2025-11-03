from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2

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


def detect_initial(frame) -> Dict[str, HoopLabel]:
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


def draw(frame, labels: Dict[str, HoopLabel], current: str, frame_idx: int, ts: float, total_frames: int, step: int) -> None:
    canvas = frame.copy()
    h, w = frame.shape[:2]
    for name, lab in labels.items():
        color = (0, 255, 0) if name == current else (0, 128, 255)
        if lab.enabled:
            cv2.circle(canvas, (int(lab.x), int(lab.y)), max(5, int(lab.radius)), color, 2)
            status = "ON"
        else:
            cv2.circle(canvas, (int(lab.x), int(lab.y)), max(5, int(lab.radius)), (128, 128, 128), 1)
            status = "OFF"
        cv2.putText(
            canvas,
            f"{name}: x={lab.x:.1f} y={lab.y:.1f} r={lab.radius:.1f} [{status}]",
            (int(lab.x - lab.radius), max(int(lab.y - lab.radius - 10), 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color if lab.enabled else (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
    # timeline bar
    cv2.rectangle(canvas, (0, h - 22), (w, h), (25, 25, 25), -1)
    progress = frame_idx / max(total_frames - 1, 1)
    cv2.rectangle(canvas, (0, h - 22), (int(w * progress), h), (0, 180, 255), -1)
    info = (
        f"frame {frame_idx}/{total_frames-1} | ts={ts:.2f}s | Mouse click=center | TAB switch | X toggle | +/- radius | "
        f"L/R or ./, step +/-{step} | SPACE save | G goto frame | T goto time(s) | Q quit"
    )
    cv2.putText(canvas, info, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("Hoop Video Labeler", canvas)


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


def save_label(video: str, frame_idx: int, ts: float, width: int, height: int, labels: Dict[str, HoopLabel]) -> None:
    folder = LABELS_ROOT / Path(video).stem
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "video": video,
        "frame": frame_idx,
        "ts": ts,
        "width": width,
        "height": height,
        "labels": {name: lab.to_dict() for name, lab in labels.items() if lab.enabled},
    }
    (folder / f"frame_{frame_idx:06d}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive hoop labeler with video playback")
    parser.add_argument("--video", required=True, help="Video file name or path")
    parser.add_argument("--step", type=int, default=30, help="Frame step for next/prev navigation")
    parser.add_argument("--scale", type=float, default=1.0, help="Optional display resize factor")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        video_path = BASE / "input_videos" / video_path.name
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Video açılamadı: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frame_idx = 0
    current = "left"
    cached_labels: Dict[int, Dict[str, HoopLabel]] = {}

    window = "Hoop Video Labeler"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            labels = cached_labels[param]
            labels[current].x = float(x)
            labels[current].y = float(y)
            labels[current].enabled = True

    while 0 <= frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        if args.scale != 1.0:
            frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]

        if frame_idx not in cached_labels:
            existing = load_existing(video_path.stem, frame_idx)
            cached_labels[frame_idx] = existing or detect_initial(frame)

        cv2.setMouseCallback(window, on_mouse, frame_idx)
        ts = frame_idx / fps

        while True:
            draw(frame, cached_labels[frame_idx], current, frame_idx, ts, total_frames, args.step)
            key = cv2.waitKey(0) & 0xFFFF

            if key in (ord("q"), 27):
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key in (ord(" "), 13):
                save_label(video_path.name, frame_idx, ts, w, h, cached_labels[frame_idx])
                frame_idx = min(total_frames - 1, frame_idx + args.step)
                break
            elif key in (ord("l"), ord("."), 83):
                frame_idx = min(total_frames - 1, frame_idx + args.step)
                break
            elif key in (ord("h"), ord(","), 81):
                frame_idx = max(0, frame_idx - args.step)
                break
            elif key == 9:  # TAB
                current = "right" if current == "left" else "left"
            elif key in (ord("+"), ord("=")):
                cached_labels[frame_idx][current].radius += 2
            elif key in (ord("-"), ord("_")):
                cached_labels[frame_idx][current].radius = max(5, cached_labels[frame_idx][current].radius - 2)
            elif key in (ord("w"), 82):
                cached_labels[frame_idx][current].y -= 2
            elif key in (ord("s"), 84):
                cached_labels[frame_idx][current].y += 2
            elif key in (ord("a"), 81):
                cached_labels[frame_idx][current].x -= 2
            elif key in (ord("d"), 83):
                cached_labels[frame_idx][current].x += 2
            elif key in (ord("x"), ord("X")):
                lab = cached_labels[frame_idx][current]
                lab.enabled = not lab.enabled
            elif key in (ord("g"), ord("G")):
                try:
                    target = int(input("Goto frame index: "))
                    frame_idx = max(0, min(total_frames - 1, target))
                    break
                except ValueError:
                    print("[WARN] invalid frame index")
            elif key in (ord("t"), ord("T")):
                try:
                    target_ts = float(input("Goto time (sec): "))
                    frame_idx = max(0, min(total_frames - 1, int(target_ts * fps)))
                    break
                except ValueError:
                    print("[WARN] invalid time value")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
