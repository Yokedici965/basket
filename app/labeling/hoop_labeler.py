from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import cv2

BASE = Path(__file__).resolve().parents[2]
SAMPLE_ROOT = BASE / "configs" / "calibrations"
LABEL_ROOT = BASE / "data" / "hoops" / "labels_v1"


@dataclass
class HoopLabel:
    x: float
    y: float
    radius: float
    confidence: float
    enabled: bool = True

    def to_dict(self) -> Dict[str, float | bool]:
        return {
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "radius": round(self.radius, 2),
            "confidence": round(self.confidence, 3),
            "enabled": self.enabled,
        }


def load_samples(video_stem: str) -> tuple[list[Path], list[Path]]:
    sample_dir = SAMPLE_ROOT / f"{video_stem}_samples"
    meta_dir = sample_dir / "meta"
    if not sample_dir.exists() or not meta_dir.exists():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")
    image_paths = sorted(p for p in sample_dir.glob("raw/*.jpg"))
    meta_paths = sorted(meta_dir.glob("*.json"))
    return image_paths, meta_paths


def select_initial(meta: Dict[str, object]) -> Dict[str, HoopLabel]:
    width = meta.get("width", 1920)
    height = meta.get("height", 1080)
    default_radius = min(width, height) * 0.07
    circles = meta.get("circles", [])
    labels = {
        "left": HoopLabel(width * 0.25, height * 0.5, default_radius, 0.1, False),
        "right": HoopLabel(width * 0.75, height * 0.5, default_radius, 0.1, False),
    }
    for circ in circles:
        name = "left" if circ["x"] <= width / 2 else "right"
        labels[name] = HoopLabel(circ["x"], circ["y"], circ["radius"], circ.get("fill", 0.0), True)
    return labels


def draw(frame, labels: Dict[str, HoopLabel], current: str, video: str, idx: int, total: int) -> cv2.Mat:
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
    cv2.rectangle(canvas, (0, h - 20), (w, h), (30, 30, 30), -1)
    progress = (idx + 1) / max(total, 1)
    cv2.rectangle(canvas, (0, h - 20), (int(w * progress), h), (0, 255, 0), -1)
    info = (
        f"video={video} ({idx+1}/{total}) | Mouse click=center | +/- radius | TAB switch | X toggle | SPACE save"
        " | B back | Q quit"
    )
    cv2.putText(canvas, info, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def label_video(video_stem: str) -> None:
    image_paths, meta_paths = load_samples(video_stem)
    labels_dir = LABEL_ROOT / video_stem
    labels_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    current = "left"
    cached: Dict[str, Dict[str, HoopLabel]] = {}
    window = "Hoop Sample Labeler"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            label_state = cached[param]
            label_state[current].x = float(x)
            label_state[current].y = float(y)
            label_state[current].enabled = True

    while 0 <= idx < len(image_paths):
        img_path = image_paths[idx]
        meta = json.loads(meta_paths[idx].read_text(encoding="utf-8"))
        frame = cv2.imread(str(img_path))
        key = img_path.stem
        label_state = cached.get(key) or select_initial(meta)
        cached[key] = label_state

        cv2.setMouseCallback(window, on_mouse, key)

        while True:
            display = draw(frame, label_state, current, video_stem, idx, len(image_paths))
            cv2.imshow(window, display)
            key_val = cv2.waitKey(0) & 0xFF

            if key_val in (ord("q"), 27):
                cv2.destroyAllWindows()
                return
            elif key_val in (ord(" "), ord("n")):
                payload = {
                    "video": meta.get("video"),
                    "frame": meta.get("frame"),
                    "ts": meta.get("ts"),
                    "width": meta.get("width"),
                    "height": meta.get("height"),
                    "labels": {name: lab.to_dict() for name, lab in label_state.items() if lab.enabled},
                }
                (labels_dir / f"{key}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                idx += 1
                break
            elif key_val in (ord("b"), ord("p")):
                idx = max(idx - 1, 0)
                break
            elif key_val == 9:  # TAB
                current = "right" if current == "left" else "left"
            elif key_val in (ord("+"), ord("=")):
                label_state[current].radius += 2
            elif key_val in (ord("-"), ord("_")):
                label_state[current].radius = max(5, label_state[current].radius - 2)
            elif key_val in (ord("x"), ord("X")):
                label_state[current].enabled = not label_state[current].enabled

    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive hoop labeling tool for sampled frames")
    parser.add_argument("--video", required=True, help="Video stem (e.g. mac2)")
    args = parser.parse_args()
    label_video(args.video)


if __name__ == "__main__":
    main()
