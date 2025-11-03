from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

BASE = Path(__file__).resolve().parents[1]
VIDEO_DIR = BASE / "input_videos"
OUTPUT_DIR = BASE / "outputs"
RENDER_DIR = BASE / "renders"
CFG_PATH = BASE / "configs" / "salon.yaml"

COLOR_MAP: Dict[int, Tuple[int, int, int]] = {
    0: (68, 189, 50),    # player -> green
    32: (235, 87, 87),   # ball -> red
}
LABEL_PREFIX = {0: "P", 32: "B"}
OUTLINE_COLOR = (245, 245, 245)


def resolve_csv(stem: str, source: str) -> Path:
    """Resolve CSV path for given video stem and source (tracks or detections)."""
    candidates = [
        OUTPUT_DIR / f"{stem}_{source}.csv",
        OUTPUT_DIR / f"{stem.lower()}_{source}.csv",
        OUTPUT_DIR / f"{stem.upper()}_{source}.csv",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"CSV not found for {stem} ({source}) in outputs/")


def load_court_shapes(cfg_path: Path) -> Tuple[List[Tuple[int, int]], List[Tuple[str, float, float]]]:
    polygon: List[Tuple[int, int]] = []
    hoops: List[Tuple[str, float, float]] = []
    if not cfg_path.exists():
        return polygon, hoops
    try:
        with cfg_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception:
        return polygon, hoops
    court = cfg.get("court", {}) or {}
    polygon_data = court.get("polygon") or []
    for point in polygon_data:
        try:
            x, y = point
            polygon.append((int(round(x)), int(round(y))))
        except (TypeError, ValueError):
            continue
    hoop_entries = court.get("hoops", []) or []
    for entry in hoop_entries:
        try:
            name = str(entry.get("name", "hoop"))
            x = float(entry["x"])
            y = float(entry["y"])
        except (KeyError, TypeError, ValueError):
            continue
        hoops.append((name, x, y))
    return polygon, hoops


def draw_hoops(frame, hoop_points: Iterable[Tuple[str, float, float]], color=(52, 152, 219), radius=40):
    for name, x, y in hoop_points:
        center = (int(round(x)), int(round(y)))
        cv2.circle(frame, center, radius, color, 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"Hoop: {name}",
            (center[0] - radius, center[1] - radius - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def render_video(
    video_path: Path,
    csv_path: Path,
    source: str,
    keep_all: bool,
    scale: float,
    hoop_points: Optional[List[Tuple[str, float, float]]] = None,
    hoop_radius: int = 40,
    court_outline: Optional[List[Tuple[int, int]]] = None,
    hoop_color: Tuple[int, int, int] = (52, 152, 219),
) -> Path:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No rows found in {csv_path.name}")

    grouped = df.groupby("frame")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if scale != 1.0:
        width = int(width * scale)
        height = int(height * scale)

    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RENDER_DIR / f"{video_path.stem}_{source}_overlay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    pbar = tqdm(total=total_frames, desc=f"Rendering {video_path.name}", unit="frame")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            draws = grouped.get_group(frame_idx) if frame_idx in grouped.groups else None
            if draws is not None or keep_all:
                if scale != 1.0:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                if draws is not None:
                    for _, row in draws.iterrows():
                        x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])
                        cls = int(row.get("cls", -1))
                        tid_value = row.get("track_id") if "track_id" in row else None
                        tid = int(tid_value) if pd.notna(tid_value) else None
                        color = COLOR_MAP.get(cls, (52, 152, 219))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                        prefix = LABEL_PREFIX.get(cls, f"C{cls}")
                        if tid is not None:
                            label = f"{prefix}#{tid}"
                        else:
                            label = prefix
                        conf = row.get("conf")
                        if pd.notna(conf):
                            label += f" ({float(conf):.2f})"
                        cv2.putText(
                            frame,
                            label,
                            (x1, max(y1 - 12, 24)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2,
                            cv2.LINE_AA,
                        )
                if hoop_points:
                    draw_hoops(frame, hoop_points, color=hoop_color, radius=hoop_radius)
                if court_outline and len(court_outline) >= 2:
                    cv2.polylines(frame, [np.array(court_outline, dtype=np.int32)], True, OUTLINE_COLOR, 2, cv2.LINE_AA)
                writer.write(frame)

            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        writer.release()

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render detections/tracks overlay onto video")
    parser.add_argument("--video", required=True, help="Video file name or path (must exist in input_videos/)")
    parser.add_argument(
        "--source",
        choices=["tracks", "detections"],
        default="tracks",
        help="Which CSV to use for overlay (default: tracks)",
    )
    parser.add_argument(
        "--keep-all-frames",
        action="store_true",
        help="Write all frames even if no detections (default: only frames with boxes)",
    )
    parser.add_argument("--scale", type=float, default=1.0, help="Optional resize factor for output video")
    parser.add_argument(
        "--config",
        type=Path,
        default=CFG_PATH,
        help="Config file containing hoop definitions (default: configs/salon.yaml)",
    )
    parser.add_argument("--hoop-radius", type=int, default=40, help="Hoop circle radius in pixels (default: 40)")
    parser.add_argument(
        "--no-hoops",
        action="store_true",
        help="Disable hoop overlay even if config has hoop entries",
    )
    parser.add_argument(
        "--no-court",
        action="store_true",
        help="Disable court polygon overlay",
    )
    parser.add_argument(
        "--hoop-color",
        type=str,
        help="Override hoop circle color as B,G,R (e.g. 0,255,255)",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        video_path = VIDEO_DIR / video_path.name
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    csv_path = resolve_csv(video_path.name, args.source)
    court_outline: Optional[List[Tuple[int, int]]] = None
    hoop_points: Optional[List[Tuple[str, float, float]]] = None
    if not args.no_hoops or not args.no_court:
        polygon, hoops = load_court_shapes(args.config)
        if not args.no_court and polygon:
            court_outline = polygon
        if not args.no_hoops and hoops:
            hoop_points = hoops
    hoop_color: Tuple[int, int, int] = (52, 152, 219)
    if args.hoop_color:
        try:
            parts = [int(p.strip()) for p in args.hoop_color.split(",")]
            if len(parts) == 3:
                hoop_color = (parts[0], parts[1], parts[2])
        except ValueError:
            hoop_color = (52, 152, 219)
    out_path = render_video(
        video_path,
        csv_path,
        args.source,
        args.keep_all_frames,
        args.scale,
        hoop_points=hoop_points,
        hoop_radius=args.hoop_radius,
        court_outline=court_outline,
        hoop_color=hoop_color,
    )
    print(f"[RENDER] Saved overlay video to {out_path}")


if __name__ == "__main__":
    main()
