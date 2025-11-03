from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from app.calibration.hoop_locator import (
    FrameSample,
    HoopCandidate,
    _compute_homographies,
    _detect_circles,
    _sample_frames,
    _warp,
)

BASE = Path(__file__).resolve().parents[2]
OUT_DIR = BASE / "configs" / "calibrations"


@dataclass
class SampleInfo:
    frame_idx: int
    ts: float
    raw_path: Path
    overlay_path: Path
    meta_path: Path
    skipped: bool


def draw_overlays(frame: np.ndarray, candidates: List[HoopCandidate]):
    canvas = frame.copy()
    for cand in candidates:
        cv2.circle(canvas, (int(cand.x), int(cand.y)), int(cand.radius), (0, 255, 0), 2)
        cv2.putText(
            canvas,
            f"r={int(cand.radius)} conf={cand.confidence:.2f}",
            (max(int(cand.x - cand.radius), 0), max(int(cand.y - cand.radius - 10), 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Export hoop calibration samples with overlays and metadata")
    parser.add_argument("--video", required=True, help="Video file name or path")
    parser.add_argument("--stride", type=float, default=5.0, help="Sampling stride in seconds")
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--keep-empty", action="store_true", help="Save frames even when Hough detection is empty")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        video_path = BASE / "input_videos" / video_path.name
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_dir = OUT_DIR / f"{video_path.stem}_samples"
    overlay_dir = sample_dir / "overlay"
    raw_dir = sample_dir / "raw"
    meta_dir = sample_dir / "meta"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Video açılamadı: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride_frames = max(1, int(args.stride * fps))

    samples: List[FrameSample] = _sample_frames(cap, stride_frames, args.max_samples)
    if not samples:
        raise RuntimeError("Örnek kare alınamadı")

    ref_frame = samples[0].frame
    homographies, _ = _compute_homographies(ref_frame, [sample.frame for sample in samples])
    h, w = ref_frame.shape[:2]

    report: List[SampleInfo] = []

    for sample, H in zip(samples, homographies):
        warp = _warp(sample.frame, H, (w, h))
        candidates = _detect_circles(warp, sample.frame_idx, sample.ts, w)

        raw_path = raw_dir / f"frame_{sample.frame_idx:06d}_t{sample.ts:.2f}.jpg"
        overlay_path = overlay_dir / f"frame_{sample.frame_idx:06d}_t{sample.ts:.2f}.jpg"
        meta_path = meta_dir / f"frame_{sample.frame_idx:06d}.json"

        if candidates or args.keep_empty:
            cv2.imwrite(str(raw_path), warp)
            overlay = draw_overlays(warp, candidates)
            cv2.imwrite(str(overlay_path), overlay)
            meta: Dict[str, object] = {
                "video": video_path.name,
                "frame": sample.frame_idx,
                "ts": sample.ts,
                "width": w,
                "height": h,
                "homography": H.tolist(),
                "detections": [
                    {
                        "method": cand.method,
                        "x": float(cand.x),
                        "y": float(cand.y),
                        "radius": float(cand.radius),
                        "confidence": float(cand.confidence),
                    }
                    for cand in candidates
                ],
            }
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            report.append(SampleInfo(sample.frame_idx, sample.ts, raw_path, overlay_path, meta_path, skipped=False))
        else:
            report.append(SampleInfo(sample.frame_idx, sample.ts, raw_path, overlay_path, meta_path, skipped=True))

    cap.release()

    kept = sum(1 for r in report if not r.skipped)
    skipped = len(report) - kept
    print(f"[SAMPLES] kept {kept} frames, skipped {skipped} (keep_empty={args.keep_empty}) -> {sample_dir}")


if __name__ == "__main__":
    main()
