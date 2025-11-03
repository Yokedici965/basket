from __future__ import annotations

import argparse
from pathlib import Path

from app.calibration.hoop_locator import calibrate_hoops

BASE = Path(__file__).resolve().parents[2]
CFG_PATH = BASE / "configs" / "salon.yaml"
CALIB_DIR = BASE / "configs" / "calibrations"


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated hoop calibration from video frames")
    parser.add_argument("--video", required=True, help="Video file name in input_videos or absolute path")
    parser.add_argument("--stride", type=float, default=5.0, help="Sample stride in seconds (default 5)")
    parser.add_argument("--max-samples", type=int, default=20, help="Max sampled frames")
    parser.add_argument("--yolo-weights", help="Optional Ultralytics model path for hoop fallback detection")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="Confidence threshold for YOLO fallback")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        video_path = BASE / "input_videos" / video_path.name
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    qc = calibrate_hoops(
        video_path,
        config_path=CFG_PATH,
        output_dir=CALIB_DIR,
        sample_stride_sec=args.stride,
        max_samples=args.max_samples,
        yolo_weights=Path(args.yolo_weights) if args.yolo_weights else None,
        yolo_conf=args.yolo_conf,
    )
    print("[CALIBRATION]", qc)


if __name__ == "__main__":
    main()
