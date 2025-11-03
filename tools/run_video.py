from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from app import run as run_mod
from app.events import cli as events_cli
from app.utils.qc_metrics import compute_phase1_metrics, validate_detection_schema, validate_track_schema
from tools import phase1_metrics as metrics_mod  # noqa: F401  # ensures dependency visibility
from tools.render_overlays import load_court_shapes, render_video, resolve_csv

BASE = Path(__file__).resolve().parents[1]
VIDEO_DIR = BASE / "input_videos"
OUTPUT_DIR = BASE / "outputs"
RUNS_DIR = BASE / "runs" / "phase1"
QC_LOG = BASE / "docs" / "reports" / "highlight_qc.md"


def _resolve_video(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    candidate = VIDEO_DIR / candidate.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Video not found: {path}")


def _append_qc_row(video_name: str, ball_gap: float, static_filtered: int) -> None:
    if not QC_LOG.exists():
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    row = f"| {timestamp} | {video_name} | {ball_gap} | {static_filtered} |  |\n"
    with QC_LOG.open("a", encoding="utf-8") as fh:
        fh.write(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detection → metrics → overlay pipeline for a single video.")
    parser.add_argument("--video", required=True, help="Video file name or path")
    parser.add_argument("--config", type=Path, help="Optional config yaml (default: configs/salon.yaml)")
    parser.add_argument("--frame-stride", type=int, help="Override processing.frame_stride")
    parser.add_argument("--duration-sec", type=float, help="Optional processing duration override")
    parser.add_argument("--start-sec", type=float, help="Optional start time override")
    parser.add_argument("--run-events", action="store_true", help="Trigger app.events CLI after detection")
    parser.add_argument("--skip-overlay", action="store_true", help="Skip overlay rendering")
    parser.add_argument("--metrics-out", type=Path, help="Optional metrics json path (default runs/phase1/<video>_metrics.json)")
    parser.add_argument("--overlay-scale", type=float, default=1.0, help="Overlay output scale (default 1.0)")
    parser.add_argument("--overlay-hoop-radius", type=int, default=60, help="Overlay hoop circle radius (default 60)")
    parser.add_argument("--no-qc-log", action="store_true", help="Do not append to docs/reports/highlight_qc.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.config:
        run_mod.CFG_PATH = Path(args.config)
    cfg, model_path = run_mod.load_config()
    cfg_local = copy.deepcopy(cfg)

    proc_cfg = cfg_local.setdefault("processing", {})
    if args.frame_stride:
        proc_cfg["frame_stride"] = int(args.frame_stride)
    if args.duration_sec is not None:
        proc_cfg["duration_sec"] = float(args.duration_sec)
    if args.start_sec is not None:
        proc_cfg["start_time_sec"] = float(args.start_sec)

    detection_cfg = cfg_local.get("detection", {})
    use_gpu = bool(detection_cfg.get("use_gpu", True))
    half_precision = bool(detection_cfg.get("half_precision", True))
    device_kw = run_mod.get_device_kw(use_gpu=use_gpu, half_precision=half_precision)

    video_path = _resolve_video(args.video)
    stats = run_mod.process_video(video_path, cfg_local, model_path, device_kw)

    tracks_csv = OUTPUT_DIR / f"{video_path.name}_tracks.csv"
    det_csv = OUTPUT_DIR / f"{video_path.name}_detections.csv"

    if not tracks_csv.exists():
        raise FileNotFoundError(tracks_csv)
    track_df = pd.read_csv(tracks_csv)
    validate_track_schema(track_df)

    metrics = compute_phase1_metrics(
        track_df,
        person_cls=int(detection_cfg.get("classes", {}).get("person", run_mod.DEFAULT_PERSON_CLS)),
        ball_cls=int(detection_cfg.get("classes", {}).get("ball", run_mod.DEFAULT_BALL_CLS)),
    )
    summary = {"metrics": metrics.to_dict()}

    if det_csv.exists():
        det_df = pd.read_csv(det_csv)
        validate_detection_schema(det_df)
        summary["detections"] = {
            "total": int(len(det_df)),
            "counts": {int(cls): int(count) for cls, count in det_df.groupby("cls").size().items()},
        }

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = args.metrics_out or (RUNS_DIR / f"{video_path.name}_metrics.json")
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[METRICS] wrote {metrics_path}")

    if args.run_events:
        try:
            events_cli.run(video=f"{video_path.stem}.mp4", output=None)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Event CLI failed: {exc}")

    if not args.skip_overlay:
        polygon, hoops = load_court_shapes(run_mod.CFG_PATH)
        csv_path = resolve_csv(video_path.name, "tracks")
        render_video(
            video_path,
            csv_path,
            "tracks",
            keep_all=True,
            scale=args.overlay_scale,
            hoop_points=hoops,
            hoop_radius=args.overlay_hoop_radius,
            court_outline=polygon,
        )

    if not args.no_qc_log and stats is not None:
        _append_qc_row(
            video_path.name,
            int(summary["metrics"]["max_ball_gap_frames"]),
            int(stats.get("static_ball_flagged", 0)),
        )


if __name__ == "__main__":
    main()
