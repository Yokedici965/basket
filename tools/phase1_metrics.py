from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from app.utils.qc_metrics import (
    DETECTION_COLUMNS,
    Phase1Metrics,
    compute_phase1_metrics,
    validate_detection_schema,
    validate_track_schema,
)


def _load_classes(cfg: dict) -> tuple[int, int]:
    detection = cfg.get("detection", {})
    classes = detection.get("classes", {})
    person = int(classes.get("person"))
    ball = int(classes.get("ball"))
    return person, ball


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg


def _summarise_detections(det_df: pd.DataFrame) -> dict:
    validate_detection_schema(det_df)
    total = len(det_df)
    by_cls = (
        det_df.groupby("cls")
        .size()
        .sort_index()
        .to_dict()
    )
    return {"total_detections": int(total), "detections_by_cls": {int(k): int(v) for k, v in by_cls.items()}}


def _save_metrics(path: Optional[Path], metrics: Phase1Metrics, extra: dict) -> None:
    payload = {"metrics": metrics.to_dict(), **extra}
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Phase 1 QC metrics for tracking outputs.")
    parser.add_argument("--tracks", required=True, type=Path, help="Path to *_tracks.csv produced by app/run.py")
    parser.add_argument("--detections", type=Path, help="Optional path to *_detections.csv")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs") / "salon.yaml",
        help="Config file used during inference (default: configs/salon.yaml)",
    )
    parser.add_argument("--out", type=Path, help="Optional JSON file to write metrics into runs/ or outputs/")
    parser.add_argument(
        "--short-track-threshold",
        type=int,
        default=3,
        help="Track length (in frames) considered short-lived for diagnostics (default: 3)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)
    person_cls, ball_cls = _load_classes(cfg)

    track_df = pd.read_csv(args.tracks)
    validate_track_schema(track_df)
    metrics = compute_phase1_metrics(
        track_df,
        person_cls=person_cls,
        ball_cls=ball_cls,
        short_track_frames=args.short_track_threshold,
    )

    summary = {}
    if args.detections:
        det_df = pd.read_csv(args.detections)
        summary["detections"] = _summarise_detections(det_df)

    _save_metrics(args.out, metrics, summary)


if __name__ == "__main__":
    main()

