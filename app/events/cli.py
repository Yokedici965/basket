from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from app.events.engine import EventEngine, Possession, ShotEvent, ReboundEvent

BASE = Path(__file__).resolve().parents[2]
OUT_DIR = BASE / "outputs"
CFG_PATH = BASE / "configs" / "salon.yaml"


def _resolve_base(video: str) -> str:
    name = Path(video).name
    return name if name.endswith(".mp4") else f"{name}.mp4"


def _load_config() -> dict:
    if not CFG_PATH.exists():
        raise FileNotFoundError(CFG_PATH)
    with CFG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _build_engine(cfg: dict) -> EventEngine:
    events_cfg = cfg.get("events", {})
    court_cfg = cfg.get("court", {})
    hoops = court_cfg.get("hoops")
    return EventEngine(
        ball_gap_sec=float(events_cfg.get("ball_gap_sec", 1.0)),
        min_possession_frames=int(events_cfg.get("min_possession_frames", 2)),
        possession_dist_threshold=float(events_cfg.get("possession_dist_threshold", 140.0)),
        hoops=hoops,
        hoop_radius=float(events_cfg.get("hoop_radius", 140.0)),
        shot_speed_threshold=float(events_cfg.get("shot_speed_threshold", 55.0)),
        shot_up_threshold=float(events_cfg.get("shot_up_threshold", 25.0)),
        rebound_window_sec=float(events_cfg.get("rebound_window_sec", 2.5)),
        made_window_sec=float(events_cfg.get("made_window_sec", 1.2)),
    )


def _qc_report(
    *,
    possessions: list[Possession],
    shots: list[ShotEvent],
    rebounds: list[ReboundEvent],
    ball_df_len: int,
    flags: list[dict],
    engine: EventEngine,
) -> dict:
    unknown_shots = sum(1 for s in shots if s.result == "unknown")
    miss_shots = sum(1 for s in shots if s.result == "miss")
    made_shots = sum(1 for s in shots if s.result == "made")
    avg_poss_len = (
        sum(p.end_ts - p.start_ts for p in possessions) / len(possessions)
        if possessions
        else 0.0
    )
    avg_poss_dist = (
        sum(p.controller_dist_mean for p in possessions) / len(possessions)
        if possessions
        else 0.0
    )
    rebound_ratio = len(rebounds) / len(shots) if shots else 0.0
    return {
        "ball_frames": ball_df_len,
        "possessions": len(possessions),
        "avg_possession_sec": round(avg_poss_len, 3),
        "avg_possession_dist": round(avg_poss_dist, 2),
        "shot_events": len(shots),
        "shot_made": made_shots,
        "shot_miss": miss_shots,
        "shot_unknown": unknown_shots,
        "rebound_events": len(rebounds),
        "rebound_per_shot": round(rebound_ratio, 3),
        "ball_gap_sec_threshold": engine.ball_gap_sec,
        "flags": len(flags),
    }


def _collect_flags(
    *,
    possessions: list[Possession],
    shots: list[ShotEvent],
    engine: EventEngine,
) -> list[dict]:
    flags: list[dict] = []
    for shot in shots:
        if shot.result == "unknown":
            flags.append({
                "type": "shot_unknown",
                "frame": shot.frame,
                "ts": shot.ts,
                "shooter_track": shot.shooter_track,
                "velocity": shot.velocity,
            })
        elif shot.velocity < engine.shot_speed_threshold * 0.75:
            flags.append({
                "type": "shot_low_velocity",
                "frame": shot.frame,
                "ts": shot.ts,
                "shooter_track": shot.shooter_track,
                "velocity": shot.velocity,
            })
    for poss in possessions:
        if poss.controller_dist_mean > engine.possession_dist_threshold:
            flags.append({
                "type": "possession_high_distance",
                "start_frame": poss.start_frame,
                "end_frame": poss.end_frame,
                "start_ts": poss.start_ts,
                "end_ts": poss.end_ts,
                "controller_track": poss.controller_track,
                "dist_mean": poss.controller_dist_mean,
            })
    return flags


def run(video: str, output: Optional[str]) -> Path:
    base_name = _resolve_base(video)
    tracks_csv = OUT_DIR / f"{base_name}_tracks.csv"
    det_csv = OUT_DIR / f"{base_name}_detections.csv"

    cfg = _load_config()
    engine = _build_engine(cfg)

    tracks_df = engine.load_tracks(tracks_csv)
    engine.load_detections(det_csv)

    possessions, nearest, ball_df = engine.compute_possessions(tracks_df)
    shot_data = engine.detect_shots_and_rebounds(ball_df, nearest)

    target_dir = OUT_DIR if output is None else Path(output)
    engine.export(
        possessions=possessions,
        shots=shot_data["shots"],
        rebounds=shot_data["rebounds"],
        base_name=base_name,
        out_dir=target_dir,
    )

    flags = _collect_flags(possessions=possessions, shots=shot_data["shots"], engine=engine)
    if flags:
        flags_path = target_dir / f"{base_name}_event_flags.csv"
        pd.DataFrame(flags).to_csv(flags_path, index=False)

    qc = _qc_report(
        possessions=possessions,
        shots=shot_data["shots"],
        rebounds=shot_data["rebounds"],
        ball_df_len=len(ball_df),
        flags=flags,
        engine=engine,
    )
    qc_path = target_dir / f"{base_name}_events_qc.json"
    with qc_path.open("w", encoding="utf-8") as fh:
        json.dump(qc, fh, ensure_ascii=False, indent=2)

    print("[QC]", json.dumps(qc, ensure_ascii=False))
    return qc_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute basketball events from tracking CSVs.")
    parser.add_argument("--video", required=True, help="Video name or outputs prefix (e.g. mac2.mp4)")
    parser.add_argument("--out-dir", help="Optional override output directory")
    args = parser.parse_args()

    run(video=args.video, output=args.out_dir)


if __name__ == "__main__":
    main()
