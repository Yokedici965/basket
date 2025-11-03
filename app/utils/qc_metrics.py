from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

import pandas as pd


TRACK_COLUMNS: Set[str] = {
    "frame",
    "ts",
    "cls",
    "track_id",
    "conf",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx",
    "cy",
    "w",
    "h",
    "age",
    "hits",
    "is_static",
    "is_predicted",
}

DETECTION_COLUMNS: Set[str] = {
    "frame",
    "ts",
    "cls",
    "conf",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx",
    "cy",
    "is_static",
}


def _assert_columns(df: pd.DataFrame, expected: Set[str], name: str) -> None:
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def validate_track_schema(df: pd.DataFrame) -> None:
    _assert_columns(df, TRACK_COLUMNS, "tracks dataframe")


def validate_detection_schema(df: pd.DataFrame) -> None:
    _assert_columns(df, DETECTION_COLUMNS, "detections dataframe")


@dataclass(frozen=True)
class Phase1Metrics:
    frames_total: int
    duration_sec: float
    player_tracks: int
    player_frame_coverage: float
    player_mean_conf: float
    ball_tracks: int
    ball_frame_coverage: float
    ball_mean_conf: float
    short_lived_tracks: int
    max_ball_gap_frames: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "frames_total": self.frames_total,
            "duration_sec": round(self.duration_sec, 3),
            "player_tracks": self.player_tracks,
            "player_frame_coverage": round(self.player_frame_coverage, 4),
            "player_mean_conf": round(self.player_mean_conf, 4),
            "ball_tracks": self.ball_tracks,
            "ball_frame_coverage": round(self.ball_frame_coverage, 4),
            "ball_mean_conf": round(self.ball_mean_conf, 4),
            "short_lived_tracks": self.short_lived_tracks,
            "max_ball_gap_frames": self.max_ball_gap_frames,
        }


def _frame_coverage(df: pd.DataFrame, frame_span: int) -> float:
    if frame_span <= 0:
        return 0.0
    return df["frame"].nunique() / frame_span


def _mean_conf(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float(df["conf"].mean())


def _count_short_tracks(df: pd.DataFrame, *, max_frames: int) -> int:
    if df.empty:
        return 0
    span = df.groupby("track_id")["frame"].agg(lambda x: x.max() - x.min() + 1)
    return int((span < max_frames).sum())


def _max_ball_gap(ball_df: pd.DataFrame) -> int:
    if len(ball_df) <= 1:
        return 0
    frames = ball_df["frame"].sort_values().to_numpy()
    gaps = frames[1:] - frames[:-1]
    return int(gaps.max(initial=0))


def compute_phase1_metrics(
    track_df: pd.DataFrame,
    *,
    person_cls: int,
    ball_cls: int,
    short_track_frames: int = 3,
) -> Phase1Metrics:
    validate_track_schema(track_df)
    matched_df = track_df[track_df["is_predicted"] == 0]
    if matched_df.empty:
        matched_df = track_df
    if matched_df.empty:
        return Phase1Metrics(
            frames_total=0,
            duration_sec=0.0,
            player_tracks=0,
            player_frame_coverage=0.0,
            player_mean_conf=0.0,
            ball_tracks=0,
            ball_frame_coverage=0.0,
            ball_mean_conf=0.0,
            short_lived_tracks=0,
            max_ball_gap_frames=0,
        )

    frame_min = int(matched_df["frame"].min())
    frame_max = int(matched_df["frame"].max())
    frames_total = frame_max - frame_min + 1
    duration_sec = float(matched_df["ts"].max() - matched_df["ts"].min()) if "ts" in matched_df else 0.0

    players = matched_df[matched_df["cls"] == person_cls]
    ball_matched = matched_df[matched_df["cls"] == ball_cls]
    ball_all = track_df[track_df["cls"] == ball_cls]

    return Phase1Metrics(
        frames_total=frames_total,
        duration_sec=duration_sec,
        player_tracks=int(players["track_id"].nunique()),
        player_frame_coverage=_frame_coverage(players, frames_total),
        player_mean_conf=_mean_conf(players),
        ball_tracks=int(ball_matched["track_id"].nunique()),
        ball_frame_coverage=_frame_coverage(ball_matched, frames_total),
        ball_mean_conf=_mean_conf(ball_matched),
        short_lived_tracks=_count_short_tracks(players, max_frames=short_track_frames),
        max_ball_gap_frames=_max_ball_gap(ball_all),
    )
