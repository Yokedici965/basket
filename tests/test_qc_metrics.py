from __future__ import annotations

import pandas as pd
import pytest

from app.utils import qc_metrics


def _sample_tracks() -> pd.DataFrame:
    rows = [
        # player track id 1 across frames 0-2
        dict(frame=0, ts=0.0, cls=0, track_id=1, conf=0.9, x1=0, y1=0, x2=10, y2=10, cx=5, cy=5, w=10, h=10, age=0, hits=1, is_static=0, is_predicted=0),
        dict(frame=1, ts=0.04, cls=0, track_id=1, conf=0.92, x1=1, y1=1, x2=11, y2=11, cx=6, cy=6, w=10, h=10, age=0, hits=2, is_static=0, is_predicted=0),
        dict(frame=2, ts=0.08, cls=0, track_id=1, conf=0.93, x1=2, y1=2, x2=12, y2=12, cx=7, cy=7, w=10, h=10, age=0, hits=3, is_static=0, is_predicted=0),
        # ball track id 100 frames 0,2
        dict(frame=0, ts=0.0, cls=32, track_id=100, conf=0.8, x1=20, y1=20, x2=24, y2=24, cx=22, cy=22, w=4, h=4, age=0, hits=1, is_static=0, is_predicted=0),
        dict(frame=2, ts=0.08, cls=32, track_id=100, conf=0.81, x1=30, y1=25, x2=34, y2=29, cx=32, cy=27, w=4, h=4, age=0, hits=2, is_static=0, is_predicted=0),
        # short-lived player track id 5 frame 1 only
        dict(frame=1, ts=0.04, cls=0, track_id=5, conf=0.7, x1=50, y1=50, x2=60, y2=60, cx=55, cy=55, w=10, h=10, age=0, hits=1, is_static=0, is_predicted=0),
    ]
    return pd.DataFrame(rows)


def _sample_detections() -> pd.DataFrame:
    rows = [
        dict(frame=0, ts=0.0, cls=0, conf=0.9, x1=0, y1=0, x2=10, y2=10, cx=5, cy=5, is_static=0, is_predicted=0),
        dict(frame=1, ts=0.04, cls=32, conf=0.8, x1=20, y1=20, x2=24, y2=24, cx=22, cy=22, is_static=0, is_predicted=0),
    ]
    return pd.DataFrame(rows)


def test_compute_phase1_metrics_basic():
    df = _sample_tracks()
    metrics = qc_metrics.compute_phase1_metrics(df, person_cls=0, ball_cls=32)
    payload = metrics.to_dict()

    assert payload["frames_total"] == 3
    assert payload["player_tracks"] == 2
    assert pytest.approx(payload["player_frame_coverage"], rel=1e-5) == 1.0
    assert payload["ball_tracks"] == 1
    assert payload["short_lived_tracks"] == 1  # short track length is 1 frame
    assert payload["max_ball_gap_frames"] == 2


def test_validate_track_schema_missing_column():
    df = _sample_tracks().drop(columns=["hits"])
    with pytest.raises(ValueError):
        qc_metrics.validate_track_schema(df)


def test_validate_detection_schema_missing_column():
    df = _sample_detections().drop(columns=["cx"])
    with pytest.raises(ValueError):
        qc_metrics.validate_detection_schema(df)
