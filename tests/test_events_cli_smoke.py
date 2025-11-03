from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.events import cli


@pytest.mark.integration
def test_events_cli_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Event CLI'nin temel akışını küçük bir örnek CSV ile doğrular.
    """
    output_dir = tmp_path
    monkeypatch.setattr(cli, "OUT_DIR", output_dir)

    video_name = "test_game.mp4"
    base_csv = output_dir / f"{video_name}_tracks.csv"

    track_rows = [
        {
            "frame": 0,
            "ts": 0.0,
            "cls": 0,
            "track_id": 1,
            "conf": 0.9,
            "x1": 0.0,
            "y1": 0.0,
            "x2": 50.0,
            "y2": 150.0,
            "cx": 25.0,
            "cy": 75.0,
            "w": 50.0,
            "h": 150.0,
            "age": 0,
            "hits": 1,
            "is_static": 0,
            "is_predicted": 0,
            "is_ref": 0,
        },
        {
            "frame": 0,
            "ts": 0.0,
            "cls": 32,
            "track_id": 100,
            "conf": 0.85,
            "x1": 60.0,
            "y1": 60.0,
            "x2": 68.0,
            "y2": 68.0,
            "cx": 64.0,
            "cy": 64.0,
            "w": 8.0,
            "h": 8.0,
            "age": 0,
            "hits": 1,
            "is_static": 0,
            "is_predicted": 0,
            "is_ref": 0,
        },
        {
            "frame": 1,
            "ts": 0.04,
            "cls": 0,
            "track_id": 1,
            "conf": 0.92,
            "x1": 2.0,
            "y1": 0.0,
            "x2": 52.0,
            "y2": 150.0,
            "cx": 27.0,
            "cy": 75.0,
            "w": 50.0,
            "h": 150.0,
            "age": 0,
            "hits": 2,
            "is_static": 0,
            "is_predicted": 0,
            "is_ref": 0,
        },
        {
            "frame": 1,
            "ts": 0.04,
            "cls": 32,
            "track_id": 100,
            "conf": 0.83,
            "x1": 66.0,
            "y1": 58.0,
            "x2": 74.0,
            "y2": 66.0,
            "cx": 70.0,
            "cy": 62.0,
            "w": 8.0,
            "h": 8.0,
            "age": 0,
            "hits": 2,
            "is_static": 0,
            "is_predicted": 0,
            "is_ref": 0,
        },
    ]
    pd.DataFrame(track_rows).to_csv(base_csv, index=False)

    detections_df = pd.DataFrame(
        [
            {
                "frame": 0,
                "ts": 0.0,
                "cls": 0,
                "conf": 0.9,
                "x1": 0.0,
                "y1": 0.0,
                "x2": 50.0,
                "y2": 150.0,
                "cx": 25.0,
                "cy": 75.0,
                "is_static": 0,
            },
            {
                "frame": 0,
                "ts": 0.0,
                "cls": 32,
                "conf": 0.85,
                "x1": 60.0,
                "y1": 60.0,
                "x2": 68.0,
                "y2": 68.0,
                "cx": 64.0,
                "cy": 64.0,
                "is_static": 0,
            },
        ]
    )
    detections_df.to_csv(output_dir / f"{video_name}_detections.csv", index=False)

    qc_path = cli.run(video=video_name, output=None)

    assert qc_path.exists()
    assert (output_dir / f"{video_name}_possessions.csv").exists()
    assert (output_dir / f"{video_name}_shots.csv").exists()
