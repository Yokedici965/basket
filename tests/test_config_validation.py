from __future__ import annotations

import pytest

from app.run import validate_config


def _base_config():
    return {
        "court": {
            "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "ref_zones": [],
        },
        "detection": {
            "classes": {"person": 0, "ball": 32},
            "thresholds": {"person": 0.1, "ball": 0.2},
            "static_ball": {"frames": 10, "grid_px": 4},
        },
        "tracking": {"iou_threshold": 0.4, "max_age": 10, "center_distance_threshold": 50.0},
        "processing": {"frame_stride": 1},
    }


def test_validate_config_with_valid_payload():
    cfg = _base_config()
    # Should not raise
    validate_config(cfg)


@pytest.mark.parametrize(
    "field, value, expected_msg",
    [
        ("court", None, "court section is required"),
        ("detection", {"classes": {"person": 0}}, "detection.classes.ball is required"),
    ],
)
def test_validate_config_missing_sections(field, value, expected_msg):
    cfg = _base_config()
    cfg[field] = value
    with pytest.raises(ValueError) as exc:
        validate_config(cfg)
    assert expected_msg in str(exc.value)


def test_validate_config_static_ball_frames_invalid():
    cfg = _base_config()
    cfg["detection"]["static_ball"]["frames"] = 0
    with pytest.raises(ValueError) as exc:
        validate_config(cfg)
    assert "detection.static_ball.frames" in str(exc.value)
