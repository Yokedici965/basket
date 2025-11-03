"""Unit tests for app/utils/iou_tracker.py tracking utilities."""
from __future__ import annotations

import pytest
import numpy as np
from app.utils.iou_tracker import (
    iou_xyxy,
    IOUTracker,
    _create_kalman_filter,
    _state_to_bbox,
    Track
)


class TestIouXyxy:
    """Tests for IOU calculation."""

    @pytest.mark.unit
    def test_perfect_overlap(self):
        """Test IOU with identical boxes."""
        box = (0, 0, 10, 10)
        assert iou_xyxy(box, box) == pytest.approx(1.0)

    @pytest.mark.unit
    def test_no_overlap(self):
        """Test IOU with non-overlapping boxes."""
        box_a = (0, 0, 10, 10)
        box_b = (20, 20, 30, 30)
        assert iou_xyxy(box_a, box_b) == 0.0

    @pytest.mark.unit
    def test_partial_overlap(self):
        """Test IOU with 50% overlap."""
        box_a = (0, 0, 10, 10)
        box_b = (5, 0, 15, 10)
        # Intersection: 5x10=50, Union: 100+100-50=150, IOU=50/150=0.333
        assert iou_xyxy(box_a, box_b) == pytest.approx(0.333, abs=0.01)

    @pytest.mark.unit
    def test_contained_box(self):
        """Test IOU when one box contains another."""
        box_a = (0, 0, 20, 20)
        box_b = (5, 5, 15, 15)
        # Intersection: 10x10=100, Union: 400
        assert iou_xyxy(box_a, box_b) == pytest.approx(0.25)

    @pytest.mark.unit
    def test_edge_touching(self):
        """Test IOU with boxes touching at edge."""
        box_a = (0, 0, 10, 10)
        box_b = (10, 0, 20, 10)
        assert iou_xyxy(box_a, box_b) == 0.0

    @pytest.mark.unit
    @pytest.mark.parametrize("bbox_a,bbox_b,expected_iou", [
        ((0, 0, 10, 10), (0, 0, 10, 10), 1.0),
        ((0, 0, 10, 10), (20, 20, 30, 30), 0.0),
        ((0, 0, 10, 10), (5, 0, 15, 10), 0.333),
    ])
    def test_iou_parametrized(self, bbox_a, bbox_b, expected_iou):
        """Parametrized IOU test with multiple cases."""
        assert iou_xyxy(bbox_a, bbox_b) == pytest.approx(expected_iou, abs=0.01)


class TestIOUTracker:
    """Tests for IOUTracker class."""

    @pytest.mark.unit
    def test_initialization(self):
        """Test tracker initialization."""
        tracker = IOUTracker(iou_thr=0.3, max_age=10, center_dist_thr=50.0)
        assert tracker.iou_thr == 0.3
        assert tracker.max_age == 10
        assert tracker.center_dist_thr == 50.0
        assert len(tracker.tracks) == 0

    @pytest.mark.unit
    def test_single_detection_creates_track(self):
        """Test that first detection creates new track."""
        tracker = IOUTracker()
        dets = [(0, (100, 200, 150, 300), 0.9)]

        results = tracker.update(dets)

        assert len(results) == 1
        assert results[0]["track_id"] == 1
        assert results[0]["cls"] == 0
        assert results[0]["matched"] is True
        assert len(tracker.tracks) == 1

    @pytest.mark.unit
    def test_same_detection_continues_track(self):
        """Test that similar detection continues existing track."""
        tracker = IOUTracker(iou_thr=0.3)

        # First detection
        dets1 = [(0, (100, 200, 150, 300), 0.9)]
        results1 = tracker.update(dets1)
        track_id_1 = results1[0]["track_id"]

        # Similar detection (high IOU)
        dets2 = [(0, (102, 202, 152, 302), 0.9)]
        results2 = tracker.update(dets2)

        assert len(results2) == 1
        assert results2[0]["track_id"] == track_id_1  # Same track
        assert tracker.tracks[track_id_1].hit == 2

    @pytest.mark.unit
    def test_distant_detection_creates_new_track(self):
        """Test that distant detection creates new track."""
        tracker = IOUTracker(iou_thr=0.3, center_dist_thr=50.0)

        # First detection
        dets1 = [(0, (100, 200, 150, 300), 0.9)]
        tracker.update(dets1)

        # Distant detection
        dets2 = [(0, (500, 600, 550, 700), 0.9)]
        results2 = tracker.update(dets2)

        assert len(tracker.tracks) == 2
        assert results2[0]["track_id"] == 2

    @pytest.mark.unit
    def test_track_ages_without_match(self):
        """Test that tracks age when not matched."""
        tracker = IOUTracker(max_age=5)

        # Create track
        dets1 = [(0, (100, 200, 150, 300), 0.9)]
        results1 = tracker.update(dets1)
        track_id = results1[0]["track_id"]

        # Update without matching detection
        results2 = tracker.update([])

        assert len(results2) == 1
        assert results2[0]["track_id"] == track_id
        assert results2[0]["matched"] is False
        assert tracker.tracks[track_id].age == 1

    @pytest.mark.unit
    def test_old_tracks_removed(self):
        """Test that tracks exceeding max_age are removed."""
        tracker = IOUTracker(max_age=2)

        # Create track
        dets = [(0, (100, 200, 150, 300), 0.9)]
        results = tracker.update(dets)
        track_id = results[0]["track_id"]

        # Age out the track
        tracker.update([])  # age=1
        tracker.update([])  # age=2
        tracker.update([])  # age=3, should be removed

        assert track_id not in tracker.tracks

    @pytest.mark.unit
    def test_multiple_detections(self):
        """Test tracking multiple objects simultaneously."""
        tracker = IOUTracker()

        dets = [
            (0, (100, 200, 150, 300), 0.9),  # person 1
            (0, (500, 200, 550, 300), 0.85), # person 2
            (32, (300, 400, 320, 420), 0.8)  # ball
        ]

        results = tracker.update(dets)

        assert len(results) == 3
        assert len(tracker.tracks) == 3
        track_ids = [r["track_id"] for r in results]
        assert len(set(track_ids)) == 3  # All unique IDs

    @pytest.mark.unit
    def test_center_distance_matching(self):
        """Test matching by center distance when IOU is low."""
        tracker = IOUTracker(iou_thr=0.5, center_dist_thr=30.0)

        # Create track
        dets1 = [(0, (100, 200, 150, 300), 0.9)]
        results1 = tracker.update(dets1)
        track_id = results1[0]["track_id"]

        # Move box slightly (low IOU but close center)
        dets2 = [(0, (110, 210, 160, 310), 0.9)]
        results2 = tracker.update(dets2)

        assert results2[0]["track_id"] == track_id


class TestKalmanFilter:
    """Tests for Kalman filter utilities."""

    @pytest.mark.unit
    def test_create_kalman_filter(self):
        """Test Kalman filter creation."""
        kf = _create_kalman_filter(100.0, 200.0, 50.0, 100.0)

        if kf is not None:  # filterpy installed
            assert kf.dim_x == 6
            assert kf.dim_z == 4
            assert kf.x[0, 0] == 100.0
            assert kf.x[1, 0] == 200.0

    @pytest.mark.unit
    def test_state_to_bbox(self):
        """Test converting Kalman state to bbox."""
        state = np.array([[100], [200], [0], [0], [50], [100]])
        bbox = _state_to_bbox(state)

        expected = (75.0, 150.0, 125.0, 250.0)  # (cx-w/2, cy-h/2, cx+w/2, cy+h/2)
        assert bbox == pytest.approx(expected)

    @pytest.mark.unit
    def test_state_to_bbox_minimum_size(self):
        """Test bbox conversion with minimum size constraint."""
        state = np.array([[100], [200], [0], [0], [0.5], [0.5]])
        bbox = _state_to_bbox(state)

        # Width/height should be clamped to 1.0
        assert bbox[2] - bbox[0] == pytest.approx(1.0)
        assert bbox[3] - bbox[1] == pytest.approx(1.0)


class TestTrack:
    """Tests for Track dataclass."""

    @pytest.mark.unit
    def test_track_creation(self):
        """Test Track object creation."""
        track = Track(
            tid=1,
            cls=0,
            bbox=(100, 200, 150, 300),
            age=0,
            hit=1,
            conf=0.9,
            cx=125.0,
            cy=250.0
        )

        assert track.tid == 1
        assert track.cls == 0
        assert track.age == 0
        assert track.hit == 1
