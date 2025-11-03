"""
Unified Event Detection Engine

Orchestrates all event detectors and manages event sequences.
Replaces the old event/engine.py with modular, scalable design.
"""

from typing import Dict, List, Optional
import numpy as np
from pathlib import Path
import pandas as pd
import cv2

from .base_detector import BaseEventDetector, Event, get_detector
from .event_sequencer import EventSequencer


class UnifiedEventEngine:
    """
    Main event detection engine.

    Manages:
    1. All event detectors
    2. Frame-by-frame event detection
    3. Event sequencing and validation
    4. Output generation (CSV, JSON)
    """

    # List of event types to detect
    DEFAULT_EVENT_TYPES = [
        "possession",
        "shot_attempt",
        "shot_make",
        "shot_miss",
        "rebound",
        "steal",
        "block",
        "assist",
        "turnover",
        "foul",
    ]

    def __init__(self, fps: float = 30.0,
                 event_types: Optional[List[str]] = None):
        """
        Initialize event engine.

        Args:
            fps: Video frame rate
            event_types: List of event types to detect
        """
        self.fps = fps
        self.frame_time = 1.0 / fps

        # Initialize detectors
        event_types = event_types or self.DEFAULT_EVENT_TYPES
        self.detectors: Dict[str, BaseEventDetector] = {}

        for event_type in event_types:
            try:
                self.detectors[event_type] = get_detector(event_type)
            except ValueError:
                print(f"Warning: Unknown event type '{event_type}', skipping")

        # Event sequencer
        self.sequencer = EventSequencer()
        self.sequencer.fps = fps

        # Statistics
        self.stats = {
            "frames_processed": 0,
            "events_detected": 0,
            "detections_by_type": {et: 0 for et in event_types},
        }

    def process_frame(self, frame_idx: int,
                     detections: Dict) -> Optional[Event]:
        """
        Process single frame and detect events.

        Args:
            frame_idx: Frame index (0-based)
            detections: Detection data from YOLO/tracker
                {
                    "tracks": [(track_id, x, y, w, h, conf), ...],
                    "ball": (x, y, w, h, conf) or None,
                    "timestamp": float,
                }

        Returns:
            Most prominent event detected in this frame, or None
        """
        self.stats["frames_processed"] += 1

        previous_events = self.sequencer.get_events()
        detected_events = []

        # Run all detectors
        for event_type, detector in self.detectors.items():
            event = detector.detect(frame_idx, detections, previous_events)

            if event is not None and detector.validate_event(event):
                detected_events.append(event)
                self.stats["detections_by_type"][event_type] += 1
                self.stats["events_detected"] += 1

        # Add detected events to sequencer
        for event in detected_events:
            self.sequencer.add_event(event)

        # Return most confident event
        if detected_events:
            return max(detected_events, key=lambda e: e.confidence)

        return None

    def process_video(self, video_path: str,
                     detections_csv: str,
                     output_json: Optional[str] = None) -> Dict:
        """
        Process complete video and generate event stream.

        Args:
            video_path: Path to video file
            detections_csv: Path to YOLO detections CSV
            output_json: Path to save JSON output

        Returns:
            Event summary dict
        """
        import cv2
        import pandas as pd

        # Load video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Load detections
        detections_df = pd.read_csv(detections_csv)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Process frames
        for frame_idx in range(total_frames):
            # Get detections for this frame
            frame_dets = detections_df[detections_df['frame'] == frame_idx]

            # Parse detections into dict
            detections = self._parse_detections(frame_dets)

            # Process frame
            self.process_frame(frame_idx, detections)

            # Progress
            if (frame_idx + 1) % 1000 == 0:
                print(f"[{frame_idx + 1}/{total_frames}] "
                      f"Events: {self.stats['events_detected']}")

        cap.release()

        # Generate output
        result = self.get_summary()

        if output_json:
            self.save_to_json(output_json)

        return result

    @staticmethod
    def _parse_detections(detections_df) -> Dict:
        """
        Parse detections from CSV into format for event detection.

        Args:
            detections_df: Pandas DataFrame of detections for one frame

        Returns:
            Dict with keys: 'tracks', 'ball', 'timestamp'
        """
        tracks = []
        ball = None

        for _, row in detections_df.iterrows():
            # CSV'ler x1,y1,x2,y2 kolonlarını tutuyor; gerekirse w/h hesapla
            x1 = float(row.get("x1", row.get("x", 0.0)))
            y1 = float(row.get("y1", row.get("y", 0.0)))
            x2 = float(row.get("x2", x1 + row.get("w", 0.0)))
            y2 = float(row.get("y2", y1 + row.get("h", 0.0)))
            w = max(x2 - x1, 0.0)
            h = max(y2 - y1, 0.0)
            cls_val = row.get("cls", row.get("class_name"))
            try:
                class_id = int(cls_val)
            except (TypeError, ValueError):
                class_id = None
            conf = float(row.get("conf", 0.0))
            track_raw = row.get("track_id", -1)
            try:
                track_id = int(track_raw)
            except (TypeError, ValueError):
                track_id = -1

            if class_id == 32:
                ball = (x1, y1, w, h, conf)
            else:
                tracks.append((track_id, x1, y1, w, h, conf))

        timestamp = float(detections_df["ts"].iloc[0]) if "ts" in detections_df and not detections_df.empty else 0.0

        return {
            "tracks": tracks,
            "ball": ball,
            "timestamp": timestamp,
        }

    def get_summary(self) -> Dict:
        """Get event detection summary."""
        events = self.sequencer.get_events()

        summary = {
            "total_frames": self.stats["frames_processed"],
            "total_events": len(events),
            "events": [e.to_dict() for e in events],
            "stats": self.stats,
            "sequence": self.sequencer.get_sequence_string(max_events=30),
        }

        return summary

    def save_to_json(self, filepath: str) -> None:
        """Save events to JSON."""
        import json

        summary = self.get_summary()

        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"Saved events to {filepath}")

    def save_to_csv(self, filepath: str) -> None:
        """Save events to CSV."""
        import pandas as pd

        events = self.sequencer.get_events()

        data = []
        for event in events:
            data.append({
                "event_type": event.event_type,
                "confidence": event.confidence,
                "frame_start": event.frame_start,
                "frame_end": event.frame_end,
                "duration_frames": event.duration_frames,
                "player_id": event.player_id,
                "team": event.team,
            })

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"Saved {len(df)} events to {filepath}")

    def validate_events(self) -> bool:
        """Validate event sequence."""
        return self.sequencer.validate_sequence()

    def print_summary(self):
        """Print human-readable summary."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("EVENT DETECTION SUMMARY")
        print("=" * 60)
        print(f"Total Frames: {summary['total_frames']}")
        print(f"Total Events: {summary['total_events']}")
        print(f"\nEvents by Type:")

        for event_type, count in summary['stats']['detections_by_type'].items():
            if count > 0:
                print(f"  {event_type}: {count}")

        print(f"\nEvent Sequence (first 20):")
        print(f"  {summary['sequence']}")
        print("=" * 60 + "\n")
