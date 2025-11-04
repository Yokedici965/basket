#!/usr/bin/env python3
"""
Visualize tracking and detection results on video.

Draws bounding boxes with track_id, confidence, and class labels on video frames.

Usage:
    python visualize_results.py --video video.mp4 --tracks tracks.csv --detections detections.csv
    python visualize_results.py --video video.mp4 --tracks tracks.csv --output output.mp4
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import csv

import cv2
import numpy as np


class VideoVisualizer:
    """Visualizes tracking and detection results on video."""

    COLORS = {
        0: (0, 255, 0),      # Person: Green
        32: (0, 0, 255),     # Ball: Red
    }

    CLASS_NAMES = {
        0: "Person",
        32: "Ball",
    }

    def __init__(self, video_path: str, output_path: str = None, show_detections: bool = False):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        # Output video
        if output_path is None:
            output_path = self.video_path.stem + "_visualized.mp4"
        self.output_path = Path(output_path)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (self.width, self.height)
        )

        self.show_detections = show_detections
        self.tracks = {}  # frame -> list of track dicts
        self.detections = {}  # frame -> list of detection dicts

    def load_tracks(self, csv_path: str):
        """Load track data from CSV."""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"⚠️  Tracks CSV not found: {csv_path}")
            return

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame = int(row["frame"])
                if frame not in self.tracks:
                    self.tracks[frame] = []

                self.tracks[frame].append({
                    "track_id": int(row["track_id"]),
                    "cls": int(row["cls"]),
                    "conf": float(row["conf"]),
                    "x1": float(row["x1"]),
                    "y1": float(row["y1"]),
                    "x2": float(row["x2"]),
                    "y2": float(row["y2"]),
                    "is_predicted": int(row["is_predicted"]),
                    "is_static": int(row["is_static"]),
                })

        print(f"✅ Loaded {len(self.tracks)} frames with tracking data")

    def load_detections(self, csv_path: str):
        """Load detection data from CSV."""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"⚠️  Detections CSV not found: {csv_path}")
            return

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame = int(row["frame"])
                if frame not in self.detections:
                    self.detections[frame] = []

                self.detections[frame].append({
                    "cls": int(row["cls"]),
                    "conf": float(row["conf"]),
                    "x1": float(row["x1"]),
                    "y1": float(row["y1"]),
                    "x2": float(row["x2"]),
                    "y2": float(row["y2"]),
                    "is_static": int(row["is_static"]),
                })

        print(f"✅ Loaded {len(self.detections)} frames with detection data")

    def draw_box(self, frame, x1, y1, x2, y2, cls_id, conf, track_id=None, is_predicted=False, is_static=False):
        """Draw bounding box with label on frame."""
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = self.COLORS.get(cls_id, (255, 255, 0))

        # Dashed line if predicted
        if is_predicted:
            # Draw dashed rectangle
            pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            for i in range(len(pts)):
                pt1 = pts[i]
                pt2 = pts[(i + 1) % len(pts)]
                for j in range(0, int(np.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)), 15):
                    ratio = j / max(np.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2), 1)
                    x = int(pt1[0] + (pt2[0] - pt1[0]) * ratio)
                    y = int(pt1[1] + (pt2[1] - pt1[1]) * ratio)
                    cv2.circle(frame, (x, y), 2, color, -1)
        else:
            # Solid rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label
        class_name = self.CLASS_NAMES.get(cls_id, f"Class {cls_id}")
        if track_id is not None:
            label = f"ID:{track_id} {class_name} {conf:.2f}"
        else:
            label = f"{class_name} {conf:.2f}"

        if is_static:
            label += " [STATIC]"

        # Background for text
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_x, text_y = x1, y1 - 5
        cv2.rectangle(frame, (text_x, text_y - text_height - baseline), (text_x + text_width, text_y), color, -1)
        cv2.putText(frame, label, (text_x, text_y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    def visualize(self, display: bool = False):
        """Visualize tracks and detections on video."""
        print(f"Processing {self.total_frames} frames...")
        processed = 0

        for frame_idx in range(self.total_frames):
            ret, frame = self.cap.read()
            if not ret:
                break

            # Draw tracks
            if frame_idx in self.tracks:
                for track in self.tracks[frame_idx]:
                    self.draw_box(
                        frame,
                        track["x1"], track["y1"], track["x2"], track["y2"],
                        track["cls"], track["conf"],
                        track_id=track["track_id"],
                        is_predicted=track["is_predicted"],
                        is_static=track["is_static"]
                    )

            # Draw detections (if enabled and tracks not available)
            if self.show_detections and frame_idx in self.detections and frame_idx not in self.tracks:
                for det in self.detections[frame_idx]:
                    self.draw_box(
                        frame,
                        det["x1"], det["y1"], det["x2"], det["y2"],
                        det["cls"], det["conf"],
                        is_static=det["is_static"]
                    )

            # Add frame info
            ts = frame_idx / self.fps
            info = f"Frame: {frame_idx}/{self.total_frames} | Time: {ts:.2f}s | FPS: {self.fps:.0f}"
            cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Write to output
            self.writer.write(frame)

            # Optional display
            if display:
                display_frame = cv2.resize(frame, (1280, 720))
                cv2.imshow("Visualization", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            processed += 1
            if processed % 100 == 0:
                print(f"  Processed {processed}/{self.total_frames} frames...")

        self.cleanup()
        print(f"✅ Visualization complete: {self.output_path}")

    def cleanup(self):
        """Release resources."""
        self.cap.release()
        self.writer.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize tracking and detection results on video",
        epilog="""
Examples:
  python visualize_results.py --video mac9.mp4 --tracks mac9.mp4_tracks.csv
  python visualize_results.py --video mac9.mp4 --tracks tracks.csv --output output.mp4
  python visualize_results.py --video mac9.mp4 --tracks tracks.csv --detections detections.csv --show-detections
        """
    )

    parser.add_argument("--video", type=str, required=True, help="Input video file")
    parser.add_argument("--tracks", type=str, required=True, help="Tracks CSV file")
    parser.add_argument("--detections", type=str, help="Detections CSV file (optional)")
    parser.add_argument("--output", type=str, help="Output video file (default: input_visualized.mp4)")
    parser.add_argument("--show-detections", action="store_true", help="Show detections if tracks not available")
    parser.add_argument("--display", action="store_true", help="Display video while processing (slower)")

    args = parser.parse_args()

    try:
        visualizer = VideoVisualizer(args.video, args.output, show_detections=args.show_detections)

        # Load data
        visualizer.load_tracks(args.tracks)
        if args.detections:
            visualizer.load_detections(args.detections)

        # Create visualization
        visualizer.visualize(display=args.display)

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
