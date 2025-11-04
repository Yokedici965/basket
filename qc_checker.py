#!/usr/bin/env python3
"""
QC (Quality Control) validation for batch processing outputs.

Validates CSV outputs from process_batch.py and provides summary statistics:
- Track count and distribution
- Detection statistics
- Static ball filtering results
- Data quality issues

Usage:
    python qc_checker.py runs/run_20240115_120000/
    python qc_checker.py runs/run_20240115_120000/ --detailed
    python qc_checker.py runs/run_20240115_120000/ --export-report
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np


class QCChecker:
    """Quality control validation for batch processing outputs."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        self.metadata_file = self.run_dir / "metadata.json"
        self.log_file = self.run_dir / "log.txt"
        self.results = {}

    def load_metadata(self):
        """Load processing metadata."""
        if not self.metadata_file.exists():
            print(f"⚠️  No metadata.json found in {self.run_dir}")
            return None
        with open(self.metadata_file) as f:
            return json.load(f)

    def check_tracks_csv(self, tracks_file):
        """Analyze tracks.csv file."""
        if not tracks_file.exists():
            return {"status": "missing"}

        try:
            df = pd.read_csv(tracks_file)
            unique_track_ids = df["track_id"].nunique()
            unique_classes = df["cls"].unique()
            person_tracks = df[df["cls"] == 0]["track_id"].nunique()
            ball_tracks = df[df["cls"] == 32]["track_id"].nunique()
            total_rows = len(df)

            return {
                "status": "ok",
                "total_rows": total_rows,
                "unique_track_ids": unique_track_ids,
                "person_tracks": person_tracks,
                "ball_tracks": ball_tracks,
                "classes_detected": list(unique_classes),
                "confidence_stats": {
                    "min": float(df["conf"].min()),
                    "max": float(df["conf"].max()),
                    "mean": float(df["conf"].mean()),
                    "median": float(df["conf"].median()),
                },
                "age_stats": {
                    "min": int(df["age"].min()),
                    "max": int(df["age"].max()),
                    "mean": float(df["age"].mean()),
                },
                "static_flagged": int(df["is_static"].sum()),
                "predicted_count": int(df["is_predicted"].sum()),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def check_detections_csv(self, detections_file):
        """Analyze detections.csv file."""
        if not detections_file.exists():
            return {"status": "missing"}

        try:
            df = pd.read_csv(detections_file)
            unique_classes = df["cls"].unique()
            person_dets = len(df[df["cls"] == 0])
            ball_dets = len(df[df["cls"] == 32])
            total_rows = len(df)

            return {
                "status": "ok",
                "total_rows": total_rows,
                "person_detections": person_dets,
                "ball_detections": ball_dets,
                "classes_detected": list(unique_classes),
                "confidence_stats": {
                    "min": float(df["conf"].min()),
                    "max": float(df["conf"].max()),
                    "mean": float(df["conf"].mean()),
                    "median": float(df["conf"].median()),
                },
                "static_flagged": int(df["is_static"].sum()),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def validate_run(self):
        """Validate all outputs in the run directory."""
        metadata = self.load_metadata()

        print("\n" + "="*70)
        print("QC CHECK REPORT")
        print("="*70)
        print(f"Run: {self.run_dir.name}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Check metadata
        if metadata:
            print("PROCESSING CONFIGURATION")
            print("-" * 70)
            print(f"  Model: {metadata.get('config_model')}")
            print(f"  Frame Stride: {metadata.get('config_frame_stride')}")
            print(f"  Ball Threshold: {metadata.get('config_ball_threshold')}")
            print(f"  Tracking IOU: {metadata.get('config_tracking_iou')}")
            print(f"  Device: {metadata.get('device')}")
            print(f"  Videos Processed: {metadata.get('videos_count')}")
            print()

        # Check each result
        print("VIDEO RESULTS")
        print("-" * 70)

        video_summaries = []

        for result in metadata.get("results", []) if metadata else []:
            video_name = Path(result["video"]).name
            status = result.get("status", "unknown")

            if status == "completed":
                elapsed = result.get("elapsed_seconds", 0)
                tracks_file = Path(result.get("tracks_file", ""))
                dets_file = Path(result.get("detections_file", ""))

                print(f"\n✅ {video_name}")
                print(f"   Elapsed: {elapsed:.1f}s")

                # Check tracks
                if tracks_file.exists():
                    tracks_info = self.check_tracks_csv(tracks_file)
                    if tracks_info["status"] == "ok":
                        print(f"   Tracks CSV: {tracks_file.name}")
                        print(f"     • Total rows: {tracks_info['total_rows']:,}")
                        print(f"     • Unique track IDs: {tracks_info['unique_track_ids']}")
                        print(f"     • Person tracks: {tracks_info['person_tracks']}")
                        print(f"     • Ball tracks: {tracks_info['ball_tracks']}")
                        print(f"     • Conf: min={tracks_info['confidence_stats']['min']:.3f}, "
                              f"max={tracks_info['confidence_stats']['max']:.3f}, "
                              f"mean={tracks_info['confidence_stats']['mean']:.3f}")
                        print(f"     • Static ball: {tracks_info['static_flagged']} flagged")
                        print(f"     • Predicted: {tracks_info['predicted_count']}")

                        # QC checks
                        issues = []
                        if tracks_info['unique_track_ids'] > 30:
                            issues.append(f"⚠️  Track ID explosion: {tracks_info['unique_track_ids']} > 30")
                        if tracks_info['ball_tracks'] > 3:
                            issues.append(f"⚠️  Multiple ball track IDs: {tracks_info['ball_tracks']} > 1")
                        if tracks_info['confidence_stats']['mean'] < 0.5:
                            issues.append(f"⚠️  Low avg confidence: {tracks_info['confidence_stats']['mean']:.3f}")

                        if issues:
                            print("   Issues:")
                            for issue in issues:
                                print(f"     {issue}")

                        video_summaries.append({
                            "video": video_name,
                            "status": "ok",
                            "track_ids": tracks_info['unique_track_ids'],
                            "conf_mean": tracks_info['confidence_stats']['mean'],
                            "issues": len(issues)
                        })

                # Check detections
                if dets_file.exists():
                    dets_info = self.check_detections_csv(dets_file)
                    if dets_info["status"] == "ok":
                        print(f"   Detections CSV: {dets_file.name}")
                        print(f"     • Total rows: {dets_info['total_rows']:,}")
                        print(f"     • Person: {dets_info['person_detections']:,}")
                        print(f"     • Ball: {dets_info['ball_detections']:,}")
                        print(f"     • Conf: min={dets_info['confidence_stats']['min']:.3f}, "
                              f"max={dets_info['confidence_stats']['max']:.3f}, "
                              f"mean={dets_info['confidence_stats']['mean']:.3f}")

            elif status == "error":
                error = result.get("error", "unknown error")
                print(f"\n❌ {video_name}")
                print(f"   Error: {error}")
                video_summaries.append({
                    "video": video_name,
                    "status": "error",
                    "issues": 1
                })

            elif status == "skipped":
                reason = result.get("reason", "unknown")
                print(f"\n⊘  {video_name}")
                print(f"   Skipped: {reason}")

        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        completed = sum(1 for s in video_summaries if s["status"] == "ok")
        errors = sum(1 for s in video_summaries if s["status"] == "error")
        total_issues = sum(s.get("issues", 0) for s in video_summaries)

        print(f"Completed: {completed}")
        print(f"Errors: {errors}")
        print(f"Total Issues Found: {total_issues}")

        if video_summaries:
            avg_track_ids = np.mean([s.get("track_ids", 0) for s in video_summaries if s["status"] == "ok"])
            print(f"Avg Track IDs per video: {avg_track_ids:.1f}")

            avg_conf = np.mean([s.get("conf_mean", 0) for s in video_summaries if s["status"] == "ok"])
            print(f"Avg Confidence: {avg_conf:.3f}")

        print("="*70)
        print(f"Log: {self.log_file}")
        print(f"Metadata: {self.metadata_file}")
        print("="*70 + "\n")

        return {
            "completed": completed,
            "errors": errors,
            "total_issues": total_issues,
            "summaries": video_summaries
        }


def main():
    parser = argparse.ArgumentParser(
        description="QC validation for batch processing outputs",
        epilog="""
Examples:
  python qc_checker.py runs/run_20240115_120000/
  python qc_checker.py runs/run_20240115_120000/ --detailed
        """
    )

    parser.add_argument("run_dir", help="Path to run directory")
    parser.add_argument("--detailed", action="store_true", help="Show detailed analysis")
    parser.add_argument("--export-report", type=str, help="Export report to JSON file")

    args = parser.parse_args()

    try:
        checker = QCChecker(args.run_dir)
        results = checker.validate_run()

        if args.export_report:
            with open(args.export_report, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Report exported to: {args.export_report}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
