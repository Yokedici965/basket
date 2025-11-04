#!/usr/bin/env python3
"""
Batch processing script for pod-based video analysis.

Usage:
    python process_batch.py --video video.mp4                    # Single video
    python process_batch.py --dir /data/videos/                  # Process all .mp4 in directory
    python process_batch.py --dir /data/videos/ --pattern "mac*" # With pattern matching

Outputs:
    runs/run_TIMESTAMP/
    ├── metadata.json           # Processing configuration and timestamps
    ├── log.txt                 # Detailed processing log
    ├── video1_tracks.csv       # Track data
    ├── video1_detections.csv   # Detection data
    └── ...
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path
from datetime import datetime
import subprocess
import time

# Add app to path
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from app.run import load_config, process_video, get_device_kw


class BatchProcessor:
    """Manages batch video processing with logging and metadata tracking."""

    def __init__(self, output_base="runs"):
        self.output_base = Path(output_base)
        self.output_base.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_base / f"run_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.run_dir / "log.txt"
        self.metadata_file = self.run_dir / "metadata.json"
        self.results = []

    def log(self, message, level="INFO"):
        """Write message to log file and stdout."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] [{level}] {message}"
        print(log_msg)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")

    def save_metadata(self, config, videos, device_info):
        """Save processing metadata to JSON."""
        metadata = {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "timestamp_start": datetime.now().isoformat(),
            "videos_count": len(videos),
            "videos": [str(v) for v in videos],
            "device": device_info,
            "config_model": config.get("model"),
            "config_frame_stride": config.get("processing", {}).get("frame_stride"),
            "config_ball_threshold": config.get("detection", {}).get("thresholds", {}).get("ball"),
            "config_tracking_iou": config.get("tracking", {}).get("iou_threshold"),
            "results": []
        }
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        self.log(f"Metadata saved to {self.metadata_file}")

    def update_metadata_results(self, results):
        """Update metadata with completed results."""
        with open(self.metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        metadata["results"] = results
        metadata["timestamp_end"] = datetime.now().isoformat()
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def process_videos(self, video_paths, cfg, model_path, device_kw):
        """Process a list of videos."""
        self.log(f"Starting batch processing of {len(video_paths)} video(s)")
        self.log(f"Run directory: {self.run_dir}")

        for i, video_path in enumerate(video_paths, 1):
            video_path = Path(video_path)
            if not video_path.exists():
                self.log(f"SKIP {i}/{len(video_paths)}: {video_path.name} (file not found)", "WARN")
                self.results.append({
                    "video": str(video_path),
                    "status": "skipped",
                    "reason": "file not found"
                })
                continue

            self.log(f"PROCESS {i}/{len(video_paths)}: {video_path.name}")
            start_time = time.time()

            try:
                # Process video (outputs go to OUT_DIR)
                result = process_video(video_path, cfg, model_path, device_kw)
                elapsed = time.time() - start_time

                # Move outputs to run directory
                out_dir = BASE / "outputs"
                tracks_file = out_dir / f"{video_path.name}_tracks.csv"
                dets_file = out_dir / f"{video_path.name}_detections.csv"

                moved_tracks = None
                moved_dets = None

                if tracks_file.exists():
                    moved_tracks = self.run_dir / f"{video_path.stem}_tracks.csv"
                    tracks_file.rename(moved_tracks)
                    self.log(f"  ✓ Saved tracks → {moved_tracks.name}", "OK")

                if dets_file.exists():
                    moved_dets = self.run_dir / f"{video_path.stem}_detections.csv"
                    dets_file.rename(moved_dets)
                    self.log(f"  ✓ Saved detections → {moved_dets.name}", "OK")

                self.results.append({
                    "video": str(video_path),
                    "status": "completed",
                    "elapsed_seconds": elapsed,
                    "tracks_file": str(moved_tracks) if moved_tracks else None,
                    "detections_file": str(moved_dets) if moved_dets else None,
                    "static_ball_flagged": result.get("static_ball_flagged", 0) if result else 0
                })

                self.log(f"  DONE in {elapsed:.1f}s", "OK")

            except Exception as e:
                elapsed = time.time() - start_time
                self.log(f"ERROR {i}/{len(video_paths)}: {str(e)}", "ERROR")
                self.results.append({
                    "video": str(video_path),
                    "status": "error",
                    "elapsed_seconds": elapsed,
                    "error": str(e)
                })

        # Update metadata with final results
        self.update_metadata_results(self.results)
        self.log(f"Batch processing complete. Results in {self.run_dir}")

    def get_summary(self):
        """Return processing summary."""
        completed = sum(1 for r in self.results if r["status"] == "completed")
        failed = sum(1 for r in self.results if r["status"] == "error")
        skipped = sum(1 for r in self.results if r["status"] == "skipped")
        total_time = sum(r.get("elapsed_seconds", 0) for r in self.results)

        return {
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "total_videos": len(self.results),
            "total_seconds": total_time,
            "run_dir": str(self.run_dir),
            "metadata_file": str(self.metadata_file),
            "log_file": str(self.log_file)
        }


def find_videos(directory, pattern=None):
    """Find all .mp4 videos in directory, optionally matching pattern."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    if pattern:
        search_pattern = f"{directory}/{pattern}*.mp4"
        videos = sorted(glob.glob(search_pattern))
    else:
        videos = sorted(directory.glob("*.mp4"))

    return [Path(v) for v in videos]


def main():
    parser = argparse.ArgumentParser(
        description="Batch process basketball videos with YOLO tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_batch.py --video /data/mac9.mp4
  python process_batch.py --dir /data/videos/
  python process_batch.py --dir /data/videos/ --pattern "mac"
  python process_batch.py --dir /data/videos/ --output batch_runs/
        """
    )

    parser.add_argument("--video", type=str, help="Process single video file")
    parser.add_argument("--dir", type=str, help="Process all .mp4 files in directory")
    parser.add_argument("--pattern", type=str, help="Pattern for matching videos (e.g. 'mac' → mac*.mp4)")
    parser.add_argument("--output", type=str, default="runs", help="Output base directory (default: runs/)")

    args = parser.parse_args()

    # Determine videos to process
    videos = []
    if args.video:
        videos = [Path(args.video)]
    elif args.dir:
        videos = find_videos(args.dir, args.pattern)
    else:
        parser.print_help()
        sys.exit(1)

    if not videos:
        print("No videos found to process")
        sys.exit(1)

    # Load configuration
    cfg, model_path = load_config()
    print(f"[INFO] Config loaded: {model_path}")

    # Setup device
    detection_cfg = cfg.get("detection", {})
    use_gpu = bool(detection_cfg.get("use_gpu", True))
    half_precision = bool(detection_cfg.get("half_precision", True))
    device_kw = get_device_kw(use_gpu, half_precision)
    device_info = "GPU (CUDA)" if device_kw else "CPU"
    print(f"[INFO] Device: {device_info}")

    # Create processor and run batch
    processor = BatchProcessor(output_base=args.output)
    processor.save_metadata(cfg, videos, device_info)
    processor.process_videos(videos, cfg, model_path, device_kw)

    # Print summary
    summary = processor.get_summary()
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    print(f"Completed: {summary['completed']}/{summary['total_videos']}")
    print(f"Failed: {summary['failed']}")
    print(f"Skipped: {summary['skipped']}")
    print(f"Total Time: {summary['total_seconds']:.1f}s")
    print(f"Run Directory: {summary['run_dir']}")
    print("="*60)


if __name__ == "__main__":
    main()
