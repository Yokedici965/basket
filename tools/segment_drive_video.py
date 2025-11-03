from __future__ import annotations

import argparse
import csv
import re
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, List

SCENE_RE = re.compile(r"pts_time:(?P<time>[0-9.]+)")


@dataclass
class Segment:
    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 0.0)


def run_cmd(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def get_duration(video_path: Path) -> float:
    result = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
    )
    return float(result.stdout.strip())


def get_fps(video_path: Path) -> float:
    result = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
    )
    rate = result.stdout.strip()
    if "/" in rate:
        return float(Fraction(rate))
    return float(rate)


def detect_scene_changes(video_path: Path, threshold: float) -> List[float]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(video_path),
        "-vf",
        f"select='gt(scene,{threshold})',showinfo",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    scenes: List[float] = []
    for line in proc.stderr.splitlines():
        match = SCENE_RE.search(line)
        if match:
            scenes.append(float(match.group("time")))
    return scenes


def build_segments(boundaries: Iterable[float], duration: float, min_seconds: float) -> List[Segment]:
    points = [0.0]
    for value in boundaries:
        if not points or value - points[-1] >= min_seconds:
            points.append(value)
    if duration - points[-1] >= min_seconds:
        points.append(duration)
    elif points[-1] != duration:
        points[-1] = duration

    segments: List[Segment] = []
    for idx in range(len(points) - 1):
        start, end = points[idx], points[idx + 1]
        if end - start < min_seconds:
            continue
        segments.append(Segment(index=idx + 1, start=start, end=end))
    return segments


def export_segment(video_path: Path, out_path: Path, segment: Segment, reencode: bool = False) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-ss",
        f"{segment.start:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{segment.duration:.3f}",
    ]
    if reencode:
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "23", "-c:a", "aac", "-b:a", "128k"]
    else:
        cmd += ["-c", "copy"]
    cmd.append(str(out_path))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-segment drive compilation videos by scene changes.")
    parser.add_argument("video", type=Path, help="Kaynak video yolu")
    parser.add_argument("--output-dir", type=Path, default=Path("data/event_samples/drive_segments"), help="Segmentlerin yazılacağı klasör")
    parser.add_argument("--threshold", type=float, default=0.32, help="FFmpeg scene change eşiği (varsayılan 0.32)")
    parser.add_argument("--min-seconds", type=float, default=3.0, help="Minimum segment süresi (saniye)")
    parser.add_argument("--prefix", type=str, default="drive_angles", help="Çıktı dosya adı ön eki")
    parser.add_argument("--no-export", action="store_true", help="Segment dosyalarını yazma, yalnızca sınırları raporla")
    parser.add_argument("--reencode", action="store_true", help="Segmentleri H.264/AAC olarak yeniden kodla (kesin zamanlama için)")
    parser.add_argument("--label", type=str, default="drive", help="Manifest label değeri")
    parser.add_argument("--context", type=str, default="drill_library", help="Manifest context değeri")
    parser.add_argument("--player-id", type=str, default="NA", help="Manifest player_id alanı")
    parser.add_argument("--notes", type=str, default="scene auto-split", help="Manifest notes alanı")
    parser.add_argument("--manifest", type=Path, help="Manifest CSV yolu (varsa append)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = args.video
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    duration = get_duration(video_path)
    fps = get_fps(video_path)
    print(f"[INFO] duration={duration:.2f}s | fps={fps:.2f}")

    scene_positions = detect_scene_changes(video_path, args.threshold)
    print(f"[INFO] detected {len(scene_positions)} scene boundaries over threshold {args.threshold}")
    segments = build_segments(scene_positions, duration, args.min_seconds)
    print(f"[INFO] derived {len(segments)} segments >= {args.min_seconds}s")

    if not args.no_export:
        for segment in segments:
            out_name = f"{args.prefix}_seg{segment.index:03d}.mp4"
            out_path = args.output_dir / out_name
            print(f"[WRITE] {out_path} | {segment.start:.3f}s -> {segment.end:.3f}s ({segment.duration:.2f}s)")
            export_segment(video_path, out_path, segment, reencode=args.reencode)
    else:
        for segment in segments:
            print(f"[SEG] #{segment.index:03d} {segment.start:.3f}s -> {segment.end:.3f}s ({segment.duration:.2f}s)")

    if args.manifest:
        manifest_entries = []
        for segment in segments:
            start_frame = int(round(segment.start * fps))
            end_frame = int(round(segment.end * fps))
            manifest_entries.append(
                [
                    f"{args.prefix}_seg{segment.index:03d}.mp4",
                    args.label,
                    args.context,
                    start_frame,
                    end_frame,
                    args.player_id,
                    args.notes,
                ]
            )
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerows(manifest_entries)
        print(f"[INFO] appended {len(manifest_entries)} rows to {args.manifest}")


if __name__ == "__main__":
    main()
