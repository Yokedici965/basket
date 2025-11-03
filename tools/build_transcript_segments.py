from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class TranscriptSegment:
    index: int
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 0.0)


def load_transcript(path: Path) -> List[TranscriptSegment]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = []
    for idx, seg in enumerate(payload.get("segments", []), start=1):
        text = re.sub(r"\s+", " ", seg.get("text", "")).strip()
        segments.append(
            TranscriptSegment(
                index=idx,
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", 0.0)),
                text=text,
            )
        )
    return segments


def filter_segments(
    segments: Iterable[TranscriptSegment],
    min_duration: float,
    max_duration: float,
    min_words: int,
) -> List[TranscriptSegment]:
    filtered: List[TranscriptSegment] = []
    for seg in segments:
        if seg.duration < min_duration:
            continue
        if max_duration > 0 and seg.duration > max_duration:
            continue
        if len(seg.text.split()) < min_words:
            continue
        filtered.append(seg)
    return filtered


def export_clip(
    video_path: Path,
    out_path: Path,
    seg: TranscriptSegment,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-ss",
        f"{seg.start:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{seg.duration:.3f}",
        "-c",
        "copy",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def append_manifest(
    manifest_path: Path,
    rows: List[List[str]],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manifest rows and optional clips from a Whisper transcript.")
    parser.add_argument("--video", type=Path, required=True, help="Kaynak video dosyası")
    parser.add_argument("--transcript", type=Path, required=True, help="Whisper JSON çıktısı")
    parser.add_argument("--output-dir", type=Path, default=Path("data/event_samples/transcript_segments"), help="Segment klipleri dizini")
    parser.add_argument("--prefix", type=str, default=None, help="Çıktı dosya adları için ön ek (varsayılan: transcript dosya adı)")
    parser.add_argument("--manifest", type=Path, default=Path("data/event_samples/shared_metadata/event_samples_manifest.csv"), help="Manifest CSV yolu")
    parser.add_argument("--label", type=str, default="drive", help="Manifest label değeri")
    parser.add_argument("--context", type=str, default="training_transcript", help="Manifest context değeri")
    parser.add_argument("--player-id", type=str, default="NA", help="Manifest player_id alanı")
    parser.add_argument("--notes-prefix", type=str, default="", help="Manifest notes alanı için önek")
    parser.add_argument("--min-duration", type=float, default=2.0, help="Segment minimum süresi (saniye)")
    parser.add_argument("--max-duration", type=float, default=0.0, help="Segment maksimum süresi (<=0 sınırsız)")
    parser.add_argument("--min-words", type=int, default=3, help="Segment minimum kelime sayısı")
    parser.add_argument("--export-clips", action="store_true", help="Segment başına MP4 oluştur")
    parser.add_argument("--fps", type=float, default=None, help="Manifest kare hesaplaması için fps (video üzerinden otomatik belirlenir)")
    return parser.parse_args()


def get_fps(video_path: Path) -> float:
    cmd = [
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
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    rate = result.stdout.strip()
    if "/" in rate:
        num, den = rate.split("/")
        return float(num) / float(den)
    return float(rate)


def main() -> None:
    args = parse_args()
    if not args.video.exists():
        raise FileNotFoundError(args.video)
    if not args.transcript.exists():
        raise FileNotFoundError(args.transcript)

    prefix = args.prefix or args.transcript.stem
    segments = load_transcript(args.transcript)
    segments = filter_segments(segments, args.min_duration, args.max_duration, args.min_words)
    if not segments:
        print("[WARN] No segments passed the filters.")
        return

    fps = args.fps or get_fps(args.video)

    manifest_rows: List[List[str]] = []
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for seg in segments:
        clip_name = f"{prefix}_seg{seg.index:03d}.mp4"
        clip_path = output_dir / clip_name
        if args.export_clips:
            export_clip(args.video, clip_path, seg)
        start_frame = int(math.floor(seg.start * fps))
        end_frame = int(math.ceil(seg.end * fps))
        note_text = seg.text
        if args.notes_prefix:
            note_text = f"{args.notes_prefix} | {note_text}"
        manifest_rows.append(
            [
                clip_name if args.export_clips else args.video.name,
                args.label,
                args.context,
                start_frame,
                end_frame,
                args.player_id,
                note_text,
            ]
        )

    append_manifest(args.manifest, manifest_rows)
    print(f"[OK] appended {len(manifest_rows)} rows -> {args.manifest}")
    if args.export_clips:
        print(f"[OK] exported clips -> {output_dir}")


if __name__ == "__main__":
    main()
