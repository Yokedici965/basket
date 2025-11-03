from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def extract_audio(
    video_path: Path,
    audio_path: Path,
    start_sec: Optional[float],
    duration_sec: Optional[float],
    sample_rate: int,
) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-y"]
    if start_sec is not None:
        cmd += ["-ss", f"{start_sec}"]
    cmd += ["-i", str(video_path)]
    if duration_sec is not None:
        cmd += ["-t", f"{duration_sec}"]
    cmd += ["-ac", "1", "-ar", str(sample_rate), str(audio_path)]
    subprocess.run(cmd, check=True)


def load_whisper(model_name: str):
    try:
        import whisper  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "whisper package is required. Install with:\n"
            "  pip install openai-whisper\n"
            "  (Optionally add torch with CUDA support beforehand)"
        ) from exc
    return whisper.load_model(model_name)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_srt(transcript: dict, srt_path: Path) -> None:
    try:
        from whisper.utils import write_srt  # type: ignore
    except ImportError:  # pragma: no cover - dependency guard
        return
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    with srt_path.open("w", encoding="utf-8") as fh:
        write_srt(transcript["segments"], fh)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe a video with Whisper and save JSON/SRT outputs.")
    parser.add_argument("--video", type=Path, required=True, help="Video file path")
    parser.add_argument("--model", default="small", help="Whisper model name (default: small)")
    parser.add_argument("--language", default=None, help="Force language (e.g. en, tr). Auto-detect if omitted.")
    parser.add_argument("--start-sec", type=float, default=None, help="Optional start time for partial transcription")
    parser.add_argument("--duration-sec", type=float, default=None, help="Optional duration for partial transcription")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate for ffmpeg downmix")
    parser.add_argument("--output-dir", type=Path, default=Path("data/event_samples/transcripts"), help="Output directory")
    parser.add_argument("--prefix", type=str, default=None, help="Override output filename prefix")
    parser.add_argument("--word-timestamps", action="store_true", help="Include word-level timestamps in transcript")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path: Path = args.video
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    prefix = args.prefix or video_path.stem
    output_dir: Path = args.output_dir

    audio_dir = output_dir / "audio_cache"
    audio_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(prefix="whisper_", suffix=".wav", dir=audio_dir, delete=False) as tmp:
        audio_path = Path(tmp.name)
    try:
        extract_audio(video_path, audio_path, args.start_sec, args.duration_sec, args.sample_rate)
        model = load_whisper(args.model)
        options = dict(task="transcribe")
        if args.language:
            options["language"] = args.language
        if args.word_timestamps:
            options["word_timestamps"] = True
        result = model.transcribe(str(audio_path), **options)

        json_path = output_dir / f"{prefix}.json"
        write_json(json_path, result)

        srt_path = output_dir / f"{prefix}.srt"
        write_srt(result, srt_path)

        print(f"[OK] transcript saved -> {json_path}")
        print(f"[OK] subtitles saved  -> {srt_path}")
    finally:
        if audio_path.exists():
            audio_path.unlink()


if __name__ == "__main__":
    main()
