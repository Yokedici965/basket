# Transcript Pipeline TODO (20.10.2025)

## Completed
- Added tools/transcribe_video.py and tools/build_transcript_segments.py; Whisper-based transcript -> segment automation is ready.
- Transcribed initial 120s (workout_sample) and 90s (scorer_sample) excerpts; JSON/SRT outputs live under data/event_samples/transcripts/.
- Produced 29 transcript-driven MP4 clips in data/event_samples/transcript_segments/.
- Ran tracking on 8 drive_angles_v2 clips and workout_sample_seg001.mp4; metrics stored in runs/phase1/.

## Open Items
1. **RunPod GPU pipeline**
   - Build and push Docker image with repo + dependencies (Python 3.12, FFmpeg, whisper, ultralytics).
   - Define batch scripts (`tools/batch_run_video.py`, optional `batch_transcribe.py`) for remote execution.
   - Set up storage sync (RunPod volume or S3/R2) for input videos and renders.
   - Create RunPod pod template (A40) with SSH + Jupyter enabled; document start/stop workflow.
2. **Transcript batches**
   - Inventory existing transcript clips (workout_full_*, scorer_full_*), confirm manifest entries, and mark problematic ones (`phase2_pending`).
   - Run batch tracking on remaining clips (frame_stride=1) via RunPod; archive overlays in renders/.
3. **QA & Reporting**
   - Update docs/reports/highlight_qc.md with new clip metrics (max_gap, coverage, notes).
   - Export summary CSV of metrics from runs/phase1 for quick reference.
4. **Test harness**
   - Add fixtures (short CSV + transcript) under tests/ for future heuristics regression.
5. **Handover**
   - Document RunPod usage, storage sync commands, and batch script parameters in docs/ or README appendix.
   - Record remaining local tasks (phase2 clips, config tweaks) in final delivery note.

## Notes
- Whisper runs in FP32 on CPU; use --model base plus GPU if speed becomes an issue.
- Manifest growth is fast: back it up regularly and keep subsets isolated via the context column.
