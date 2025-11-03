# Roadmap

## Phase 0 - Baseline and Repository Audit
- [ ] Run the current pipeline (`app/run.py`, `app/events/cli.py`) on mac7/8/9 videos to produce reference CSV and QC outputs.
- [ ] Document detection and speed metrics (FPS, inference time, ball/player recall) and publish a baseline benchmark report.
- [ ] Catalogue camera scenarios (static, moving, low light, different resolutions) to build a data source profile.
- [ ] Decide on storage strategy: temporary (`outputs/`, `renders/`) versus persistent (S3/MinIO) directories and access policies.

## Phase 1 - Detection and Tracking Reliability
Goal: reliable player/ball tracking and hoop reference across scenarios.
- [x] Automate hoop calibration (`app.calibration.hoops_cli`) and sync results into `configs/salon.yaml` (mac7/8/9 averages).
- [x] Add `tools/phase1_metrics.py` + pytest coverage for schema/metrics; archive runs under `runs/phase1/`.
- [x] Upgrade overlay renderer with court outline, hoop anchors, player/ball ID labels (ref: `renders/mac9_tracks_overlay_v2.mp4`).
- [ ] Add ball static-object filter (speed-based) and stride-aware tracker tuning (reduce long ball gaps).
- [ ] Plan YOLOv8l/x fine-tune for basketball use; define dataset needs, augmentation strategy, and training calendar.
- [ ] Integrate ByteTrack + Kalman (separate tuning for ball and players) and refactor `app/run.py`.
- [ ] Enable moving-camera support by inserting ORB + RANSAC stabilisation into the pipeline and measuring quality impact.
- [ ] Improve referee/bench/crowd filtering; validate saturation + corridor heuristics and add a lightweight classifier if required.
- [ ] GPU target: achieve < 3h end-to-end inference for 4K/30 on a single RTX 4090 and capture the benchmark report.

## Phase 2 - Event Semantics and Player Highlights
Goal: automatically derive player level highlight categories.
- [ ] Collect labelled clips (drive-to-hoop, pull-up jumper, spot-up, assist, PnR/PnP) – ≥20 segments per class.
- [ ] Extend event pipeline with possession segmentation, shot outcomes (2P/3P/FT/and-one), assist attribution.
- [ ] Build shot classification (2P, 3P, FT, and-one) by combining hoop ROI, ball trajectory, and foul indicators.
- [ ] Implement offence highlights: isolation, crossover/spin, pick and roll/pop, creative assists, fast breaks, put-backs.
- [ ] Implement drive detection (inbound control → dribble vector → paint entry → finish/kick-out classification).
- [ ] Implement defence highlights: steals, blocks, 1v1 stops, help contests, transition defence stops.
- [ ] Implement hustle plays: loose ball dives, charges, critical rebounds, deflections; design heuristic + human review labelling.
- [ ] Extend the event data structure with category, player id, team id, timestamp, confidence, source CSV reference.
- [ ] Stand up an annotation workflow: automatic pre-label + GUI review (extend existing tools) + expert sample plan.
- [ ] Log QC findings per run (`docs/reports/highlight_qc.md`) and adjust thresholds/models iteratively.

## Phase 3 - Team Tactics and Sequence Analytics
Goal: generate team level set and organisation highlights.
- [ ] Detect press/press-break sequences using court occupancy + turnover metrics.
- [ ] Detect transition vs half-court possessions; annotate success/failure (offence & defence).
- [ ] Detect pick and roll/pop, dribble handoff, off-ball screen, stagger, ghost screen sequences via player motion analysis.
- [ ] Detect Blob/SLOB (baseline/sideline out-of-bounds) via ball entry events and immediate scoring/assist outcomes.
- [ ] Characterise defensive schemes (man, zone, press, trap) through spatial occupancy and rotation heuristics.
- [ ] Recognise press break and inbound sets: entry events, pass sequences, and scoring outcomes.
- [ ] Reconstruct scoreboard flow to extract clutch and momentum runs (8-0 streaks, final two minutes scenarios).
- [ ] Define "Top 3 Plays" ranking logic using event value and context (clutch, and-one, momentum shift).

## Phase 4 - Highlight Packaging and Media Outputs
Goal: deliver highlight clips with 95% precision.
- [ ] Build highlight picker MVP prioritising: drive, pull-up, spot-up, assist, fast break, Blob/SLOB.
- [ ] Build clip generator with configurable pre/post windows and multi-format rendering (16:9, 9:16).
- [ ] Expand `tools/render_overlays.py` to add scoreboard, stats, and textual overlays.
- [ ] Define JSON manifest schema: category, player, team, score context, confidence, file references for API/panel.
- [ ] Create social media presets (reels, top-10, player mix) and automatic draft export.
- [ ] Implement QC workflow: highlight validation tool and dashboard tracking precision toward the 95% target.

## Phase 5 - Platform, API, and GPU Operations
Goal: automate the Vast.ai on-demand GPU workflow and expose a management panel.
- [ ] Prepare RunPod Docker base (`FROM runpod/pytorch:2.2.0-cuda12.1`) with YOLOv8 + ByteTrack + ffmpeg + yt-dlp stack.
- [ ] Implement serverless handler (`handler.py`) to download (yt-dlp/S3), run pipeline, upload outputs (S3/R2), clean `/tmp`.
- [ ] Configure NAS sync via `rclone` (cron) and storage lifecycle policies (raw vs manifest retention).
- [ ] Prepare Docker/conda base image (PyTorch 2.3+, CUDA 12, ffmpeg, rclone).
- [ ] Integrate Vast.ai API: instance launch, persistent volume mount, job scheduling, automatic shutdown.
- [ ] Develop FastAPI backend for video upload, job queueing, and configuration overrides (thresholds, models, formats).
- [ ] Build web panel (Streamlit or React) with coach/analyst/player roles, highlight filtering, clip preview, download, share.
- [ ] Connect to S3-compatible storage for manifest + clip + CSV archiving with lifecycle policies.
- [ ] Add monitoring and logging: GPU usage, inference timing, error alerts, and cost tracking.

## Phase 6 - Quality, Data Expansion, and Continuous Training
Goal: sustain >95% accuracy and prepare for future features.
- [ ] Establish precision/recall tracking for each highlight category against curated ground truth.
- [ ] Automate hoop QC aggregation (`tools/update_hoops_from_qc.py` multi-run averaging) and drift alerts.
- [ ] Add active learning loop that routes low-confidence or QC-flagged events to the annotation queue.
- [ ] Schedule periodic retraining for ball/hoop/gesture detectors, tactical classifiers, and pose-based foul models.
- [ ] Support multi-team and multi-season usage: tenant identifiers, IAM, data isolation, role-based access.
- [ ] Automate reporting: match summaries (PDF/HTML), QC status, cost and duration analysis.
- [ ] Review roadmap regularly with user feedback, new highlight requests, and release notes.

---

**Dependencies and Notes**
- Phase 2 and beyond rely on the reliable tracking and calibration delivered in Phase 1.
- Annotation resources (internal or external) should be confirmed by the end of Phase 0 and available for Phase 2.
- GPU benchmark and cost outputs (Phase 1 and 5) should be archived under `docs/reports/`.
- Default social media presets (YouTube 16:9, TikTok 9:16) will be provided in the panel, with user overrides supported.
