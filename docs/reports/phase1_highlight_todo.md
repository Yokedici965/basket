# Highlight Phase – TODO Checklist (mac9 Baseline)

## 1. Detection & Tracking (Completed)
- [x] Validate `configs/salon.yaml` schema (court polygon, hoop coords, detection thresholds).
- [x] Run YOLOv8l inference via `python app/run.py` (stride configurable) and archive logs/metrics (`runs/phase1/...`).
- [x] Generate QC metrics with `python -m tools.phase1_metrics --tracks ... --detections ...`.

## 2. Event Engine Prep
- [ ] Re-run event CLI on mac9 once possession/shot heuristics are tuned:  
      `python -m app.events.cli --video mac9.MP4`.
- [ ] Review `outputs/mac9.MP4_events_qc.json` for shot/possession accuracy targets (≥90 % shot recall, ≥85 % possession).
- [ ] Flag anomalous ranges (ball gap ≥60 frames) and feed back into threshold tuning / manual labelling.

## 3. Highlight Feature Extraction
- [ ] Define highlight labels mapped to hedefler.txt categories (e.g., `shot_made`, `block`, `fast_break`).
- [ ] Store per-event metadata (start/end timestamp, track IDs, hoop proximity) in `events.csv`.
- [ ] Create selection heuristics (confidence, score impact, uniqueness) for mac9 sample.

## 4. Overlay & Visualization
- [x] Update `tools/render_overlays.py` to draw hoop anchors and track IDs for review (mac9 run).
- [ ] Extend renderer to highlight chosen events: colour-code highlight frames, add metadata banner.
- [ ] Export 4K review MP4 for every highlight candidate; archive under `renders/highlights/`.

## 5. QC & Feedback Loop
- [ ] Conduct manual review of rendered highlights; document feedback in `docs/reports/highlight_qc.md`.
- [ ] Iterate thresholds/models based on QC (adjust YAML, re-run pipeline).
- [ ] Once targets met, publish mac9 highlight summary (CSV + MP4) and declare Phase 1 highlight readiness.

