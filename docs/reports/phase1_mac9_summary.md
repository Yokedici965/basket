# Phase 1 Tracking Validation — mac9.MP4 (2025-10-17)

## Run Inputs
- **Command (stride=2 sample):** `python -c "...process_video(Path('input_videos/mac9.MP4'), ...)"` with frame_stride=2, duration_sec=120 (overrides applied in-line).
- **Command (stride=1 review clip):** `python -m tools.run_video --video mac9.MP4 --frame-stride 1 --duration-sec 60`.
- **Config:** `configs/salon.yaml` (statik top filtresi + Kalman destekli tracker; hakem tespiti kaldırıldı).
- **Model:** `yolov8l.pt`
- **Video:** `input_videos/mac9.MP4` (1,908 frames, 25 FPS)
- **Çıktılar:**
  - Tracks: `outputs/mac9.MP4_tracks.csv`
  - Detections: `outputs/mac9.MP4_detections.csv`
  - Metrics: `runs/phase1/mac9_stride2_120s/metrics.json`, `runs/phase1/mac9.MP4_metrics.json`
  - Overlay: `renders/mac9_tracks_overlay.mp4`

## Key Metrics (tool.run_video, stride=1, 60 s)
| metric | değer |
| --- | --- |
| Frames processed | 1,500 |
| Player tracks | 138 |
| Ball tracks | 250 |
| Ball frame coverage | 69.27 % |
| Ball mean confidence | 0.356 |
| Short-lived tracks (<3 frame) | 30 |
| Max ball gap (predictions dâhil) | 46 frames |
| Static ball flagged | 544 |

## Observations
- Kalman filtresi + `max_age=60` ile ID sürekliliği arttı, ancak top boşluğu hâlen 46 kare → daha güçlü takip (ByteTrack/optik akış) gerekiyor.
- Hakem tespiti devre dışı; CSV'ler sadece oyuncu ve top izlerini içeriyor.
- Statik top işaretleri `is_static` kolonu ile işaretleniyor, event ve highlight katmanında filtrelenmeli.
- Overlay 4K formatında; sahayı, hoop noktalarını ve ID etiketlerini çiziyor.

## Validation Artifacts
- `pytest` (config + QC + overlay smoke) → `tests/test_config_validation.py`, `tests/test_qc_metrics.py`, `tests/test_render_overlays.py`.
- `tools/run_video.py` inference → metrics → overlay otomasyon komutu; QC tablosunu (`docs/reports/highlight_qc.md`) güncelliyor.

## Next Steps
1. Top sürekliliği için ByteTrack/Kalman + optik akış gibi ileri seviye takip yöntemlerini dene (`max_ball_gap < 20`).
2. Statik top işaretlerini olay motoru ve highlight seçiminde filtrele / raporla.
3. Faz 2 için drive/pull-up/spot-up/assist etiketli veriseti çıkar ve heuristik geliştirmeye başla.
4. `tools/update_hoops_from_qc.py` çekildiğinde drift raporu üretecek şekilde genişlet.
