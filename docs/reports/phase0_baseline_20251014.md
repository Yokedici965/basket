# Phase 0 Baseline Report (2025-10-14)

## Baseline Performance

| video    |   processing_seconds |   effective_fps |   detections_player |   tracks_player |   player_frame_coverage |   player_precision_proxy |   detections_ball |   tracks_ball |   ball_frame_coverage |   ball_precision_proxy |   referee_track_ids |   lost_track_ids |
|:---------|---------------------:|----------------:|--------------------:|----------------:|------------------------:|-------------------------:|------------------:|--------------:|----------------------:|-----------------------:|--------------------:|-----------------:|
| mac7.MP4 |                701.7 |           0.929 |                  66 |              66 |                    10.1 |                      100 |               676 |           676 |                  93.3 |                    100 |                   0 |               44 |
| mac8.MP4 |                674.4 |           0.816 |                  80 |              80 |                    14.4 |                      100 |               516 |           516 |                  88.7 |                    100 |                   0 |               45 |
| mac9.MP4 |                181.2 |           0.872 |                   9 |               9 |                     5.7 |                      100 |               174 |           174 |                  95.6 |                    100 |                   0 |                7 |

- Player frame coverage proxy = tespit içeren frame sayısının toplam örnek frame'lere oranı.
- Precision proxy = track/detection oranı (ground-truth olmadığı için karşılaştırma yoktur).
- Kayıp tracking ID sayıları 3 kare veya daha kısa ömürlü IDleri temsil eder.

## Quality Issues

### mac7.MP4

| category    |   frame |   timestamp_sec | detail              |
|:------------|--------:|----------------:|:--------------------|
| camera_move |     798 |           31.92 | delta_cx=475.6      |
| camera_move |    1104 |           44.16 | delta_cx=613.1      |
| camera_move |    1656 |           66.24 | delta_cx=600.0      |
| lighting    |    5304 |          212.16 | avg_conf=0.151      |
| lighting    |    6948 |          277.92 | avg_conf=0.151      |
| lighting    |    7185 |          287.4  | avg_conf=0.150      |
| occlusion   |     397 |           15.88 | ball_gap_frames=53  |
| occlusion   |     565 |           22.6  | ball_gap_frames=218 |
| occlusion   |     829 |           33.16 | ball_gap_frames=80  |

### mac8.MP4

| category    |   frame |   timestamp_sec | detail              |
|:------------|--------:|----------------:|:--------------------|
| camera_move |    1269 |           50.76 | delta_cx=478.6      |
| camera_move |    1347 |           53.88 | delta_cx=911.5      |
| camera_move |    1419 |           56.76 | delta_cx=1001.8     |
| lighting    |    2814 |          112.56 | avg_conf=0.152      |
| lighting    |    2847 |          113.88 | avg_conf=0.152      |
| lighting    |    4344 |          173.76 | avg_conf=0.151      |
| occlusion   |      52 |            2.08 | ball_gap_frames=107 |
| occlusion   |     178 |            7.12 | ball_gap_frames=80  |
| occlusion   |     259 |           10.36 | ball_gap_frames=89  |

### mac9.MP4

| category    |   frame |   timestamp_sec | detail              |
|:------------|--------:|----------------:|:--------------------|
| camera_move |     402 |           16.08 | delta_cx=1180.2     |
| camera_move |     633 |           25.32 | delta_cx=1072.8     |
| camera_move |    1107 |           44.28 | delta_cx=642.6      |
| lighting    |    1665 |           66.6  | avg_conf=0.152      |
| lighting    |    1827 |           73.08 | avg_conf=0.152      |
| lighting    |    1830 |           73.2  | avg_conf=0.152      |
| occlusion   |      19 |            0.76 | ball_gap_frames=89  |
| occlusion   |     115 |            4.6  | ball_gap_frames=161 |
| occlusion   |     472 |           18.88 | ball_gap_frames=56  |

## Storage Audit

**Video Dosyaları**

| video    |   size_gb |
|:---------|----------:|
| mac7.MP4 |     3.975 |
| mac8.MP4 |     3.976 |
| mac9.MP4 |     0.546 |

**Çıktı Artifaktları**

| artifact                 |   size_mb |
|:-------------------------|----------:|
| mac7.mp4_detections.csv  |     0.093 |
| mac7.mp4_possessions.csv |     0     |
| mac7.mp4_rebounds.csv    |     0     |
| mac7.mp4_shots.csv       |     0     |
| mac7.mp4_tracks.csv      |     0.121 |
| mac8.mp4_detections.csv  |     0.075 |
| mac8.mp4_tracks.csv      |     0.098 |
| mac9.mp4_detections.csv  |     0.023 |
| mac9.mp4_tracks.csv      |     0.03  |

- Toplam video boyutu: 8.497 GB
- Çıktı artifaktları: 0.0 GB
- Toplam storage: 8.497 GB → S3 Standard aylık ≈ $0.195

## Next Bottlenecks Before Phase 1

- Saha poligonu 1920x1080'da kaldığı için oyuncu tespiti %10-14 kapsama seviyesinde; 4K ROI güncellemesi ve model hassasiyeti gerekiyor.
- CPU inference ~0.8-0.9 FPS (10-12 dk / 5 dk video); RTX 4090 veya stride/model ayarı olmadan faz hedefleri gerçekleştiremeyiz.
- Top/oyuncu track ID'leri 7-45 arası kısa ömürlü kayıplarla bölünüyor; ByteTrack/Kalman olmadan highlight sürekliliği sağlanamaz.

## Artifacts

- Metrics CSV: `outputs/phase0_metrics_20251014.csv`
- Quality flags & storage JSON: `outputs/phase0_quality_20251014/`
- Pipeline config: `configs/salon.yaml` (model=`yolov8s.pt`, frame_stride=3, duration=300s).

## Observations

- Ball detections kapsama oranı %88-96 iken oyuncu kapsaması %5-14; ROI ve model tuning olmadan oyuncu bazlı highlight mümkün değil.
- Aydınlatma güven skorları 0.15 civarına düştüğü frameler top occlusion segmentleriyle örtüşüyor; bu kesitler Phase 2 heuristiklerini etkiliyor.
- Event CSVleri boş kaldı; top tracking mevcut olsa da oyuncu verisi yetersiz olduğu için event motoru tetiklenmedi.