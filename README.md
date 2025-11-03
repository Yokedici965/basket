# Basket Analytics Pipeline

Bu depo, amatör basketbol videolarını işleyip oyuncu/top takibi ve temel olay çıkarmak için kullanılan Python 3.12 tabanlı pipeline'ın hazır paketlenmiş halidir. YOLOv8 dedektörü, IOU + Kalman takipçisi ve heuristik event motoru tek komutla koşacak şekilde yapılandırıldı.

## Dizin Yapısı
- `app/` : Çekirdek kod; `run.py` video inference, `events/` possession/shot/rebound mantığı, `utils/` QC yardımcıları.
- `configs/` : Varsayılan ayarlar (`salon.yaml`) ve kalibrasyon çıktıları.
- `tools/` : CLI araçları (video koşturma, overlay render, metrikler, transcript segmentleri).
- `tests/` : Pytest tabanlı duman ve birim testleri.
- `docs/` : Yol haritaları, QC logları, deploy notları.

## Hızlı Başlangıç
1. Kurulum
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows için .venv\\Scripts\\activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
   Ek araçlar: `ffmpeg`, `whisper`, `git-lfs` (opsiyonel transcript/video işlemleri için).

2. Model & Veri
   - YOLO ağırlığını `models/yolov8x.pt` olarak indir (Ultralytics hub veya yerel arşiv).
   - Videoları `input_videos/` altına bırak.

3. Pipeline Çalıştır
   ```bash
   python tools/run_video.py --video input_videos/mac7_first3min.mp4 --frame-stride 2 --run-events
   ```
   Çıktılar `outputs/` altında (detections/tracks CSV, possession/shot/rebound dosyaları, `*_events_qc.json`). Overlay videosu gerekiyorsa `--skip-overlay` bayrağını kaldır.

4. Test
   ```bash
   pytest
   ```
   Event CLI duman testi ve QC yardımcı testleri çalışır.

## Hedefler & Yol Haritası
- **Kısa Vadede**: Top takip boşluklarını (`max_ball_gap < 20`) azaltmak, statik top flag'lerini event/highlight katmanından hariç tutmak.
- **Orta Vadede**: Highlight seçici ve takım analitiği (tempo, run, Blob/SLOB) üretmek; RunPod/TensorDock GPU dağıtımı ile otomasyonu tamamlamak.
- **Uzun Vadede**: Özel modeller (PnR, press break), SaaS paneli ve sürekli öğrenme döngüsü.

Bu paket GitHub deposuna doğrudan kopyalanıp `git init` ile versiyonlanabilir. Daha fazla ayrıntı için `docs/reports/` ve `docs/roadmap.md` dosyalarına göz atabilirsin.
