# RunPod Deployment Plan (Education Pipeline)

## Goals
- Hızlı overlay/render ve transkript işleri için GPU on-demand ortamı kurmak.
- Mevcut `tools/run_video.py`, `tools/transcribe_video.py` ve segment batch komutlarını buluta taşımak.

## Architecture
- **Compute**: RunPod on-demand pod, 1×A40 GPU (48 GB VRAM), 48 GB RAM, 9 vCPU. İmaj: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`.
- **Storage**: S3/R2 veya RunPod Volume; `data/event_samples/transcript_segments/`, `renders/`, `runs/phase1/` klasörlerini senkron tutacak.
- **Orchestration**: Deployment sırasında shell script / CLI ile `git clone`, `pip install -r requirements.txt`, `python tools/batch_run_video.py` vb.

## Docker Image & Setup
1. Dockerfile (öneri):
   ```Dockerfile
   FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
   RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*
   WORKDIR /workspace
   # Repo runtime sırasında clone edilecek; env değişkenleri ile konfig yöneteceğiz.
   ```
2. Pod başlatıldığında:
   ```bash
   git clone https://.../basket.git
   cd basket
   pip install -r requirements.txt openai-whisper rclone
   ```
3. Storage’a bağlan (ör. R2):
   ```bash
   rclone sync remote:bucket/transcript_segments data/event_samples/transcript_segments --fast-list
   ```

## Batch Scripts
- `tools/batch_run_video.py` (TODO):
  - Manifest veya dosya listesi alır, `python -m tools.run_video` komutunu sırayla çalıştırır.
  - Parametreler: `--list`, `--config`, `--frame-stride`, `--skip-overlay`.
- `tools/batch_transcribe.py` (opsiyonel):
  - Uzun videoları Whisper ile transkripte eder; segment scriptini tetikler.

## Execution Flow
1. Pod’u başlat (`runpodctl` veya web arayüzü). SSH veya VSCode remote ile bağlan.
2. Depoyu klonla, bağımlılıkları kur.
3. Girdi videolarını storage’dan çek.
4. Batch script ile render/transkript işle. Örn.:
   ```bash
   python tools/batch_run_video.py --list lists/scorer_clips.txt --frame-stride 1 --config configs/salon.yaml
   ```
5. Çıktıları storage’a geri yükle:
   ```bash
   rclone sync renders remote:bucket/renders --fast-list
   ```
6. Pod’u kapat (cost kontrol).

## Security & Config
- API anahtarlarını `.env` veya RunPod gizli değişkenlerinde tut.  
- `configs/salon.yaml` override gerekiyorsa çevresel değişken veya alternatif config dosyası kullan.
- SSH/Jupyter erişimini yalnızca ihtiyaç halinde aç.

## Monitoring & Logging
- `runs/phase1/*.json` dosyalarını toplu analiz için storage’a gönder.
- Gerekirse Slack/webhook ile status bildirimi ekle.
- Pipeline süresi ve GPU kullanımını RunPod dashboard üzerinden takip et.

## Next Steps
1. `tools/batch_run_video.py` ve opsiyonel `batch_transcribe.py` scriptlerini yaz.
2. Dockerfile’ı finalize edip registry’ye push et.
3. Rclone/S3 senkronizasyon komutlarını test et; manifest/overlay akışını doğrula.
4. Pod’u manuel başlatıp end-to-end deneme koşusu yap.
