# Phase 1 Durum Raporu (2025-10-17)

## Genel Durum
- Takip pipeline'ı `app/run.py` içinde YAML doğrulaması, statik top filtresi ve Kalman destekli ID izleme ile çalışıyor.
- Hoop kalibrasyonu otomatikleşti; mac7/mac8/mac9 ortalamaları `configs/salon.yaml` içinde.
- `tools/run_video.py` tek komutla inference → metrik → overlay (opsiyonel events) akışını çalıştırıyor ve QC loguna kayıt ekliyor.
- Test paketi (`pytest`) config, QC metrikleri ve overlay bileşenlerinin şemasını doğruluyor.

## Son Koşu (mac9.MP4, 60 sn, stride=1)
- Komut: `python -m tools.run_video --video mac9.MP4 --frame-stride 1 --duration-sec 60`
- Metrikler (`runs/phase1/mac9.MP4_metrics.json`):
  - Player tracks: 138 | Ball tracks: 250 | Ball coverage: 69.27 %
  - Max ball gap (tahminler dahil): 46 kare | Statik top işaretleri: 544
- Overlay: `renders/mac9_tracks_overlay.mp4` (4K, court + hoop + ID etiketleri)
- QC tablosu (`docs/reports/highlight_qc.md`) güncellendi.

## Eksikler / Riskler
1. **Top sürekliliği**: Kalman + max_age=60 uygulanmasına rağmen gap 46 kare → ByteTrack/Kalman + optik akış gibi daha güçlü çözümler denenmeli.
2. **Statik top işaretleri**: Filtre işaretliyor, fakat olay motorunda hâlâ temizlenmedi; downstream’de k rules eklenmeli.
3. **Olay motoru**: Shot/possession çıkarımı hâlâ bağlanmadı; highlight kategorileri için temel veri akışı eksik.
4. **Pota drift raporu**: `tools/update_hoops_from_qc.py` çoklu QC ortalaması ve drift uyarısı üretmiyor.

## Yapılacaklar (Kısa Vadede)
- `max_ball_gap_frames < 20` hedefi için daha gelişmiş takip (ByteTrack + Kalman, optik akış stabilizasyonu) araştır.
- Statik top etiketlerini olay motorunda yok sayacak/raporlayacak kural ekle.
- Faz 2 için drive/pull-up/spot-up/assist etiketli mini veri seti oluştur.
- `tools/update_hoops_from_qc.py`’yi birden fazla QC dosyası ortalayıp drift raporu verecek şekilde genişlet.
- Olay motorunu (possession + shot) pipeline'a bağlayacak ilk versiyonunu hazırlayıp `tools/run_video.py` içinde opsiyonel çalıştır.

## Uzun Vadeli Notlar
- RunPod / Serverless GPU mimarisi için Docker + handler taslağı hazırlandığında Phase 5’e geçiş hızlanacak.
- Highlight MVP kategorileri: drive, pull-up, spot-up, assist, fast break, Blob/SLOB.
- QC süreçleri için `docs/reports/highlight_qc.md` tablosu her koşuda güncel tutulmalı.
