# Phase 1 - Gap Remediation TODO

## 1. Top Sürekliliği
- [x] `detection.thresholds.ball` değerini 0.12’ye çıkar.
- [x] `tracking.center_distance_threshold` değerini düşür (örn. 60.0).
- [x] Tracker öncesi hız tabanlı `static_ball` filtresi ekle; N kare hareketsiz kalan “ball” kayıtlarını işaretle.
- [ ] `tools/phase1_metrics` çıktısında `max_ball_gap_frames < 20` olduğunu doğrula (mevcut değer: 46).

## 2. Pipeline Otomasyonu & QC Kaydı
- [x] `tools/run_video.py` oluştur (run → metrics → overlay).
- [x] `docs/reports/highlight_qc.md` dosyasını şablonla.
- [x] CLI sonucunda QC loguna otomatik satır ekle (manüel not alanları bırak).

## 3. Test ve Bakım
- [x] Config doğrulaması için pytest (eksik poligon, hatalı class).
- [x] Overlay fonksiyonuna smoke test ekle (hoop/court çizimi beklenir).

## 4. Pota Drift İzleme
- [ ] `tools/update_hoops_from_qc.py` betiğini çoklu QC girdisini destekleyecek şekilde genişlet.
- [ ] Drift raporunu stdout/log’a yaz (önceki değer vs. yeni ortalama).
