# Hoop Auto-Detection Roadmap (mac7/mac8/mac9)

## Target Outcomes
- [x] Tek video girişinden pota konumu ve yarıçapını otomatik çıkarmak.
- [x] Çıkan koordinatları `configs/salon.yaml` ile senkronize edip inference/overlay aşamasında kullanmak.
- [x] Her maç için QC raporu (`configs/calibrations/*.json`) üretip doğruluk farklarını kaydetmek.

## Action Plan
1. **Örnek Çıktı Üretimi**
   - [x] `python -m app.calibration.hoops_cli --video mac9.MP4 --stride 5 --max-samples 30 --yolo-weights models/hoop_detector_n.pt`
   - [x] Aynı komutu mac7/mac8 için de çalıştır.
   - [x] `configs/calibrations/mac9.MP4_calibration_qc.json` içinde `aggregated` alanını doğrula.
2. **Config Senkronizasyonu**
   - [x] QC JSON’dan hoop merkezlerini alacak küçük yardımcı betik yaz (`tools/update_hoops_from_qc.py`).
   - [ ] Betik, hedef YAML içinde ilgili video profiline göre değerleri güncellesin; manuel override’ı loglasın (multi-QC ortalaması eklenecek).
3. **Overlay Doğrulaması**
   - [x] Güncellenen YAML ile `python -m tools.render_overlays --video mac9.MP4 --source tracks --keep-all-frames` çalıştır.
   - [x] Pota halkasını videoda doğru yerde görsel olarak doğrula; gerekirse `hoop_radius`/offset ayarla (`--hoop-radius 60`).
4. **Kalite Kayıtları**
   - [x] `docs/reports/phase1_mac9_summary.md` dosyasına otomatik pota tespiti sonuçlarını ekle.
   - [ ] Kıyaslama için hatalı/manuel koordinat logunu `docs/reports/hoop_calibration_notes.md` altında tut.

## Açık Sorular
- Hoop modeli için tek GPU/CPU konfigürasyonu yeterli mi, yoksa mac9 kadrajında ekstra augmentation gerekebilir mi?
- Sabit köşede duran top gibi durağan nesneler, hoop tespiti veya stabilizasyonu etkiliyor mu?
