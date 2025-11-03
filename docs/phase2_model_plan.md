# Faz 2.2 - Pota Modeli ve Homografi Entegrasyonu Planý

## A) Veri Toplama & Etiketleme
- [x] 1. Kalibrasyon çýktýsýný görselleþtiren yarý-otomatik etiketleme aracý (frame + Hough sonucu + kullanýcý düzeltmesi).
- [ ] 2. Farklý salon/kamera kayýtlarýndan en az 500 etiketli kare topla; JSON/CSV formatýnda pota center + radius sakla.
- [ ] 3. Dataset versiyonlama (ör. `data/hoops/labels_v1/`).

## B) Model Eðitimi
- [x] 4. Lightweight YOLO/Keypoint pipeline hazýrlýðý (training script, augmentasyon, config).
- [ ] 5. Etiket setiyle eðitim + doðrulama; en az %90 tespit doðruluðu hedefi.
- [ ] 6. Eðitim sonrasý model export (`models/hoop_detector.pt`) ve inference wrapper.

## C) Homografi & Temporal Filtre
- [ ] 7. Stabilizasyon fazýný güncelle; homografi matrislerini tüm kareler için sakla (`output/homographies/*.npz`).
- [ ] 8. Kalibrasyon modülü: Hough + model sonuçlarýný birleþtir, Kalman/median filtre ile pota konumlarýný frame bazlý tahmin et.
- [ ] 9. Event motoru: homografi ve frame bazlý pota koordinatlarýyla çalýþacak þekilde güncelle (top/pota mesafesi, shot result).

## D) QC & UI
- [ ] 10. QC raporunu geniþlet: stabilizasyon hatasý, pota güven daðýlýmý, pota-standart sapmalarý.
- [ ] 11. Streamlit/FastAPI tabanlý arayüzle kullanýcýya kalibrasyon sonuçlarýný göster, manuel düzeltme (sürükle-býrak) imkâný ver.
- [ ] 12. Düzeltmeler dataset’e geri yazýlýp model güncelleme pipeline’ýna eklenir.

## E) Pipeline Entegrasyonu
- [ ] 13. `python -m app.calibration.hoops_cli` › stabilize + model + filtre + QC + config güncelleme.
- [ ] 14. `python -m app.run` ve `python -m app.events.cli` homografi/pota datalarýný kullanacak þekilde güncellenir.
- [ ] 15. Tüm akýþý mac2.mp4 ve en az bir ikinci sahada test edip doðruluk raporu çýkar.
