# Faz 2.1 - Otomatik Pota Kalibrasyonu TODO

- [x] 1. Stabilizasyon modulu: SIFT/ORB + RANSAC homografi, guven skoru ve opsiyonel frame ornekleme.
- [x] 2. Pota aday tespiti: Hough + renk filtresi pipeline (HSV turuncu + morfoloji), homografi sonrasinda konumlandirma.
- [x] 3. Yedek derin model entegrasyonu: lightweight YOLO/Keypoint dedektoruyle pota adaylarini dogrulama.
- [x] 4. Temporal filtre/Kalman ile pota merkezlerinin zaman boyunca stabilizasyonu, kayip durumunda tahmin.
- [x] 5. Config guncelleme ve kalibrasyon raporu uretimi (`configs/calibrations/*.yaml`, `*_calibration_qc.json`).
- [ ] 6. Event motoruna homografi destekli pota koordinati besleme ve kisa test (mac2.mp4'ten kisa kesit).
- [ ] 7. QC uyari arayuzu: dusuk guvenli kalibrasyonlari kullaniciya gosterip manuel duzeltme olanagi saglama.
