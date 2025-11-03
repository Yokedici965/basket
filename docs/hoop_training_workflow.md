# Hoop Model Eðitim Akýþý

## 1. Örnek Kareleri Üret
```
python -m app.calibration.export_samples --video mac2.mp4 --stride 5 --max-samples 12
```
- `configs/calibrations/<video>_samples/raw` klasörüne ham kareler,
- `overlay/` klasörüne Hough iþaretli görseller,
- `meta/` klasörüne homografi ve tespit bilgileri kaydedilir.

## 2. Manuel Etiketleme (Hýzlý Onay/Düzeltme)
```
python -m app.labeling.hoop_labeler --video mac2
```
- Yön tuþlarý/WASD ile x-y konumu, `+/-` ile yarýçap ayarla.
- `TAB` sol/sað pota arasýnda geçiþ.
- `SPACE` kaydedip sonraki kareye geç; `B` geri dön; `Q` çýkýþ.
- Etiketler `data/hoops/labels_v1/<video>/frame_XXXXXX.json` olarak saklanýr.

## 3. Dataset Hazýrlama
```
python tools/prepare_hoop_dataset.py
```
- Etiketli kareler YOLO formatýna çevrilir.
- Görseller `data/hoops/dataset/images/{train,val}`, label dosyalarý `labels/` altýna yazýlýr.
- Dataset yapýlandýrmasý `data/hoops/dataset/dataset.yaml` olarak oluþturulur.

## 4. Eðitim Paneli
```
python tools/hoop_training_panel.py
```
- Menüden `1` › dataset hazýrlama (adým 3 ile ayný komut).
- `2` › eðitim parametrelerini (model, epoch, imgsz vb.) girerek `tools/train_hoop_model.py` çaðrýlýr.
- Eðitim çýktý dizini `runs/hoop_training/<run_name>` altýnda oluþur; en iyi aðýrlýk `weights/best.pt` olarak kaydedilir.

## 5. Modeli Kullanma
- Eðitilen modeli `models/hoop_detector.pt` gibi anlamlý bir adla `models/` klasörüne kopyala.
- Kalibrasyon modülüne entegrasyon için Faz 2.2’nin sonraki adýmlarýna geç.

## Notlar
- Etiketleme seti büyüdükçe `labels_v2`, `labels_v3` gibi klasörler açýp versiyonlama yap.
- `export_samples` komutu artýk ham (raw) kareleri de kaydediyor; dataset hazýrlanýrken bu kareler kullanýlýr.
- Eðitim baþlamadan önce NVIDIA sürücüleri ve Ultralytics gereksinimleri hazýr olmalý (CUDA 11.8 + torch uyumu).
\n> Not: Daha önceki etiketler eski formatta kaydedildiyse data/hoops/labels_v1/ ve configs/calibrations/<video>_samples/ klasörlerini temizleyip yeni aracý kullanarak tekrar etiketleyin.
