# Veri & QC Paneli

Bu panel, geliştirme sürecinde ihtiyaç duyduğumuz geri bildirimleri tek yerde toplamayı amaçlar. Her koşudan sonra ilgili tabloları güncel tutman yeterli.

---

## 1. Çalıştırma Özeti
`tools/run_video.py` veya benzeri bir pipeline koşusu tamamlandığında doldur.

| Tarih / Saat | Video | Komut (stride/süre) | Konfig Notu | Ball Gap (frame) | Statik Top (adet) | Notlar (lag, kamera, hatalar) |
| --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |

**Not:** Ball gap ve statik top değerlerini `runs/phase1/*.json` veya terminal logundan alabilirsin.

---

## 2. Etiketleme & Olay Doğrulama Kuyruğu
Drive, pull-up, spot-up, assist vb. klipleri işaretlemek/incelemek için kullan.

| Clip ID / Path | Etiket | Frame Başlangıç | Frame Bitiş | Durum (Yeni / Kontrol / Onaylandı) | QA Notu |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

**İpucu:** Etiketleri GUI araçlarıyla üretip (`app/labeling/gui_labeler.py`, `app/labeling/video_labeler.py`) bu tabloya referans vermen yeterli.

---

## 3. Pota Kalibrasyon & Drift Takibi
`tools/update_hoops_from_qc.py` veya manuel kontroller sonrası sapmaları kaydet.

| Video / QC Dosyası | Sol Pota Sapması (px) | Sağ Pota Sapması (px) | Radius Farkı (px) | Aksiyon (yeniden kalibrasyon?) | Not |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

---

## 4. Takip / Statik Top Sorun Kayıtları
Örneğin köşede duran top, ID sıçraması, kamera kayması gibi dikkat çeken durumları burada tut.

| Video | Frame veya Zaman | Sorun Tipi | Açıklama | Öncelik / Öneri |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

---

## 5. Ek Dosya & Referanslar
Paylaştığın etiket paketleri, ekran kayıtları, karar belgeleri gibi kaynakları listeler.

| Tarih | Açıklama | Konum / Link | Parola (varsa) | Not |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

---

### Kullanım Hatırlatmaları
- Çalıştırma sonrası tablo 1’i güncelle, QC notlarını tablo 4’e yaz.
- Etiketlediğin her klibi Tablo 2’ye ekle; durum sütununu “Onaylandı”ya çektiğinde event heuristiklerinin doğruluğunu ölçebilirim.
- Pota drift verisini tablo 3’e gir; sapma yüksekse yeniden kalibrasyon planlarız.
- Ek materyalleri tablo 5’e ekleyerek hepsine tek noktadan erişiriz.

Bu paneli düzenli doldurman, bütün işlerin izlenebilirliğini sağlayacak. Herhangi bir tabloya ek sütun lazım olursa haber ver, birlikte genişletiriz.***
