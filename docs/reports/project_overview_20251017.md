# Proje Durum Ozeti (17.10.2025)

## Projenin Amaci
- Tek / cok kamerali amator basketbol maclarini isleyip oyuncu ve top takip verisi uretmek.
- Bu takibi temel alarak:
  1. Mac ve oyuncu bazli istatistikler (box score, tempo, run analizleri),
  2. Otomatik highlight videolari (oyuncu-takim kategorileri),
  3. Takim taktiklerinin (press, PnR, Blob/SLOB vb.) one ciktigi klipler uretmek.

## Guncel Durum
- **Detection & Tracking:** `app/run.py` Kalman destekli IOU tracker + statik top filtresi ile calisiyor; hakem tespiti kaldirildi.
- **Hoop Kalibrasyonu:** `app/calibration/hoops_cli.py` ile mac7/8/9 ortalamasi `configs/salon.yaml` dosyasina islendi.
- **Otomasyon:** `tools/run_video.py` tek komutta inference + metrik + overlay akisini sagliyor, QC tablosunu (`docs/reports/highlight_qc.md`) guncelliyor.
- **Overlay:** `tools/render_overlays.py` sahayi, hoop noktalarini, oyuncu/top ID'lerini ciziyor; cikti `renders/mac9.MP4_tracks_overlay.mp4`.
- **Testler:** `pytest` temel dogrulama senaryolarini kapsiyor (`tests/test_config_validation.py`, `tests/test_qc_metrics.py`, `tests/test_render_overlays.py`).
- **Raporlama:** `docs/reports/phase1_status_20251017.md` ve `docs/reports/highlight_qc.md` guncel metrikleri ve gozlemleri iceriyor.

## Eksikler / Riskler
1. **Top Surekliligi:** Kalman + `max_age=60` sonrasinda da `max_ball_gap_frames = 46`; top izlerinde kopma var.
2. **Statik Top Isaretleri:** CSV'lerde `is_static=1` olarak isaretleniyor ancak olay motoru/highlight secici bu kayitlari henuz filtrelemiyor.
3. **Olay Motoru:** Possession/shot/drive heuristikleri tamamlanmadi; events.csv uretimi yok.
4. **Pota Drift Izleme:** `tools/update_hoops_from_qc.py` coklu QC dosyasini ortalasa da drift uyarisi uretmiyor.
5. **RunPod Hazirligi:** Docker/handler taslagi henuz yok; GPU pipeline plan beklemede.

## Hedefler (Kisa Vadede)
1. **Takip Iyilestirme:** ByteTrack + Kalman veya optik akis stabilizasyonu ile `max_ball_gap < 20` hedefine ulasmak.
2. **Event Motoru MVP:** Drive / pull-up / spot-up / assist heuristikleri icin mini etiketli veri seti hazirlayip `app/events` tarafini devreye almak.
3. **Statik Top Filtreleme:** `is_static` bayrakli kayitlari event/highlight katmanindan haric tutacak kurallari yazmak; overlay'de istege bagli farkli renk kullanmak.
4. **Hoop Drift Raporu:** `tools/update_hoops_from_qc.py` coklu QC girdisini ortalayip sapmalari raporlayacak sekilde genisletmek.
5. **QC Surekliligi:** `docs/reports/highlight_qc.md` tablosunu her kosudan sonra doldurup gelismeleri izlemek.

## Hedefler (Orta Vadede)
1. **Highlight Secici:** Drive, pull-up, spot-up, assist, fast break, Blob/SLOB kategorileri icin otomatik klip secimi ve QC panosu.
2. **Takim Analitigi:** Possession bazli tempo/run/lead-change olcumleri, Blob/SLOB basari oranlari.
3. **Otomasyon / RunPod:** GPU tabanli handler (yt-dlp, YOLO, ffmpeg) + S3/R2 senkronizasyonu (rclone) + NAS entegrasyonu.
4. **Raporlama:** Mac basina PDF/HTML ozetleri, QC metriklerini ve highlight sonuclarini iceren standart rapor sablonu.

## Hedefler (Uzun Vadede)
1. **Model Egitimi:** Top/oyuncu davranislari icin ozel modeller (PnR tespiti, press break, court vision vb.) egitmek.
2. **SaaS Hazirligi:** Web panel (coach/analyst/player rolleri), FastAPI arayuzu, kullanici bazli proje yonetimi.
3. **Surekli Ogrenme:** QC'den gelen hatali orneklerin otomatik etiketlenip yeniden egitime girmesi, drift tespiti.
