# Known Issues and TODOs

## Current Gaps (Faz 1)
- Pota overlay 5. dakikadan sonra kayboluyor; homografi + Kalman takviyesi planla.
- YOLOv8n top tespiti başarısız (cls=32 yok). YOLOv8s denenecek; gerekirse basketbola özel fine-tune.
- `app.events.cli` raporları sabit kamerada top tespiti olmadığından boş. Şut/ribaund doğrulaması yapılamadı.
- mac1/2/3 arşivde; odak mac7/8/9 videolarında.

## Planned Enhancements
- Stabilizasyon + homografi kaydı (Faz 2).
- ByteTrack entegrasyonu (top/oyuncu).
- Sekans modeli (LSTM/Transformer) – veri etiketleme gereksinimi.
- Kalibrasyon ve QC paneli (Streamlit/GUI) otomatik onay.

