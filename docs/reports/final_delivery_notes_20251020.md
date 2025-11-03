# Final Delivery Notes (20.10.2025)

## Scope
- Transcript tabanlı segment üretimi, QC analizi ve RunPod GPU kurulum planı.
- Tüm değişiklikler repo içine işlendi; kullanıcı faz 2 kalabalık klipleri devre dışı bırakacak şekilde yönlendirildi.

## Key Artifacts
- `tools/transcribe_video.py`, `tools/build_transcript_segments.py`: Whisper entegrasyonu ve segment çıkarma scriptleri.
- `data/event_samples/transcript_segments/`: `scorer_full_*`, `workout_full_*` dahil toplam 349 klip (S fitrasıyla üretildi).
- `runs/phase1/`: Scorer & Workout örnek setleri için `frame_stride=2` ve (problemli klipler) `frame_stride=1` metrikleri.
- `renders/`: `drive_angles_v2_seg014`…`seg021`, `scorer_full_seg00X`, `workout_full_seg001/002/004` overlay videoları.
- `docs/reports/deployment_plan_runpod.md`: RunPod on-demand GPU konfigürasyon ve otomasyon planı.
- `docs/reports/todo_transcripts_20251020.md`: Güncel TODO listesi (RunPod kurulumu, batch scriptler, QC raporlaması).

## Current Findings
- SCORER örnek seti: `max_ball_gap=2`, top kapsaması ~%49 (başarılı).
- Workout örnek seti: `frame_stride=1` ile tekrar işlendi; `workout_full_seg004` hariç top kapsaması %30-40 civarına yükseldi. Başlangıç karelerinde top görünmediğinden gap’ler doğal.
- `drive_angles_v2_*` manifest notları `phase2_pending` olarak güncellendi; kalabalık sahneler sonraki fazda değerlendirilecek.

## Outstanding Work
- RunPod otomasyonuna yönelik script ve Docker imajı yok; plan dosyası doğrultusunda geliştirilmesi gerekiyor.
- Manifeste eklenen yüzlerce transcript klibinin tamamı henüz QC’den geçmedi; batch script ile GPU üzerinde çalıştırılmalı.
- `docs/reports/highlight_qc.md` güncel değil; yeni metrikler işlenmesi gerekiyor.
- `tests/` altında event heuristiklerine yönelik fixture/test eklenmedi.

## Recommended Next Steps
1. `tools/batch_run_video.py` scriptini yazıp RunPod ortamında test etmek.
2. Storage senkronizasyon (rclone/S3) komutlarını dokümante edip otomatikleştirmek.
3. Transcript segmentleri için overlay/render batch’ini GPU üzerinde tamamlamak; QC raporlarına eklemek.
4. Manifesti kategorilere göre etiketleyip event heuristiklerine zemin hazırlamak.
5. Phase2 (drive_angles_v2) kliplerini daha güçlü tracker (ByteTrack vb.) ile tekrar değerlendirmek.
