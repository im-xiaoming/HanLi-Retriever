# HT-Step: tổng kết 07/11 – 09/11/2026

Nhánh `htstep-omni-distill`. Model cuối: commit `95d9592` (code dọn lại ở `5c5b63a`, phép tính không đổi). Chi tiết
từng run (commit, `--set`, kết quả theo epoch) nằm trong `docs/experiments_report.md`.

## 1. Kết quả

Đo trên val_seen (510 video có feature; các chỉ số omni đo trên 1080 đoạn GT có vector teacher).

| Model | mAP (article cho sẵn) | ArtAcc | OmniCos | OmniCosC | OmniR@1 |
|---|---|---|---|---|---|
| Baseline 01/11 (chưa có artc) | 25.26 | – | – | – | – |
| pc_v6 (07/11, InfoNCE + bank) | 30.50 | 61.6 | 86.5 | 72.9 | 62.1 |
| **a_kd (cuối, KL distillation)** | **30.37** | 59.8 | **91.1** | **75.2** | 60.8 |

- **mAP:** mAP article grounding chính thức, tIoU 0.3–0.7.
- **ArtAcc:** tỉ lệ đoán đúng article trong 617 article.
- **OmniCos:** cosine giữa student và teacher OmniRetriever trên đoạn GT có start–end cho sẵn. Mục tiêu đặt ra là trên 80, và đã đạt 91.
- **OmniCosC:** như OmniCos nhưng trừ vector trung bình của teacher train. Một model chỉ đoán vector trung bình đã đạt OmniCos 79.7, nên chỉ số này mới cho thấy phần thật sự học được.
- **mAP khi article do model tự đoán** (chọn bước mềm): 21.23 với pc_v6. Chưa đo lại với a_kd.

So với pc_v6, model cuối có OmniCos cao hơn 4.6 và OmniCosC cao hơn 2.3. mAP ngang, còn ArtAcc thấp hơn 1.8.

## 2. Model

Backbone UniAV gồm hai luồng video (InternVideo2 K400) và audio (BEATs), có cross-attention và pyramid 6 tầng. Trên backbone có các phần sau:

1. **Article conditioning (artc).** Feature attend vào text các step của article cho sẵn. Phần này đóng góp khoảng +5 mAP.
2. **Event head và boundary head.** Kèm một điểm IoU-quality dùng để xếp hạng.
3. **Chọn step.** Vector của đoạn (trung bình feature tầng 0, rồi qua `SegmentContext`) được so với text InternVideo2 của các step trong article.
4. **Article head.** Chọn article trong 617, dựa trên text các step của nó cộng một vector học được cho mỗi article.
5. **Omni distillation.** Model học lại vector `encode_av` của OmniRetriever (3584 chiều) cho từng đoạn GT, qua hai head:
   - point head: mọi điểm nằm trong đoạn;
   - span head: attention pooling trên đoạn có start–end cho sẵn.

   Loss gồm KL giữa phân phối similarity của student và của teacher trên các teacher trong batch, cộng `2 · (1 − cos)`.

Training mất khoảng 12 epoch, khoảng 45 phút trên A100. Kết quả hoàn toàn tất định: cùng code, cùng dữ liệu và cùng seed cho ra đúng cùng các con số. Đổi seed chỉ làm mAP lệch khoảng 0.05.

## 3. Thay đổi nào có ích, thay đổi nào không

**Có ích, đã giữ:**

| Thay đổi | Tác dụng |
|---|---|
| Article conditioning | mAP từ ~25.5 lên ~30 |
| Vector học được cho mỗi article | +5–7 ArtAcc |
| Loss `1 − cos` (trọng số 2.0) | OmniCos từ ~40 (chỉ có InfoNCE) lên ~86 |
| Span head (8 query, hidden 4096) | Bỏ đi thì mAP −1.2, OmniCosC −5.6, OmniR@1 −11 |
| KL distillation thay InfoNCE | OmniCos +1.3, OmniCosC +0.4 |
| Chọn bước mềm khi article tự đoán | +2 mAP |

**Không có ích, đã xoá:**
- Bank omni gồm 4096 hoặc 60000 negative có mining.
- Che negative cùng step.
- Backbone TemporalMaxer và DyFADet.
- Chọn article cứng (top-k).
- Attention pooling cho article head.
- Label smoothing.
- Các cờ bật/tắt của những phần luôn bật.
- Phần lớn filter dữ liệu.

Tổng cộng, code ít hơn khoảng 210 dòng.

**Phải giữ cùng nhau:**
- `emb_loss`, nhánh IoU, `SegmentContext`, EMA và random crop. Bỏ riêng từng phần thì không mất gì, nhưng bỏ cả 5 thì mAP giảm khoảng 2 (28.42).
- Giữ lại EMA và crop vẫn còn thiếu khoảng 1.3 mAP (29.05). Thêm lại riêng một trong ba phần còn lại thì cũng không đủ, phải có cả ba mới đạt 30.37.

**Text của OmniRetriever** (`encode_text` cho 4958 step), đã thử và không dùng:
- Thay text InternVideo2 bằng text omni: mAP −0.5, ArtAcc +1.4.
- Ghép hai loại text: ngang mốc.
- Thêm loss kéo span head về text omni: các chỉ số omni giảm 5–10.
- Ghép điểm lúc dự đoán: Acc chọn step +1.85, nhưng mAP không đổi.

## 4. Bài học

- **Rò rỉ đáp án qua artc.** Feature sau artc đã "thấy" text của article đúng. Mọi điểm article tính từ những feature này (vector step, vector omni) đều bị rò rỉ. Lỗi này đã gặp hai lần: ArtAcc_step lên 70% ngay ở epoch 1, và ghép omni làm ArtAcc tăng giả 4.5. Phải tính trên feature trước artc.
- **Ablation từng phần không đủ.** Nhiều phần trông vô dụng khi bỏ riêng, nhưng bỏ cùng lúc thì mất 1–2 mAP. Trước khi chốt một bản gọn, cần chạy thử bản đã bỏ tất cả.
- **Chênh lệch giữa các biến thể là thật, không phải nhiễu.** Vì training tất định, một khoảng chênh 0.5–1 mAP giữa hai bản code phản ánh đúng ảnh hưởng của thay đổi.
- **ArtAcc bị giới hạn bởi dữ liệu.** Mới có 5826 trên 17472 video train có feature. Theo learning curve, mỗi lần gấp đôi dữ liệu thì ArtAcc tăng khoảng 9 điểm.

## 5. Việc tiếp theo

1. Trích k400 feature cho khoảng 11.6k video train còn thiếu. Đây là cách lớn nhất để tăng ArtAcc: dự đoán lên khoảng 72–74%.
2. Đo mAP khi article tự đoán với model a_kd. Cần train lại để có checkpoint, vì checkpoint a_kd nằm trên Colab và đã mất.
3. Nếu ưu tiên ArtAcc hơn mAP: dùng text omni làm vector step. Chỉ cần đổi 2 dòng config, đổi được +1.4 ArtAcc với giá −0.5 mAP.

## 6. Dữ liệu và tài nguyên

| Dữ liệu | Nơi lưu |
|---|---|
| Feature k400 | HF `htstep-feats-k400` (train), `htstep-val-seen-feats` (val) |
| Vector omni AV | HF `htstep-omni-av` (train), `htstep-val-seen-feats/omni_av/` (val) |
| Text step InternVideo2 và omni | HF `htstep-text-feats/text_feats/` |

- **Log:** `logs/` (PC), `logs/omab/`, `logs/simp/`, `logs/otext/` (Colab).
- **Colab:** gói Pro+ ở tài khoản minhnguyen47431. Số dư còn khoảng 425 CU, đợt này dùng khoảng 45 CU.
