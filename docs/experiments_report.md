# Thí nghiệm HT-Step, 31/10/2026 – 01/11/2026

- Nhánh: `htstep-ov`
- Log: `logs/train_<tên>.log`
- Checkpoint: `ckpt/htstep_<tên>/model_best.pth.tar`

## Thiết lập chung cho mọi lần chạy

### Dữ liệu

- Feature k400_f16: `k768` cho video, `a768` cho audio, mỗi giây 1 dòng.
- Tập train có feature cho 4810 trong 17472 video có nhãn.
- Tập val_seen có feature cho 505 trong 600 video:
  - 90 video không còn video hay feature;
  - 5 video có file npz bị hỏng, xem `broken_feats.txt`.

### Huấn luyện

- Batch 32, fp32.article_head
- lr 3e-4, `clip_proj_lr` 3e-3.
- 12 epoch, tính cả warmup.
- Sau mỗi epoch, chấm điểm model EMA.

### Thang đo

- Dùng mAP article grounding chính thức của HT-Step, với tIoU từ 0.3 đến 0.7, trên 505 video val_seen.
- Trên 505 video này, hàm `article_grounding_map` của mình cho đúng số như script chính thức `data/htstep/evaluation/eval_article_grounding.py`.
- Nếu chấm trên đủ 600 video GT, script chính thức cho điểm thấp hơn, vì video thiếu bị tính là trượt. Ví dụ baseline chỉ còn 21.79.

### Cách chọn checkpoint

- Chọn checkpoint và dừng sớm đều theo mAP.
- R@0.5 chỉ được ghi lại để tham khảo. GT của HT-Step chỉ phủ khoảng 20% độ dài video, nên R@0.5 (tính trên top-k) phạt cả những bước có thật mà chưa được gán nhãn.

### Độ nhiễu

Mỗi lần chạy chỉ dùng một seed, nên chênh lệch dưới khoảng 0.5 mAP coi như nằm trong dao động giữa các lần chạy.

### Test cfg

| | `max_seg_num` | `nms_sigma` |
|---|---|---|
| cũ | 100 | 0.5 |
| mới (đang là mặc định) | 200 | 0.75 |

Cùng một checkpoint baseline đạt 25.26 với test cfg cũ và 25.70 với test cfg mới.

## 1. Baseline và thiết lập lúc suy luận

| Lần chạy | Thay đổi | mAP tốt nhất | Epoch | R@0.5 |
|---|---|---|---|---|
| b32 | lần chạy batch 32 đầu tiên, chọn theo R@0.5 | 24.22 (các epoch cuối) | – | 31.66 |
| map | như trên, chọn theo mAP, test cfg cũ | 25.26 | 8 | 32.24 |
| **baseline** | checkpoint của `map`, test cfg mới | **25.70** | 8 | – |

Các cách chỉnh lúc suy luận đã thử trên checkpoint baseline, nhưng **không giúp được gì**:

- chuẩn hoá điểm theo từng video;
- chấm điểm theo thứ hạng;
- prior theo activity;
- re-ranker.

Đã thử thêm cách của UniAV: nhân điểm của segment với xác suất của lớp, tức P(step)^g. Cách này thử trên checkpoint artc:

| g | 0 | 0.25 | 0.5 | 1.0 |
|---|---|---|---|---|
| mAP | **30.32** | 30.13 | 28.63 | 25.32 |

## 2. Ablation khi chưa có article conditioning

So với baseline: 25.26 nếu dùng test cfg cũ, 25.70 nếu dùng test cfg mới.

| Lần chạy | Thay đổi | Test cfg | mAP tốt nhất | Epoch |
|---|---|---|---|---|
| qfl | event head học theo IoU, dùng quality focal loss | cũ | 25.33 | 7 |
| ctr | centre sampling, bán kính 1.5 | cũ | 25.00 | 8 |
| deep | `backbone_arch` (2,6,5) thay cho (2,3,5) | cũ | 24.81 | 7 |
| crop | `crop_ratio` (0.7, 1.0) thay cho (0.9, 1.0) | cũ | 24.78 | 8 |
| drop1 | dropout 0.1 | cũ | 24.68 | 8 |
| big | `embd_dim` 768, `n_head` 8 | cũ | 24.54 | 7 |
| wd | `weight_decay` 0.05 | cũ | 24.34 | 8 |
| lr1 | lr 1e-4 (`clip_proj_lr` 1e-3) | cũ | 24.34 | 6 |
| half | chỉ dùng một nửa số video train | cũ | 22.76 | 7 |
| ema99 | EMA decay 0.99, dừng sau 1 epoch (xem chú thích dưới bảng) | – | – | – |
| noaudio | chỉ dùng luồng video | mới | 25.27 | 11 |
| tmax | backbone TemporalMaxer (`convPooler`) | mới | 24.51 | 7 |
| dyfa | encoder layer của DyFADet (`dynE`) | mới | 24.94 | 7 |
| 512 | `max_seq_len` 512 (Colab L4) | mới | 25.66 | 9 |

**Chú thích ema99:** decay thực tế là (1+u)/(10+u), với u là số bước đã train. Đường này tăng rất chậm, nên trong một lần train decay không bao giờ chạm tới 0.999. Vì vậy hạ decay xuống 0.99 không tạo ra khác biệt, và run được dừng sớm.

**Kết quả:** không cách nào vượt được baseline. Yếu tố ảnh hưởng mạnh nhất là lượng dữ liệu: dùng một nửa dữ liệu đạt 22.76, dùng toàn bộ đạt 25.26.

## 3. Article conditioning (artc) và các cách ghép thêm

**artc** cho feature ở mọi tầng pyramid cross-attention sang vector text của các bước trong article của video.
- Mỗi bước có thêm một order embedding, cho biết đó là bước thứ mấy trong công thức.
- Lớp output được khởi tạo bằng 0, nên lúc mới train model y hệt baseline.
- Ý tưởng giống ActionFormer-T trong bài báo HT-Step.

Article là đầu vào có sẵn của bài toán article grounding. Script chấm chính thức cũng lọc dự đoán theo article.

| Lần chạy | Thay đổi | mAP tốt nhất | Epoch | R@0.5 |
|---|---|---|---|---|
| **artc** | article conditioning, 256 | **30.32** | 8 | 35.32 |
| a_qfl | artc + QFL | 30.08 | 9 | 35.93 |
| a_ctr | artc + centre sampling 1.5 | 30.41 | 10 | 36.04 |
| a_deep | artc + `backbone_arch` (2,6,5) | 30.11 | 9 | 36.11 |
| **a512** | artc + `max_seq_len` 512 (Colab L4) | **30.62** | 9 | 36.38 |

**Kết quả:**
- artc tăng **+4.6 mAP** so với baseline.
- Các cách ghép thêm đều chỉ chênh với artc trong mức dao động. Ghép 512 tăng nhiều nhất, khoảng +0.3.

Để tham khảo, Table 2 của bài báo (chấm trên test_seen) cho:

| Mô hình | mAP |
|---|---|
| ActionFormer | 25.4 |
| ActionFormer-T | 26.3 |
| Mô hình tốt nhất của họ | 30.2 |

## 4. Head article: đoán article khi không được cho trước

**ArticleHead** lấy feature ở tầng 0, gộp theo thời gian bằng trung bình và max (bỏ phần padding).
- Kết quả gộp được so với vector của từng article. Vector của một article là trung bình vector text các bước của nó.
- Head được train bằng cross-entropy trên 617 article, trọng số 0.5.
- Checkpoint được chọn khi chấm với article cho sẵn. Sau đó chấm lại một lần với `test_cfg.pred_article=True`, tức article là top-1 do model đoán.

| Lần chạy | Thay đổi | mAP (article cho sẵn) | ArtAcc | mAP (article tự đoán) |
|---|---|---|---|---|
| a512_ah | artc + 512 + head học chung với backbone | 28.88 (ep 10) | **53.5%** | **18.30** |
| a_ahd | artc + 256 + head tách gradient | **31.03** (ep 9) | 30.7% | 11.57 |
| zero-shot | dùng embedding bước của baseline: cộng xác suất bước theo article | – | 41.2% (top-5 69.7%) | chưa chạy (ước ~14) |

Thử dùng gộp các bước của top-k article làm ứng viên, trên checkpoint a_ahd. Càng tăng k, mAP càng giảm:

| k | 1 | 2 | 3 | 5 | 10 |
|---|---|---|---|---|---|
| mAP | 11.57 | 11.59 | 10.92 | 10.07 | 9.01 |

**Kết quả:**
- mAP khi tự đoán article ≈ mAP khi article cho sẵn × tỉ lệ đoán đúng article.
- Đoán sai article thì mất điểm cả video, vì script chính thức chỉ giữ các bước thuộc article thật. Gộp nhiều article cũng không cứu được.
- So sánh hai cách gắn head:

| Cách gắn head | Ưu điểm | Nhược điểm |
|---|---|---|
| Học chung với backbone | đoán article tốt hơn | mất khoảng 2 mAP khi article cho sẵn |
| Tách gradient | giữ được khả năng phát hiện bước (31.03) | đoán article kém |

Con số 31.03 của head tách gradient có thể là do gradient clipping: norm được tính trên nhiều tham số hơn, nên gradient của phần còn lại bị thu nhỏ.

## 5. Các kiểm tra khác

- **Video val bị thiếu:** có 18 trong 90 video lấy lại được.
  - 5 video vẫn còn public trên YouTube.
  - 15 video có trong bản ETBench `ht_step.tar.gz`.
  - Hai nguồn trùng nhau 2 video, nên tổng là 18.
  - Số còn lại phải điền form xin tải của tác giả HowTo100M.
- **505 hay 600 video:** mAP chấm trên 505 video là ước lượng hợp lý cho chất lượng model. Điểm val và test thường chỉ chênh nhau khoảng 1 điểm.

## 6. Kết luận và việc tiếp theo

**Benchmark (article cho sẵn):**
- Dùng artc, đạt khoảng 30.5–31 mAP.
- Thêm 512 hay head tách gradient chỉ tăng ở mức dao động.

**Khi không có article:**
- Tốt nhất hiện tại là 18.30, với head học chung.
- Muốn tăng phải đoán article tốt hơn. Cách khả thi nhất là thêm video train: hiện mỗi article chỉ có khoảng 8 video.
- Một cách khác là dùng 2 model: head học chung để đoán article, rồi artc + 512 để tìm các bước. Ước tính được khoảng 18–19.

**Chưa chạy:**
- OV-AVEL `audio_text`: đã có code, mới chạy smoke test.
- Kiểm tra lại clip_grad / lr. Kết quả của a_ahd gợi ý nên thử điều này.
