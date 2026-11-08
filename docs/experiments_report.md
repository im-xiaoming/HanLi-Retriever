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

---

# Thí nghiệm HT-Step, 07/11/2026 – 08/11/2026

- Nhánh: `htstep-omni-distill`. Commit cuối: `a53480a`.
- Log: `logs/<tên>.log` cho các lần chạy trên PC. Log Colab nằm trên VM nên đã mất; các dòng theo epoch được chép lại vào report này.
- Checkpoint: `ckpt/<tên>/model_best.pth.tar` (mAP tốt nhất) và `ckpt/<tên>/model_best_article.pth.tar` (ArtAcc tốt nhất).
- Máy:
  - PC RTX 3060: 160–245 s mỗi epoch.
  - Colab T4 miễn phí: 170–360 s mỗi epoch. Tài khoản có 0 CU nên A100 và L4 bị từ chối.

## 1. Dữ liệu

| | Train | val_seen |
|---|---|---|
| Video có feature k400 | 5826 (5816 dùng được) | 510 |
| Video có vector omni (OmniRetriever `encode_av`, 1 vector 3584 chiều mỗi đoạn GT) | 5465 | 179 |

Tổng số video có feature k400 là 6336.

Nguồn dữ liệu:
- `tools/fetch_htstep.py` tải từ HF:
  - k400 lấy từ `htstep-feats-k400` và `htstep-val-seen-feats`. Các tar `zz_fix_*` được giải nén sau cùng để ghi đè lên bản cũ.
  - Omni lấy vào `datasets/omni_av_feats/train` và `datasets/omni_av_feats/val`.
- Vector omni của 179 video val đã được chuyển sang `htstep-val-seen-feats/omni_av/`. Repo `htstep-omni-av` giờ chỉ còn dữ liệu train.

Kiểm tra bằng `tools/validate_htstep.py`:
- Không có lỗi chặn.
- `seg_idx` và `step_id` của omni khớp với annotation.
- 20 video có số dòng feature lệch so với `duration`. Dataset bỏ qua những video có nhiều dòng hơn `duration + max(5, 5%)`, vì đó có thể là một bản video khác.

## 2. Thiết lập chung

**Huấn luyện:**
- Batch 32, fp32.
- lr 3e-4, `clip_proj_lr` 3e-3, weight decay 1e-4.
- 12 epoch, tính cả 2 epoch warmup.
- EMA 0.999.
- `max_seq_len` 256.

**Dừng sớm:**
- Theo dõi mAP, ArtAcc và OmniCos.
- Một chỉ số được tính là "tốt hơn" khi tăng ít nhất `--min_delta 0.3`.
- Dừng khi 2 epoch liên tiếp không có chỉ số nào tốt hơn (`--patience 2`).

**Omni head:**
- Mỗi điểm nằm trong một đoạn GT học vector teacher của chính đoạn đó.
- Loss InfoNCE (trọng số 0.2, nhiệt độ 0.05). Negative gồm:
  - các đoạn khác trong batch;
  - một danh sách ứng viên 4096 phần tử (`OmniCandidateBank`). Mỗi step, các đoạn của batch được thêm vào. Khi danh sách đầy, mining sẽ bỏ các ứng viên yếu nhất, tức là ít giống student nhất.
- Negative cùng step với anchor bị mask.

**Thang đo:**

| Chỉ số | Ý nghĩa |
|---|---|
| mAP | mAP article grounding chính thức trên 510 video, article cho sẵn |
| ArtAcc | tỉ lệ đoán đúng article (trên 617 article) bằng ArticleHead |
| OmniCos | cosine trung bình giữa student và teacher trên 1080 đoạn GT val, start–end cho sẵn |
| OmniCosC | cosine sau khi trừ vector trung bình của teacher train |
| OmniR@1 | tỉ lệ đoạn mà teacher của chính nó gần student nhất, so với mọi teacher val |

## 3. Config từng lần chạy

**Config chung:** `train.py --output <tên> --set ...`. Các khoá ở bảng dưới được ghi đè lên config tại commit đã nêu. Các khoá còn lại giữ mặc định cũ:
- `article_cond` False, `article_head` False, `article_learned` 0, `omni_head_dim` 3584;
- `loss_weight_omni_cos` 1.0 (từ commit `aed4828`);
- `omni_span_head` False, `pool_queries` 4, `omni_span_hidden` 2048.

Các khoá có đánh dấu † đã bị xoá khỏi code vì không có ích.

| Lần chạy | Máy | Commit | Override (`--set`) | Ghi chú |
|---|---|---|---|---|
| pc_ah_noomni | PC | 65a72a6 | `model.article_head=True model.omni_head_dim=0` | không có omni |
| t4_ah_omni | T4 | 65a72a6 | `model.article_head=True` | omni chỉ có InfoNCE, chưa có loss cos; dừng ở ep 6 |
| pc_ah_learn_omni | PC | 5476780 | `model.article_head=True model.article_learned=617` | dừng ở ep 1, thay bằng run kế tiếp |
| pc_ah_learn_cos | PC | aed4828 | `model.article_head=True model.article_learned=617` | loss cos trọng số 1.0 |
| t4_ah_cos | T4 | aed4828 | `model.article_head=True` | loss cos 1.0, không có `article_learned`; dừng ở ep 5 |
| pc_v3 | PC | b00e953 | `model.article_head=True model.article_learned=617 model.article_all_levels=True† model.train_cfg.article_label_smoothing=0.1† model.train_cfg.loss_weight_omni_cos=2.0` | |
| t4_v3a | T4 | b00e953 | `model.article_head=True model.article_learned=617 model.train_cfg.loss_weight_omni_cos=2.0` | |
| pc_crop03 | PC | 85efcca | như t4_v3a + `model.train_cfg.article_crop_min=0.3†` | dừng ở ep 4 vì không tăng |
| t4_half | T4 | 85efcca | như t4_v3a + `dataset.train_fraction=0.5` | learning curve |
| pc_v4 | PC | 0a4e317 | như t4_v3a + `model.article_cond=True model.omni_span_head=True model.article_attention_pool=True†` | |
| t4_v4_noattn | T4 | 0a4e317 | như pc_v4 nhưng không có `article_attention_pool` | Colab bị thu hồi ở ep 9, mất checkpoint |
| pc_v5 | PC | 7d2dcfb | như pc_v4 + `model.pool_queries=8 model.omni_span_hidden=4096` | |
| **pc_v6** | PC | 7d2dcfb | như pc_v5 nhưng không có `article_attention_pool` | **= config mặc định từ `a53480a`** |

Config đầy đủ của pc_v6, cũng là mặc định hiện tại:

```
model.article_cond = True              model.article_head = True          model.article_learned = 617
model.omni_head_dim = 3584             model.omni_span_head = True        model.pool_queries = 8
model.omni_span_hidden = 4096
model.train_cfg.loss_weight_omni_emb = 0.2      model.train_cfg.loss_weight_omni_cos = 2.0
model.train_cfg.loss_weight_omni_span_nce = 0.2 model.train_cfg.loss_weight_omni_span_cos = 2.0
model.train_cfg.omni_bank_size = 4096  model.train_cfg.omni_temperature = 0.05
model.train_cfg.loss_weight_article = 0.5
model.test_cfg.article_soft = False (True khi article tự đoán)   model.test_cfg.article_soft_weight = 0.5
```

## 4. Kết quả

Cột mAP là mAP lớn nhất trong log (epoch trong ngoặc). Các cột ArtAcc, OmniCos, OmniCosC và OmniR@1 là giá trị lớn nhất trong cả lần chạy. Dấu – nghĩa là chưa có chỉ số đó.

| Lần chạy | Epoch đã chạy | mAP | Acc | ArtAcc | OmniCos | OmniCosC | OmniR@1 |
|---|---|---|---|---|---|---|---|
| pc_ah_noomni | 12 | 25.02 (10) | 52.6 | 56.27 | – | – | – |
| t4_ah_omni | 6 | 24.84 (6) | 52.6 | 52.94 | 40.4 | – | 43.7 |
| pc_ah_learn_cos | 12 | 25.57 (9) | 56.5 | 58.82 | 81.3 | 65.6 | 49.5 |
| t4_ah_cos | 5 | 24.18 (5) | 51.4 | 48.24 | 81.4 | 64.0 | 38.6 |
| pc_v3 | 12 | 25.47 (10) | 57.5 | 58.43 | 85.8 | 68.3 | 49.1 |
| t4_v3a | 12 | 25.40 (9) | 54.4 | 59.41 | 85.6 | 68.1 | 48.9 |
| pc_crop03 | 4 | 23.12 (4) | 48.1 | 49.61 | 85.5 | 64.1 | 33.2 |
| t4_half | 12 | 23.55 (8) | 49.6 | 50.00 | 85.1 | 65.3 | 40.1 |
| pc_v4 | 10 | 29.25 (9) | 66.6 | 59.22 | 86.4 | 72.5 | 61.2 |
| t4_v4_noattn | 9 | 30.49 (9) | 65.9 | 60.59 | 86.3 | 72.5 | 60.4 |
| pc_v5 | 11 | 30.10 (10) | 67.7 | 59.22 | 86.5 | 72.7 | 60.7 |
| **pc_v6** | 11 | **30.50** (9) | 66.4 | **61.57** | 86.5 | **72.9** | **62.1** |

**mAP khi article do model tự đoán** (`test_cfg.pred_article=True`):

| Checkpoint | Cách chọn bước | mAP |
|---|---|---|
| pc_v3 | top-1 cứng | 16.09 |
| pc_v3 | mềm, trọng số 2.0 / 1.0 / **0.5** | 17.39 / 17.63 / **18.14** |
| pc_v3 | mềm, p(article) lấy từ fused z-score | 17.89–18.21 (không đáng kể, đã bỏ) |
| pc_v4 | top-1 cứng / mềm | 18.41 / 18.43 |
| pc_v4 | top-3 article cho artc + mềm | 16.83 |
| **pc_v6** | mềm, trọng số 0.5 | **21.23** |

Chọn mềm (`article_soft`) nghĩa là mọi bước đều được xét, với điểm `log p(bước | đoạn) + w · log p(article của bước | video)`. Cách này không bắt buộc phải đoán đúng article top-1.

## 5. Các phép đo phụ

**Teacher omni:**
- Cosine trung bình giữa hai teacher train bất kỳ là 0.637.
- Nếu đoán một vector hằng là trung bình teacher train, OmniCos đạt **79.7**. Mốc OmniCos > 80 vì vậy gần như có sẵn, nên cần xem thêm OmniCosC.
- Oracle dùng prototype của bước: 88.5. Oracle dùng prototype của article: 85.6.

**Đoán article bằng teacher omni** (179 video val, dùng vector teacher của các đoạn GT):

| Cách | ArtAcc |
|---|---|
| Max-kNN theo đoạn | 45.8% |
| Trung bình video so với prototype article | 67.0% |

**Feature thô** (mean+max của k768 và a768, không qua backbone): ArtAcc tốt nhất 41.8% với linear, 41.4% với MLP. Kết quả này thấp hơn head trên backbone (~56%), nên backbone có giúp ích.

**Phân tích lỗi article** (checkpoint pc_ah_noomni):

| | Kết quả |
|---|---|
| Đúng article | 55.9 |
| Đúng món | 59.0 |
| Top-5 | 80.8 |

Khoảng 93% lỗi là đoán sang món khác hẳn, không phải nhầm biến thể của cùng món. Trên pc_ah_learn_cos, nối điểm head với điểm omni được 59.6, so với 58.6 của head một mình.

**Learning curve:**
- t4_half (50% dữ liệu train) chững lại ở ArtAcc ~50%.
- t4_v3a (100% dữ liệu, cùng config) đạt 59.4%.
- Vậy mỗi lần gấp đôi dữ liệu thì ArtAcc tăng khoảng 9 điểm.

## 6. Kết luận

**Những thay đổi có ích, đã giữ lại:**

| Thay đổi | Tác dụng |
|---|---|
| Omni distill theo đoạn (loss InfoNCE + bank) | khoảng +0.5–0.7 mAP và +3 Acc so với không có omni |
| Loss `1 − cos`, trọng số 2.0 | OmniCos lên ~86; với InfoNCE riêng chỉ đạt ~40 |
| `article_learned` | +5–7 ArtAcc ở các epoch đầu |
| artc | mAP khi article cho sẵn tăng từ ~25.5 lên ~30 |
| Omni span head (attention pooling cho đoạn cho sẵn) | OmniCosC từ 68 lên 72.9, OmniR@1 từ 49 lên 62 |
| Tăng capacity span head (8 query, hidden 4096) | khoảng +0.5–1 mAP |
| Chọn bước mềm khi article tự đoán | +2 mAP |

**Những thay đổi không có ích, đã xoá:**
- Pool qua mọi tầng pyramid cho article.
- Label smoothing.
- Cắt cửa sổ ngẫu nhiên cho article.
- Attention pooling cho article head: không có nó lại ngang hoặc tốt hơn.
- Điểm article tính từ step và omni, cùng cách ghép chúng.

**Cảnh báo về rò rỉ:** khi bật artc với article cho sẵn, không được tính điểm article từ vector step hoặc omni. Những vector này đã chứa text của article đúng, nên ArtAcc_step từng bị đẩy lên 70% ngay ở epoch 1.

**Việc tiếp theo:**
- ArtAcc đang bị giới hạn bởi dữ liệu. Mới có 5826 trên 17472 video train có feature k400. Nếu learning curve giữ đúng xu hướng, đủ dữ liệu có thể đưa ArtAcc lên khoảng **72–74%**.
- Để làm việc này cần trích feature cho khoảng 11.6k video, tức là cần A100 hoặc compute unit.

# Thí nghiệm HT-Step, 08/11/2026: ablation omni, rút gọn code, KL distillation

- Nhánh: `htstep-omni-distill`. Commit cuối: `95d9592` (model cuối = run `a_kd`).
- Máy: Colab A100 (gói Pro+ của tài khoản minhnguyen47431). Chạy 3 run cùng lúc trên một A100 40 GB, khoảng 220 s mỗi epoch. Chạy 4 run cùng lúc thì bị OOM.
- Log: `logs/omab/<tên>.log` (ablation omni) và `logs/simp/<tên>.log` (rút gọn code, KL). Checkpoint nằm trên VM Colab nên không còn.
- **Training tất định:** chạy lại cùng code, cùng dữ liệu, cùng seed cho ra kết quả giống hệt đến từng chữ số (`r_kd` lặp lại đúng `a_kd`). Vì vậy khác biệt giữa các run là do code, không phải do nhiễu ngẫu nhiên. Riêng việc đổi seed làm mAP lệch khoảng 0.05 (`f_emacrop` 29.05 so với `s_seed2` 29.00).
- Dừng sớm, thang đo, batch và lr giống mục 07–08/11.

## 1. Ablation omni (code trước khi rút gọn)

Các run này dùng commit `b878676` (vẫn còn bank), cấu hình mặc định là pc_v6, mỗi run chỉ đổi đúng một khoá.

| Lần chạy | Override (`--set`) | mAP (ep) | ArtAcc | OmniCos | OmniCosC | OmniR@1 |
|---|---|---|---|---|---|---|
| pc_v6 (PC, mốc cũ) | – | 30.50 | 61.6 | 86.5 | 72.9 | ~62 |
| o_nobank | `model.train_cfg.omni_bank_size=0` | **31.15** (9) | 60.8 | 89.8 | 74.8 | 61.6 |
| o_bank60k | `model.train_cfg.omni_bank_size=60000` | 30.91 (9) | 61.4 | 87.0 | 73.3 | 62.0 |
| o_nopoint | `model.train_cfg.loss_weight_omni_emb=0.0 model.train_cfg.loss_weight_omni_cos=0.0` | 30.22 (9) | 60.2 | 86.4 | 72.4 | 58.1 |
| o_nospannce | `model.train_cfg.loss_weight_omni_span_nce=0.0` | 29.73 (9) | 59.8 | 92.0 | 75.0 | 55.6 |
| o_nospan | `model.omni_span_head=False` | 29.30 (10) | 57.3 | 85.9 | 69.2 | 50.4 |

Kết luận:
- Bank (4096 hay 60000) không giúp gì nên **đã xoá** (`omni_bank.py`, commit `2f40370`).
- Span head và InfoNCE của span head có ích, giữ lại.
- Loss omni theo point gần như không giúp mAP, nhưng giữ được khoảng 4 OmniR@1. Vẫn giữ.

## 2. Rút gọn code và ablation từng phần

Commit `0c85e95` rút gọn code:
- Omni viết lại thành một danh sách teacher cho mỗi batch, mỗi student có một `target` trỏ tới dòng teacher của nó.
- Các head luôn bật (bỏ cờ).
- Bỏ các tuỳ chọn không dùng và các filter dữ liệu gần như không loại gì.

Mốc mới là `s_base`, tức code rút gọn với cấu hình mặc định. Mỗi run `a_*` bỏ đúng một phần so với `s_base`.

| Lần chạy | Code | Override (`--set`) | Bỏ phần nào | mAP (ep) | ArtAcc | OmniCos | OmniCosC | OmniR@1 |
|---|---|---|---|---|---|---|---|---|
| c_old | `6f7292b` (trước rút gọn) | – | (đối chứng) | 30.73 (10) | 62.2 | 89.7 | 74.7 | 62.0 |
| s_base | `0c85e95` | – | (mốc) | 29.78 (10) | 60.8 | 89.8 | 74.8 | 62.8 |
| a_nostep | `abl-nostep` | – | che teacher cùng step trong InfoNCE | 30.10 (10) | 60.0 | 89.9 | 74.9 | 62.4 |
| a_noctx | `abl-noctx` | – | `SegmentContext` (vector đoạn = trung bình feature) | 29.91 (9) | 59.4 | 89.7 | 74.7 | 62.0 |
| a_noemb | `0c85e95` | `model.train_cfg.loss_weight_emb=0.0` | InfoNCE theo point với text step | 30.42 (10) | 60.8 | 89.8 | 75.0 | 62.9 |
| a_noiou | `0c85e95` | `model.train_cfg.loss_weight_iou=0.0 model.test_cfg.iou_power=0.0` | nhánh IoU-quality | 29.61 (10) | 60.2 | 89.8 | 74.8 | 62.2 |
| a_noema | `0c85e95` | `train_cfg.ema_decay=0.0` | EMA | 29.77 (10) | 60.2 | 89.6 | 74.5 | 63.2 |
| a_nocrop | `0c85e95` | `dataset.crop_ratio=[1.0,1.0]` | random crop | 30.26 (10) | 60.2 | 89.8 | 74.9 | 62.4 |
| **a_kd** | `abl-kd` | – | InfoNCE → **KL distillation** (không che step) | **30.37** (10) | 59.8 | **91.1** | **75.2** | 60.8 |

Bỏ riêng từng phần đều không làm kết quả kém đi so với `s_base`. KL distillation (gợi ý của bạn) cho OmniCos cao hơn khoảng 1.3 và OmniCosC cao hơn 0.4, đổi lại OmniR@1 thấp hơn khoảng 2.

## 3. Ghép nhiều phần bỏ lại với nhau

Các run `f_*` đều dùng KL, không che step. Cột "Giữ lại" cho biết run đó còn giữ những phần nào trong số EMA, crop, `emb_loss`, nhánh IoU và `SegmentContext`.

| Lần chạy | Code | Override (`--set`) | Giữ lại | mAP (ep) | ArtAcc | OmniCos | OmniCosC | OmniR@1 |
|---|---|---|---|---|---|---|---|---|
| f_min | `ff66e7e` | – | không giữ phần nào | 28.42 (9) | 59.8 | 91.1 | 75.2 | 62.1 |
| f_ema | `0b329c0` | `dataset.crop_ratio=[1.0,1.0]` | EMA | 28.58 (10) | 59.6 | 91.2 | 75.5 | 61.7 |
| f_emacrop | `0b329c0` | – | EMA, crop | 29.05 (10) | 60.2 | 91.1 | 75.3 | 61.3 |
| s_seed2 | `463a9bf` (= `0b329c0`) | `init_rand_seed=42` | EMA, crop | 29.00 (9) | 59.4 | 91.1 | 75.3 | 61.2 |
| f_iou | `abl-kd-noctx` | `model.train_cfg.loss_weight_emb=0.0` | EMA, crop, IoU | 29.01 (9) | 61.2 | 91.1 | 75.3 | 61.2 |
| f_ctx | `abl-kd` | `model.train_cfg.loss_weight_emb=0.0 model.train_cfg.loss_weight_iou=0.0 model.test_cfg.iou_power=0.0` | EMA, crop, `SegmentContext` | 29.09 (9) | 60.2 | 91.1 | 75.4 | 59.9 |
| f_emb | `abl-emb` | – | EMA, crop, `emb_loss` | 29.65 (10) | 58.2 | 91.0 | 75.1 | 60.8 |
| **a_kd** | `abl-kd` | – | **tất cả** | **30.37** (10) | 59.8 | 91.1 | 75.2 | 60.8 |

Kết luận:
- Bỏ riêng từng phần thì không sao, nhưng bỏ cùng lúc thì mAP giảm. Bỏ cả 5 phần làm mAP giảm khoảng 2 điểm (`f_min` 28.42).
- Thêm lại riêng IoU hoặc riêng `SegmentContext` không lấy lại được gì. Thêm lại riêng `emb_loss` được khoảng +0.6 mAP. Phải có đủ cả ba (`emb_loss`, IoU, `SegmentContext`) mới đạt 30.37.
- Các chỉ số omni (OmniCos ~91, OmniCosC ~75, OmniR@1 ~61) gần như không phụ thuộc vào những phần này. Chúng chỉ phụ thuộc vào KL, span head và loss cos.

## 4. Tối ưu KL

Cả ba run dùng code `ff66e7e`. Các run bị dừng sớm vì đã kém rõ ràng.

| Lần chạy | Override (`--set`) | mAP (ep) | ArtAcc | OmniCos | OmniCosC | OmniR@1 | Ghi chú |
|---|---|---|---|---|---|---|---|
| f_min | – (nhiệt độ 0.05, trọng số KL 0.2) | 28.42 | 59.8 | 91.1 | 75.2 | 62.1 | |
| k_t02 | `model.train_cfg.omni_temperature=0.02` | 28.43 | 58.4 | 91.0 | 73.6 | **66.4** | OmniR@1 +4, OmniCosC −1.6 |
| k_t10 | `model.train_cfg.omni_temperature=0.1` | 21.89 (3) | 41.8 | 89.2 | 64.8 | 30.5 | dừng ở ep 3 |
| k_kd1 | `model.train_cfg.loss_weight_omni_kd=1.0` | 20.71 (2) | 30.2 | 82.3 | 61.9 | 47.8 | OmniCos tụt, dừng ở ep 2 |

Giữ nhiệt độ 0.05 và trọng số KL 0.2. Nhiệt độ 0.02 chỉ đáng dùng nếu ưu tiên OmniR@1 hơn OmniCosC.

## 5. Model cuối (`95d9592`, giống hệt phép tính của `a_kd`)

Đã kiểm tra: một bước train trên cùng batch cho ra loss giống nhau đến 6 chữ số thập phân. Hai bản chỉ khác tên khoá (`nce` đổi thành `kd`).

| mAP | ArtAcc | OmniCos | OmniCosC | OmniR@1 |
|---|---|---|---|---|
| 30.37 | 59.8 | 91.1 | 75.2 | 60.8 |

Config đầy đủ (`libs/core/config.py`, không cần `--set`):

```
dataset: max_seq_len 256, trunc_thresh 0.5, crop_ratio [0.9, 1.0], num_classes 48
model:   num_articles 617, pool_queries 8, omni_span_hidden 4096, omni_head_dim 3584, backbone convTransformer
train_cfg: loss_weight_reg 1.0, loss_weight_iou 1.0, loss_weight_emb 0.2, loss_weight_span 0.2,
           loss_weight_omni_kd 0.2, loss_weight_omni_cos 2.0, omni_temperature 0.05, loss_weight_article 0.5
test_cfg:  iou_power 0.3, nms_sigma 0.75, max_seg_num 200, article_soft False (True khi article không cho sẵn)
train:     batch 32, lr 3e-4, clip_proj_lr 3e-3, weight decay 1e-4, 10 + 2 epoch warmup, EMA 0.999
```

Những gì đã xoá trong ngày (tổng cộng khoảng 210 dòng code ít hơn so với `6f7292b`):
- Hai backbone `convPooler` (TemporalMaxer) và `dynE` (DyFADet).
- Bank omni và mining.
- `pred_article` chọn cứng, `article_detach`, `nmax`, `center_sample_radius`, `train_fraction`, `DropPath`.
- Các cờ bật/tắt của artc, article head, `article_learned`, span head.
- `seg_key`, cột positive và việc che cùng step trong loss omni.
- Bước kiểm tra step_id của omni, filter annotation dài 0, filter >48 đoạn, filter duration, `filtering()`. Thay bằng một quy tắc duy nhất: giữ đoạn bắt đầu trước khi feature kết thúc và cắt đuôi.

## 6. Text của OmniRetriever (thử ngày 08–09/11, không giữ lại)

Trích xuất:
- `tools/omni_step_text.py` chạy `encode_text("<activity>: <headline>")` cho 4958 step, adapter `omniretriever-7b`, bf16.
- Trên Colab L4 mất 70 giây. Kết quả lưu ở HF `htstep-text-feats/text_feats/step_emb_omni.npz` (3584 chiều).

**Đo zero-shot với checkpoint pc_v6** (đoạn GT cho sẵn):
- Chọn step trong article:
  - InternVideo2: 63.15.
  - Text omni so với AV của teacher: 59.35.
  - Text omni so với span head: 52.41.
  - Ghép InternVideo2 + omni: 65.00 (+1.85 Acc, không đổi mAP vì mAP article grounding không so sánh step).
- Đoán article (510 video):
  - Head: 61.57.
  - Ghép thêm omni của cả video tính sau artc thì lên 66.08, nhưng **đây là rò rỉ đáp án**, vì artc đã nhận text của article đúng.
  - Tính trước artc thì không tăng (61.76). Ngay cả mức trần dùng teacher trên đoạn GT cũng chỉ +2.8.

**Train với text omni** (code `95d9592`, mốc = `a_kd`):

| Lần chạy | Máy | Thay đổi | mAP (ep) | Acc | ArtAcc | OmniCos | OmniCosC | OmniR@1 |
|---|---|---|---|---|---|---|---|---|
| a_kd | A100 | – (mốc) | 30.37 (10) | 68.5 | 59.8 | 91.1 | 75.2 | 60.8 |
| t_omni | L4 | `dataset.step_emb_file=./datasets/step_emb_omni.npz model.clip_dim=3584` (text omni thay InternVideo2 ở mọi chỗ) | 29.85 (10) | 67.6 | **61.2** | 91.2 | 75.4 | 62.2 |
| t_both | PC | `dataset.step_emb_file=./datasets/step_emb_both.npz model.clip_dim=4096` (ghép InternVideo2 và omni) | 30.55 (8) | 68.7 | 59.6 | 91.1 | 75.2 | 62.1 |
| t_align | L4 | nhánh `abl-omnitext` (đã xoá): thêm loss span head chọn text omni của step, trọng số 0.2 | 24.40 (3) | 55.0 | 47.8 | 84.3 | 65.5 | 35.9 |

`t_align` bị dừng ở epoch 3, vì các chỉ số omni tụt 5–10 điểm.

**Kết luận:** không cách nào cải thiện rõ ràng, nên giữ text InternVideo2 như cũ.
- `t_omni` đổi được +1.4 ArtAcc nhưng mất 0.5 mAP.
- `t_both` ngang mốc.

File text omni vẫn còn trên HF và `tools/omni_step_text.py` vẫn được giữ, phòng khi cần dùng lại.
