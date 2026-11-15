# Tổng hợp các run, 15/11/2026


- **Tập đánh giá:** val_seen, 510 video.

## Thang đo

### Dữ liệu đánh giá

| | Số lượng |
|---|---|
| Video val_seen có feature | 510 |
| Đoạn GT (mỗi đoạn là một bước của công thức) | 2990 |
| Đoạn GT có vector teacher OmniRetriever | 2990 |
| Activity | 120 |
| Article (công thức) | 146, trong tổng số 617 của taxonomy |

- Mỗi epoch, model EMA được chấm trên toàn bộ val_seen. Code nằm ở `benchmark/validate.py` và `benchmark/metrics.py`.
- Mọi thang đo đều nhân 100, cao hơn là tốt hơn.

### Tóm tắt

| Thang đo | Đo cái gì | Head chịu trách nhiệm |
|---|---|---|
| mAP | Phát hiện đúng vị trí các bước trong video (article grounding của HT-Step) | event, boundary, embed |
| R@0.5, mIoU | Đoạn dự đoán trùng thời gian với đoạn GT đến mức nào | event, boundary |
| Acc | Đoạn đã khớp thời gian có được gán đúng bước không | embed, SegmentContext |
| ArtAcc | Đoán đúng công thức (article) của video | article head |
| OmniCosC | Vector omni của model gần vector OmniRetriever đến mức nào, chỉ tính phần riêng của từng đoạn | omni span head (từ run10: span + point) |
| OmniR@1 | Từ vector omni có tìm lại đúng đoạn của nó trong toàn bộ đoạn val không | như OmniCosC |

### mAP: article grounding (thang đo chính)

Cách tính giống script chính thức của HT-Step `eval_article_grounding.py`.

1. **Lọc dự đoán.** Model đưa ra tối đa 200 đoạn cho mỗi video, mỗi đoạn có điểm và một bước. Bỏ những đoạn có bước không thuộc công thức của video, hoặc bước không phải bước GT của video nào trong val.
2. **Gom theo activity.** Mỗi activity xếp mọi đoạn dự đoán của các video thuộc activity đó theo điểm, từ cao xuống thấp.
3. **Ghép với GT.** Lần lượt từng đoạn dự đoán, chọn đoạn GT **cùng video**, chưa được ghép, có tIoU cao nhất. Nếu tIoU ≥ ngưỡng thì đoạn đó là đúng (TP), không thì là sai. **Bước dự đoán không được so với bước GT**: chỉ cần đúng vị trí. Bước chỉ ảnh hưởng tới việc lọc ở bước 1.
4. **AP.** Tính AP nội suy kiểu VOC 2011 từ dãy TP. Mẫu số là tổng số đoạn GT của activity.
5. **Trung bình.** Lấy trung bình trên 5 ngưỡng tIoU (0.3, 0.4, 0.5, 0.6, 0.7) và trên 120 activity.


### R@0.5 và mIoU: độ khớp thời gian

1. Với mỗi video có n đoạn GT, lấy n đoạn dự đoán có điểm cao nhất.
2. Với mỗi đoạn GT, lấy IoU cao nhất giữa nó và n đoạn dự đoán đó.
3. Trên cả 2990 đoạn GT:
   - **R@0.5** = tỉ lệ đoạn GT có IoU cao nhất ≥ 0.5.
   - **mIoU** = trung bình của IoU cao nhất.


### Acc: gán đúng bước

1. Dùng cùng cách ghép như R@0.5: mỗi đoạn GT ghép với đoạn dự đoán có IoU cao nhất trong top-n.
2. Chỉ xét các đoạn GT mà IoU đó ≥ 0.3.
3. **Acc** = tỉ lệ đoạn có bước dự đoán trùng bước GT.


### ArtAcc: đoán đúng công thức

- Article head chấm điểm cả 617 công thức cho mỗi video. Điểm là cosine giữa vector video và vector công thức, trong đó vector công thức là text của các bước cộng một vector học được.
- **ArtAcc** = tỉ lệ video có công thức điểm cao nhất trùng công thức thật, trên 510 video.
- Mỗi video chiếm 0.2 điểm. Dao động giữa các run (khoảng ±1.5) tương đương 7–8 video.
- ArtAcc chỉ ảnh hưởng tới mAP và Acc khi bật `article_soft` (công thức không được cho sẵn). Ở các run trong report này, chế độ đó tắt.

### OmniCosC: độ gần với teacher, đã trừ phần chung

- **Teacher** t: vector OmniRetriever của từng đoạn GT, 3584 chiều, đã chuẩn hoá L2.
- **Student** s: vector omni do model dự đoán cho đoạn đó, khi biết trước thời điểm bắt đầu và kết thúc.
- **Công thức:** OmniCosC = 100 × trung bình trên 2990 đoạn của cos(s − μ, t − μ), với μ là trung bình teacher của tập **train**.
- **Vì sao phải trừ μ:** mọi teacher có chung một phần lớn. Chỉ cần dự đoán μ cho mọi đoạn đã được cosine thường khoảng 80. Trừ μ đi thì chỉ còn phần riêng của từng đoạn được chấm: dự đoán μ cho mọi đoạn được khoảng 0, còn 100 là trùng khít.

### OmniR@1: tìm lại đúng đoạn bằng vector omni

- Với mỗi student, so tích vô hướng với **cả 2990 teacher val**. Nếu teacher gần nhất chính là teacher của đoạn đó thì tính là trúng.
- **OmniR@1** = tỉ lệ trúng. Đoán ngẫu nhiên được khoảng 0.03.
- Không cần trừ μ: s·μ như nhau với mọi teacher, nên thứ hạng không đổi.

### Độ nhiễu

Mỗi run chỉ chạy một seed. Theo các run có cùng article head:
- ArtAcc dao động khoảng ±1.5.
- mAP chênh dưới khoảng 0.5 thì coi như ngang nhau.

## Các run

| Run | Máy, batch | Thay đổi so với run trước |
|---|---|---|
| run1 | 3060, 32 | Mốc so sánh. SegmentContext lấy trung bình đoạn; omni span head chỉ dùng tầng 0; loss omni gồm KD và cos. |
| run2 | 3060, 32 | SegmentContext dùng AttentionPool thay cho trung bình. |
| run3 | 3060, 32 | Thêm InfoNCE cho omni. |
| run4 | 3060, 32 | Omni span head lấy cả 6 tầng (level embedding + AttentionPool). |
| run7 | A100, **64** | Article head dùng AttentionPool trên cả 6 tầng. |
| run8 | A100, 32 | Article head quay về mean/max tầng 0 (feat0). |
| run9 | A100, 32 | Thêm skip connection cho omni (point head và span head). |

## Kết quả

Mỗi cột là giá trị tốt nhất trên mọi epoch, nên các cột có thể đến từ những epoch khác nhau. Cột "Epoch" là số epoch đã chạy, kèm epoch đạt mAP tốt nhất.

| Run | Epoch (mAP tốt nhất) | mAP | R@0.5 | mIoU | Acc | ArtAcc | OmniCosC | OmniR@1 |
|---|---|---|---|---|---|---|---|---|
| run1 | 11 (e10) | 30.83 | **36.29** | **35.06** | 67.94 | **61.57** | 71.84 | 51.84 |
| run2 | 12 (e11) | **30.89** | 35.85 | 34.66 | 68.37 | 59.22 | **71.86** | 52.37 |
| run3 | 4 (e4) | 26.21 | 32.41 | 32.50 | 57.02 | 55.69 | 66.76 | 54.18 |
| run4 | 11 (e9) | 29.57 | 35.25 | 34.59 | 67.93 | 60.98 | 68.03 | 66.25 |
| run7 | 11 (e9) | 28.59 | 33.95 | 33.46 | 69.00 | 56.67 | 67.63 | 67.09 |
| run8 | 11 (e9) | 29.98 | 35.59 | 34.62 | 69.89 | 61.18 | 68.15 | 66.69 |
| run9 | 13 (e11) | 30.17 | 35.32 | 34.66 | **70.33** | 58.82 | 64.62 | **67.53** |

## So sánh với các bài báo

Chỉ mAP và R@1 có số để so với bài báo khác. Acc, ArtAcc, OmniCosC và OmniR@1 là thang đo riêng của đề tài này, không bài nào báo cáo.

### Lưu ý khi so

- **Số video.** Các bài báo chấm trên đủ 600 video val_seen; video không có dự đoán bị tính là bỏ sót. Bảng kết quả ở trên chỉ chấm 510 video có feature, nên cao hơn.

### mAP article grounding

| Phương pháp | Cách học | Tập chấm | @0.3 | @0.5 | @0.7 | TB 0.3–0.7 |
|---|---|---|---|---|---|---|
| ActionFormer (HT-Step, Table 2) | có giám sát | test seen | | | | 25.4 |
| ActionFormer-T (HT-Step, Table 2) | có giám sát | test seen | 36.9 | 26.5 | 15.3 | 26.3 |
| ActionFormer-T + pre-train HowTo100M (HT-Step, Table 2) | có giám sát | test seen | 41.2 | 30.8 | 18.3 | 30.2 |
| NaSVA (chạy lại trong bài HSD) | học yếu | val_seen 600 | 15.02 | 5.70 | 1.11 | 6.89 |
| BaGLM, InternVL2.5-8B | không train, online | val_seen 600 | 15.90 | 6.70 | 1.50 | 7.70 |
| UniTime-7B (trong bài HSD) | temporal grounding | val_seen 600 | 19.92 | 11.48 | 5.52 | 12.27 |
| HSD, Qwen3-VL-8B | không train, online | val_seen 600 | 26.45 | 13.78 | 6.16 | 15.13 |
| **Đề tài** | có giám sát | val_seen 510 | 42.85 | 29.84 | 15.74 | 29.70 |

Theo mAP trung bình, đề tài cao hơn các phương pháp không train (HSD 15.13) và học yếu (NaSVA 6.89) khoảng 10–19 điểm. So với các phương pháp có giám sát của bài HT-Step, đề tài xấp xỉ ActionFormer-T (26.3) nhưng thấp hơn bản có pre-train HowTo100M (30.2), dù hai bên chấm trên hai tập khác nhau.


### R@1 step grounding

Một bước được tính là tìm thấy nếu thời điểm model chọn cho bước đó nằm trong một đoạn GT của bước.

| Phương pháp | Cách học | R@1 |
|---|---|---|
| TAN* | học yếu | 30.7 |
| VINA | học yếu | 39.1 |
| NaSVA | học yếu | 53.1 |
| NaSVA-Online | học yếu, online | 46.1 |
| BaGLM | không train, online | 57.4 (bài HSD chạy lại được 54.3) |
| HSD, Qwen2.5-VL-7B | không train, online | 55.8 |
| **Đề tài** (510 video) | có giám sát | 42.13 |


### Nguồn

- HT-Step: Afouras và cộng sự, "HT-Step: Aligning Instructional Articles with How-To Videos", NeurIPS 2023 Datasets and Benchmarks.
- BaGLM: Zanella và cộng sự, "Training-free Online Video Step Grounding", NeurIPS 2025, arXiv 2510.16989.
- HSD: "Hierarchical Step Decomposition for Training-free Online Video Step Grounding", arXiv 2610.40153. Các dòng NaSVA, UniTime và bảng R@1 lấy từ bài này.

## Nhận xét

### 1. InfoNCE cho omni: OmniR@1 tăng mạnh, OmniCosC giảm (run2 → run4)

| | run2 | run4 |
|---|---|---|
| OmniR@1 | 52.4 | 66.3 (**+14**) |
| OmniCosC | 71.9 | 68.0 (**−3.8**) |
| mAP | 30.89 | 29.57 (−1.3) |

- InfoNCE dạy student phân biệt đoạn này với đoạn khác. Nhưng nó không kéo student sát teacher, nên OmniCosC giảm.
- Từ run4 trở đi, OmniCosC luôn đạt đỉnh khoảng epoch 7 rồi giảm.

### 2. SegmentContext: AttentionPool không hơn trung bình (run1 → run2)

- mAP 30.83 → 30.89 và Acc 67.9 → 68.4 đều nằm trong mức nhiễu.
- ArtAcc giảm 2.3, cũng gần mức nhiễu.
- AttentionPool thêm khoảng 5M tham số mà không có cải thiện rõ.

### 3. Article head: feat0 là tốt nhất

| Article head | Run | ArtAcc |
|---|---|---|
| mean/max tầng 0 (feat0) | run1 / run4 / run8 | 61.6 / 61.0 / 61.2 |
| AttentionPool cả 6 tầng | run7 (batch 64) | 56.7 |

- Các tầng thô được downsample từ tầng 0, nên không mang thêm thông tin cho việc nhận ra món.

### 4. Batch 64 kém hơn batch 32

| | Batch 64 (run7) | Batch 32 (run8, run9) |
|---|---|---|
| mAP | 28.2–28.6 | 30.0–30.2 |
| Thời gian một epoch (train + eval) | ≈109 s | ≈116 s |


### 5. Skip connection (run8 → run9)

| | run8 | run9 | Chênh |
|---|---|---|---|
| mAP | 29.98 | 30.17 | +0.2 |
| Acc | 69.89 | 70.33 | +0.4 |
| OmniR@1 | 66.69 | 67.53 | +0.8 |
| ArtAcc | 61.18 | 58.82 | **−2.4** |
| OmniCosC | 68.15 | 64.62 | **−3.5** |

Run9 thay đổi hai thứ cùng lúc, nên chưa tách được thứ nào gây ra phần giảm. Nghi vấn:

- **ArtAcc giảm:** khởi tạo Xavier làm `article_learned` có nhiễu ngay từ đầu, không còn bằng 0. Nhánh residual của `ArticleCondition` (đứng ngay sau article head) cũng vậy.
- **OmniCosC giảm:** có thể do skip connection, hoặc do khởi tạo Xavier ở nhánh omni.
