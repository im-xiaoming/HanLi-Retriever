# Thí nghiệm HT-Step

## Thiết lập chung

- **Đánh giá:** mAP article grounding chính thức của HT-Step (tIoU 0.3–0.7).
  - Chấm trên 505 video val_seen có feature.
- **Train:** batch 32, lr 3e-4, 12 epoch. Checkpoint được chọn theo mAP.
- **Dữ liệu train:** 4810 / 17472 video.

## 1. Baseline

Mô hình dự đoán activity và caption trong toàn bộ caption pool.

| Thay đổi | mAP |
|---|---|
| **Baseline** | **25.70** |
| `max_seq_len` 512 | 25.66 |
| Backbone sâu hơn (2,6,5) | 24.81 |
| Crop mạnh hơn (0.7–1.0) | 24.78 |
| Dropout 0.1 | 24.68 |
| Model to hơn (dim 768) | 24.54 |
| lr 1e-4 | 24.34 |
| Nửa dữ liệu train | 22.76 |


## 2. Article conditioning (artc)

Mô hình cho biết trước activity + variant, sau đó nhiệm vụ của nó phải dự đoán caption phù hợp cho từng segment.

| Thay đổi | mAP |
|---|---|
| artc | 30.32 |
| artc + backbone sâu hơn | 30.11 |
| **artc + `max_seq_len` 512** | **30.62** |


| Mô hình trong bài báo | mAP |
|---|---|
| ActionFormer | 25.4 |
| ActionFormer-T | 26.3 |
| Mô hình tốt nhất | 30.2 |