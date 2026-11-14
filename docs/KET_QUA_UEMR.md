# UEMR: tìm bước trong video nấu ăn bằng một câu, có chỉ ra khoảnh khắc

Báo cáo ngày 2026-11-14, theo khung `docs/uemr_chot/UEMR_CHOT_04_BAO_CAO.md`. **Bản này chưa có benchmark.** Mục 7 và câu kết luận cuối để
trống, điền sau lượt thật của `notebooks/uemr/uemr_final_benchmark.ipynb`. Mọi số dưới đây là số **dev HT-Step**: chỉ dùng để chọn tham số,
không phải kết quả kiểm định.

Quy ước:
- Mọi số chép từ file kết quả trong `Fushion/results/uemr_final/` hoặc từ Nhật ký của file plan, có ghi nguồn. Repo Fushion, nhánh `htstep`.
- `01:L205` là dòng 205 của `Fushion/docs/uemr_chot/UEMR_CHOT_01_UNIAV_BEST.md`; tương tự `00:`, `02:`, `03:` cho các file 00, 02, 03.
- Hiệu tôi tự tính (không có CI) ghi "tính từ bảng".
- Mỗi giải thích ghi **[đã kiểm chứng]** (có số đo trực tiếp) hoặc **[giả thuyết]** (chưa đo).
- Không dùng và không so số YouCook2 của các lượt cũ (Athena, Omni ft YC2).
- Số thập phân viết dấu phẩy. Số R@ là phần trăm.

| | |
|---|---|
| Config khoá | `configs/uemr_final.json`, sha256 `c9840ffbc964a4b0ed70baee157213adaaa6526178316c80a8435e7d2fcdbde3`, commit `2d6f771` |
| UniAV | `GraceFeel/uniav-htstep-best`, `model_best.pth.tar` sha256 `9447de87…`, code HanLi-Retriever `77fc4fa` (không sửa) |
| Omni | WAVE-7B + LoRA `best` (`GraceFeel/omni_lora_htstep`), adapter sha256 `5e9899aa…`, code Omni-fix `ab772eb` |
| Kho dev | HT-Step `val_seen`, 510 video, 2.990 câu, protocol `a4ca43db` |
| Máy | Colab A100-SXM4-40GB (encode); chấm ở local CPU |

---

## 1. Tóm tắt

- **Benchmark (YouCook2, zero-shot): chưa chạy.** Chưa có câu kết luận theo bảng ghi trước (file 03 §8). Lượt SMOKE 5 video đã đi hết
  pipeline, mọi CHECK OK; số của lượt đó không được dùng.
- **Phương pháp chính, chọn trên dev: UEMR-Full** = (U) với β\* = 1,0 trên R8. Đoạn từ UniAV mới (R6) cộng đoạn lấp chỗ trống. Mỗi đoạn
  chấm bằng cos(câu, Omni) + cos(câu, vector UniAV).
- **Số dev chính** (R@1 multi-positive, CI 95 %; `results/uemr_final/dev/metrics.json`, `pairs.json`):
  - UEMR-Full đạt **63,34**; UEMR-Omni ((b) trên R8) đạt 59,60; một vector Omni / video (R0 (b)) đạt 49,00.
  - R8 − R0 (b): **+10,60 [+7,24, +14,08]**.
  - (U) − (b) trên R8: **+3,75 [+1,53, +5,98]**. CI lạc quan vì β\* cũng chọn trên dev.
  - UEMR-Lite (không chạy Omni 7B phía video) − R0 (b): **+9,57 [+6,13, +13,08]**.
  - Chưa tách được ranh giới UniAV khỏi việc có nhiều clip hơn: R8 − R2 (chia đều, cùng số clip) +1,77 [−0,53, +4,02], CI chứa 0.

## 2. Bài toán và vai trò dữ liệu

Người dùng gõ một câu mô tả một bước nấu ăn. Hệ thống phải (1) xếp video trong kho sao cho video có bước đó đứng đầu và (2) chỉ ra đoạn
trong video đó. Phần (2) gọi là VCMR (video corpus moment retrieval).

| Dữ liệu | Vai trò | Số |
|---|---|---|
| HT-Step train | train Omni `best` (đã có từ trước) và train lại UniAV (file 01) | UniAV 5.826 video; Omni 6.993 video (`00:L27`, `01:L20`) |
| HT-Step `val_seen` | **dev**: chọn θ, `L_min` / `L_fill`, β\*, phương pháp chính | 510 video, 2.990 câu, 930 text khác nhau (`KET_QUA_UEMR_HTSTEP.md` mục 2) |
| YouCook2 train + val | **benchmark**: không train, không chọn; khoá protocol trước; chấm một lần | 1.790 video có annotation, 1.500 có mp4 (đếm local, `03:L143-145`) |

Tách như vậy để số cuối không lạc quan. Cả hai model đều chọn checkpoint trên `val_seen`, và mọi tham số của UEMR cũng chọn trên đó, nên số
dev chắc chắn cao hơn thật. Benchmark là bộ không model nào thấy lúc train hay lúc chọn. **[đã kiểm chứng]** Trùng YouTube ID giữa
YouCook2 và train UniAV / train Omni / dev là 0 / 0 / 0 (`03:L145`).

Kho YouCook2 gồm cả split train: user quyết định ngày 2026-11-13, trước khi có bất kỳ số truy hồi nào. Vẫn là zero-shot, vì trong plan này
không model nào train trên YouCook2 (`03:L139-142`).

## 3. Phương pháp

**UEMR = Unified Event-aware Multi-vector Retrieval** (`00:L32-61`).
- *Unified*: UniAV được chưng cất từ chính Omni dùng để truy hồi (`best`), nên mọi vector nằm trong một không gian.
- *Event-aware*: video được cắt theo bước bởi UniAV.
- *Multi-vector*: một vector cho mỗi đoạn, chấm bằng MaxSim.

```text
video ──► đặc trưng K400 + BEATs (1 dòng / giây) ──► UniAV (article đoán top-1, không dùng nhãn)
                                                       │
                     ┌─────────────────────────────────┴──────────────────────┐
                     ▼                                                        ▼
         đề xuất đoạn ≥ θ = 0,402  (R6)                         span head trên đoạn bất kỳ
                     │                                                        │
                     ▼                                                        │
   lấp chỗ trống: khoảng hở ≥ L_min = 5 s cắt thành                           │
   đoạn dài L_fill = 10 s  (R8 = R6 + đoạn lấp)                               │
                     │                                                        │
          ┌──────────┴───────────┐                                            │
          ▼                      ▼                                            ▼
   e_i = Omni best AV      (8 khung, tiếng 8 s)                   h_i = vector UniAV, cùng không gian best
          │                                                                   │
          └────────────► s_i = cos(t, e_i) + β·cos(t, h_i) ◄──────────────────┘
                                      ▲
                     câu hỏi ──► t = Omni best encode_text

   điểm video S(q, V) = max_i s_i        khoảnh khắc trả về = đoạn argmax_i s_i
```

**Bốn biến thể** (cùng chỉ mục R8, khác cách chấm đoạn):

| Biến thể | Cột | Điểm của đoạn | Omni 7B chạy phía video? |
|---|---|---|---|
| UEMR-Omni | (b) | cos(t, e_i) | có, mọi đoạn |
| UEMR-Lite | (S) | cos(t, h_i) | **không** (Omni chỉ mã hoá câu hỏi) |
| **UEMR-Full** | (U) | cos(t, e_i) + β\*·cos(t, h_i), β\* = 1,0 | có |
| UEMR-Hybrid | C1 | đoạn R6: Z_b(cos(t, e_i)); đoạn lấp: Z_S(cos(t, h_i)) | chỉ đoạn R6 |

Z(·) là z-score không dùng nhãn: μ, σ của cos trên mọi cặp (câu, đoạn R8) của kho, tính lại trên từng kho (`configs/uemr_final.json`, khoá
`rules.zscore`).

**Các dòng (cách cắt) để so** (`00:L130-141`):

| Dòng | Đoạn |
|---|---|
| R0 | cả video, 1 vector |
| R1 | trung bình (rồi L2-norm) các vector R6, 1 vector |
| R2 | đoạn đều dài L, L chọn để tổng clip bằng R8 (dev: L = 11,89 s) |
| R3 | K_pred đoạn đều, K_pred = số đoạn R6 của video |
| R6 | đoạn UniAV, article đoán, ngưỡng θ |
| R8 | R6 + lấp chỗ trống |
| R7 | đoạn GT (oracle, cận trên) |

**Ngữ cảnh đi vào qua vector trò, không qua Context Fusion.** Backbone UniAV nhìn cả video. `article_cond` mang thông tin task. Vì vậy
`h_i` của một đoạn ngắn vẫn mang ngữ cảnh của cả video. **[đã kiểm chứng]** `h_i` phụ thuộc mạnh vào article: thay article đoán bằng
article ngẫu nhiên làm (S) trên R8 giảm 22,58 điểm (mục 6.4, C3).

## 4. Huấn luyện UniAV với thầy `best` (file 01)

**Dữ liệu thầy** (`01:L168-191`, log `uemr_chot_01_teacher_best_A{1,2}.pdf`):
- Vector thầy = Omni `best` AV của từng đoạn GT, tính bằng đúng đường encode của cột (b).
- CHECK đường encode: clip cắt sẵn so với `val_embeds_best.npz`, 20 clip GT val: cos trung bình 0,99991, nhỏ nhất 0,99986 (ngưỡng 0,999).
- Train: 5.431 file, 33.445 đoạn có vector, **2.581 đoạn thiếu (7,16 %)**. Nguyên nhân: 2.321 đoạn thuộc 361 video không có mp4, 89 mp4 không
  tải hoặc chuyển được, 171 encode lỗi. Ngưỡng đoạn thiếu đổi 5 % → 8 % theo quyết định của user (`01:L146-148`).
- Val: 510 file, 2.990 đoạn, 0 thiếu.
- Encode 0,236 s/clip. Dataset `GraceFeel/htstep-omni-av-best`, `config.json` sha256 `f9abbce3…`.
- **[đã kiểm chứng]** Đoạn thiếu bị bỏ khỏi cả hai loss chưng cất bằng mặt nạ `omni_ok`; các loss khác không đổi (`01:L90-92`). Thầy cũ
  cũng thiếu đúng 361 video này (`01:L119-121`).

**Train** (`01:L197-219`, log `uemr_chot_01_train_uniav_smoke_{true,false}.pdf`):
- `python train.py --output htstep_best`, config mặc định của HanLi `77fc4fa`. Chỉ đổi thầy.
- 11 epoch (early stop), 25,1 phút trên A100. Chọn epoch 9 theo mAP trên `val_seen`.

**Độ giống thầy và chất lượng cắt** (510 video dev; `results/uemr_final/uniav_best_checks.json`, `01:L233-247`):

| Chỉ số | UniAV cũ (thầy cũ) | UniAV mới (thầy `best`) |
|---|---|---|
| mAP@[0,3:0,7], article GT | 29,70 | 30,31 |
| mAP@[0,3:0,7], article đoán | 19,55 | **20,14** |
| ArtAcc (article đoán đúng) | 60,20 | 59,61 |
| Recall bước GT, tIoU 0,3 / 0,5 / 0,7 (θ 0,402) | 42,17 / 31,74 / 20,50 | 43,58 / 33,04 / 21,57 |
| Số đoạn R6 trung bình / video | 5,884 | 5,888 |
| OmniCos / OmniCosC / OmniR@1, đoạn GT, article GT | 91,06 / 75,05 / 59,72 (1.080 đoạn) | 88,08 / 72,20 / 52,04 (2.990 đoạn) |
| OmniCos / OmniCosC / OmniR@1, đoạn GT, article đoán | chưa đo | 87,88 / 71,71 / 50,57 |

- Hàng Omni của cũ và mới không so trực tiếp được: thầy khác, tập đoạn khác.
- Theo luật ghi trước (mAP article đoán cao hơn), **đoạn R6 lấy từ UniAV mới**; `h_i` luôn từ UniAV mới.
- Chất lượng cắt vẫn là chỗ yếu. **[đã kiểm chứng]** 56,42 % bước GT không được đoạn R6 nào phủ ở tIoU ≥ 0,3 (= 100 − 43,58, tính từ bảng;
  cùng số 1.687 / 2.990 câu ở `groups.json`).
- Lúc train, UniAV luôn được cho article GT. Lúc dùng, nó phải đoán: mAP giảm từ 30,31 xuống 20,14.

## 5. Kiểm tra độ tin cậy

| Rủi ro | Cách tránh / kiểm | Kết quả |
|---|---|---|
| Rò rỉ article (UniAV có điều kiện task; article GT là nhãn) | Adapter `uemr_htstep/uniav_htstep.py` chỉ dùng article top-1 do model đoán. Kiểm: đổi `cand` GT sang article khác thì đoạn, điểm, nhãn và vector span head ở chế độ chính không đổi | **[đã kiểm chứng]** 20 / 20 video không đổi (`uniav_best_checks.json`, `new.omni.cand_check`) |
| Benchmark: bước giữ chỗ trong annotation giả (HanLi cần ≥ 1 annotation) có thể lọt vào đầu ra | Ô 6 đổi bước giữ chỗ rồi so | **[đã kiểm chứng]** ở local trên 3 video dev (trùng hệt lượt dev) và SMOKE 5 video: không đổi (`03:L164-166`; log SMOKE dòng 1391) |
| Tái lập lượt trước | Tính lại R8 (b) của lượt dev 1 từ cache | **[đã kiểm chứng]** R@1 multi 57,99, strict 21,81; mảng hạng trùng hệt (`02:L118-119`) |
| Lỗi code chấm | CHECK β = 0 cho đúng mảng hạng (b); `final_dev score` trên dữ liệu dev 1 khớp `metrics_best.json` | **[đã kiểm chứng]** OK; 12 / 12 hàng sai khác 0 (`02:L156-157`, `02:L171`) |
| Kho giao train | So danh sách ID train thật | **[đã kiểm chứng]** dev: 0 video train; YouCook2: 0 / 0 / 0 |
| Chọn nhiều lần trên dev | Ghi mọi cấu hình đã chấm vào `tried.jsonl` | 14 dòng: 6 cặp (`L_min`, `L_fill`), 1 lựa chọn, 6 β, 1 lựa chọn β\* |
| Chấm benchmark nhiều lần | Protocol in sha256 và đẩy HF trước ô chấm; ô chấm kiểm sha256; không chấm lại nếu đã có `criteria.json`; `run_manifest.json` đếm lượt | Code + SMOKE: "số lượt chấm trong run_manifest: 1". Lượt thật: **chưa chạy** |
| Đặc trưng benchmark khác đặc trưng train UniAV | Trích lại 5 video HT-Step, so với `htstep-val-seen-feats` | SMOKE: cos k768 0,99999958–0,99999976, a768 1,0 (ngưỡng 0,99) (log SMOKE dòng 2400-2430) |

## 6. Dev HT-Step (chỉ để chọn, không phải kết quả kiểm định)

### 6.1 Các lựa chọn, theo luật ghi trước (file 02 §3, §7)

| Tham số | Luật | Giá trị | Nguồn |
|---|---|---|---|
| θ | số đoạn R6 trung bình / video = số bước trung bình / video của train HT-Step (5,910) | **0,402** (K trung bình 5,888) | `segments/rows.json` |
| `L_min`, `L_fill` | R@1 multi cao nhất của (S) trên R8, lưới {3, 5} × {8, 10, 15} s | **(5, 10)** | `tried.jsonl` dòng 1-7 |
| L của R2 | tổng clip R2 = tổng clip R8 | 11,89 s (16.939 clip) | `tried.jsonl` dòng 7 |
| β\* | R@1 multi cao nhất của (U)_β trên R8, lưới {0; 0,1; 0,2; 0,3; 0,5; 1,0} | **1,0** | `metrics.json` khoá `beta_star` |
| Phương pháp chính | UEMR-Full nếu β\* > 0 và cận dưới CI ((U) − (b)) trên R8 > 0 | **UEMR-Full** | `metrics.json` khoá `main_method` |

Lưới `L_min` / `L_fill`, (S) R@1 multi trên R8: (3, 8) 58,09; (3, 10) 58,23; (3, 15) 57,56; (5, 8) 58,16; **(5, 10) 58,56**; (5, 15) 57,89.

Lưới β, (U)_β R@1 multi trên R8: 0 → 59,60; 0,1 → 61,44; 0,2 → 61,30; 0,3 → 62,44; 0,5 → 63,21; **1,0 → 63,34**. β\* nằm ở biên trên của
lưới, và đường gần như phẳng từ 0,5 đến 1,0. Lưới không mở rộng vì plan cố định lưới.

Tổng số cấu hình đã chấm trên dev: 14 dòng `tried.jsonl` (12 cấu hình có số, 2 dòng ghi lựa chọn).

### 6.2 Bảng chính, multi-positive

R@1 / R@5 / VCMR R@1 (`results/uemr_final/dev/metrics.json`, khoá `table`). VCMR của R0, R1 là "–" vì không có đoạn.

| Dòng | vec / video | (b) UEMR-Omni | (S) UEMR-Lite | (U) β = 1, UEMR-Full | C1 UEMR-Hybrid |
|---|---|---|---|---|---|
| R0 | 1,0 | 49,00 / 80,84 / – | 50,87 / 84,52 / – | 55,08 / 88,09 / – | |
| R1 | 1,0 | 58,13 / 86,69 / – | 55,05 / 86,66 / – | 58,86 / 87,63 / – | |
| R2 | 33,2 | 57,83 / 89,16 / 10,60 | 56,52 / 88,09 / 8,70 | 62,04 / 90,90 / 11,77 | |
| R3 | 5,9 | 53,38 / 87,83 / 4,41 | 55,89 / 87,39 / 5,72 | 57,99 / 89,03 / 6,42 | |
| R6 | 5,9 | 57,49 / 87,36 / 24,48 | 58,39 / 85,95 / 22,07 | 61,57 / 88,46 / 26,45 | |
| **R8** | 33,2 | 59,60 / 90,07 / 15,92 | 58,56 / 88,19 / 15,05 | **63,34 / 90,90 / 21,34** | 58,90 / 89,77 / 15,75 |
| R7 (GT, trần) | 5,9 | 65,05 / 92,07 / 56,05 | 66,62 / 90,60 / 55,52 | 70,54 / 93,88 / 59,63 | |

Cặp kèm CI 95 % (R@1 multi, cột (b) trừ khi ghi khác; `pairs.json`, mục `mode = multi`):

| Cặp | A − B | CI 95 % | CI chứa 0? |
|---|---|---|---|
| R8 − R0 | +10,60 | [+7,24, +14,08] | không |
| R6 − R0 | +8,49 | [+4,90, +11,90] | không |
| R8 − R3 | +6,22 | [+3,74, +8,73] | không |
| R6 − R3 | +4,11 | [+1,19, +6,91] | không |
| R8 − R2 | +1,77 | [−0,53, +4,02] | **có** |
| R6 − R1 | −0,64 | [−3,40, +2,24] | **có** |
| R7 − R8 | +5,45 | [+2,75, +7,95] | không |
| (U) − (b) trên R8 | +3,75 | [+1,53, +5,98] | không |
| (S) R8 − (b) R0 | +9,57 | [+6,13, +13,08] | không |
| C1 R8 − (b) R6 | +1,40 | [−0,78, +3,70] | **có** |
| VCMR R6 − R3 | +20,07 | [+17,55, +22,59] | không |

Đọc bảng:
- **[đã kiểm chứng]** Cắt theo sự kiện (R6, R8) hơn rõ một vector / video và cắt đều cùng số đoạn (R3).
- **[đã kiểm chứng]** R8 chưa hơn R2 (chia đều, cùng số clip) có ý nghĩa thống kê. Trên dev, chưa tách được lợi ích của ranh giới UniAV khỏi
  việc có nhiều clip hơn.
- **[đã kiểm chứng]** Về định vị khoảnh khắc, ranh giới UniAV giúp rõ: VCMR R6 − R3 +20,07.
- **[đã kiểm chứng]** VCMR của R8 (15,92) thấp hơn R6 (24,48), dù R@1 cao hơn. Đoạn lấp dài 10 s, thường không khớp ranh giới bước ở
  tIoU ≥ 0,5.
- **[đã kiểm chứng]** Cộng (S) vào (b) giúp ở mọi dòng. Hiệu (U) − (b) trong bảng (tính từ bảng): R0 +6,08, R1 +0,73, R2 +4,21, R3 +4,61,
  R6 +4,08, R8 +3,74, R7 +5,49. CI chỉ có cho R8.
- R7 − R8 +5,45: còn nhiều chỗ để cải thiện bằng cắt tốt hơn.

### 6.3 Strict (phụ lục)

R@1 / R@5 / VCMR R@1, chỉ video gốc của câu là đúng (`metrics.json`, `mode = strict`):

| Dòng | (b) | (S) | (U) β = 1 | C1 |
|---|---|---|---|---|
| R0 | 18,29 / 51,77 / – | 19,46 / 62,17 / – | 20,84 / 65,12 / – | |
| R1 | 22,94 / 62,98 / – | 21,74 / 63,68 / – | 23,65 / 65,75 / – | |
| R2 | 22,44 / 67,93 / 2,51 | 21,14 / 67,22 / 2,07 | 24,18 / 72,04 / 2,74 | |
| R3 | 20,27 / 64,52 / 1,07 | 21,54 / 64,01 / 1,34 | 22,01 / 67,76 / 1,47 | |
| R6 | 22,47 / 62,94 / 6,05 | 21,51 / 64,18 / 5,48 | 23,61 / 66,89 / 6,59 | |
| R8 | 22,51 / 69,03 / 4,18 | 21,64 / 68,33 / 3,81 | 24,35 / 72,14 / 5,38 | 21,94 / 69,63 / 3,95 |
| R7 | 26,12 / 73,18 / 14,38 | 25,79 / 73,31 / 13,85 | 28,60 / 77,66 / 15,82 | |

Cặp strict R@1 (`pairs.json`, `mode = strict`): R8 − R0 +4,21 [+0,96, +7,55]; R8 − R3 +2,24 [−0,72, +5,30]; R8 − R2 +0,07
[−2,62, +2,57]; (U) − (b) trên R8 +1,84 [−0,31, +4,11]; (S) R8 − (b) R0 +3,34 [−0,27, +6,67]; VCMR R6 − R3 +4,98 [+4,07, +6,06].

Ở strict, (U) − (b) có CI chứa 0. Strict thấp vì 83,21 % câu có text xuất hiện ở ≥ 2 video: video khác cùng bước bị tính là sai.

### 6.4 Phân tích nhóm (không tốn GPU; `results/uemr_final/dev/groups.json`, multi trừ khi ghi khác)

- **(i) Bước bị R6 bỏ sót** (tIoU < 0,3, 1.687 câu): R6 − R3 +1,48 [−1,75, +4,88]; R8 − R6 **+5,69 [+2,36, +8,84]**.
  **Bước R6 bắt được** (1.303 câu): R6 − R3 **+7,52 [+3,87, +11,14]**; R8 − R6 −2,53 [−5,81, +0,80].
  **[đã kiểm chứng]** Đoạn lấp giúp đúng ở chỗ R6 bỏ sót. Ranh giới UniAV giúp ở chỗ nó bắt được.
- **(ii) Text trùng ở ≥ 2 video** (2.488 câu): R8 − R0 strict +4,30 [+0,84, +7,76], multi +11,98 [+8,36, +15,77].
  **Text duy nhất** (502 câu): +3,78 [−3,04, +10,30] (strict = multi). Nhóm này nhỏ nên CI rộng.
- **(iii) Tỉ lệ đoạn đứng đầu là đoạn lấp** (R8): (b) 65,3 %, (S) 56,6 %, (U) 52,3 %, C1 51,7 %. Đoạn lấp chiếm 13.936 / 16.939 = 82,3 %
  số đoạn R8 (tính từ bảng, `cost.json`).
- **(iv) C3**, (S) trên R8 với article đoán − article ngẫu nhiên ≠ article đoán (seed 0): **+22,58 [+19,71, +25,57]**. (S) với article
  ngẫu nhiên còn 35,99 (`metrics.json`, cột `S_rand`).
- **(v)** Spearman giữa hạng (b) và hạng (S) trên R8: 0,587. hit@1: cả hai đúng 1.323, chỉ (b) đúng 459, chỉ (S) đúng 428, cả hai sai 780.
  **[giả thuyết]** Hai cột sai ở chỗ khác nhau, nên cộng lại (U) hơn từng cột. Chưa có phép kiểm nhân quả.

### 6.5 Đặt cạnh lượt dev 1 (UniAV thầy cũ; chỉ tham chiếu)

R@1 multi, lượt dev 1 → lượt này (`02:L215-217`): (b) R8 57,99 → 59,60; (b) R6 55,35 → 57,49; (b) R3 55,69 → 53,38; (b) R2 58,16 → 57,83;
(S) R8 57,56 → 58,56; (U) R8 61,34 (β = 0,1, text thầy cũ) → 63,34 (β = 1, text `best`).

Hai lượt khác cả đoạn lẫn text của (S), nên không so trực tiếp. **[giả thuyết]** Đổi thầy sang `best` làm (S) cùng thang với (b) (μ 0,0565
so với 0,0530; lượt dev 1 có μ ≈ 0,60), nên β lớn mới dùng được.

## 7. Benchmark

**Chưa có.** Phần này điền sau lượt thật (`SMOKE = False`) của `notebooks/uemr/uemr_final_benchmark.ipynb`, đầu ra trên HF
`GraceFeel/uemr-bench-yc2`.

### 7.1 YouCook2 (train + val)

| Mục | Nội dung sẽ điền | Nguồn sẽ dùng |
|---|---|---|
| Kiểm kê | annotation → có mp4 → có tiếng → đọc được → không trùng | ô 3, `inventory.json` |
| Câu hỏi | số câu, số text khác nhau, % text trùng ở ≥ 2 video | ô 4 |
| Chất lượng cắt | recall tIoU 0,3 / 0,5 / 0,7, % bỏ sót, độ phủ, K_pred so với K_GT, phân bố article HT-Step được đoán | `uniav_quality.json` |
| Bảng chính | 7 dòng × 4 cột, multi + strict; pt so với `best` trên R0, R8 | `metrics.json` |
| Tiêu chí K1–K6 | cận dưới CI từng tiêu chí, đạt / không | `criteria.json`, `pairs.json` |
| Câu kết luận | đúng một dòng của bảng file 03 §8 | `criteria.json` |

Đã biết từ trước khi chạy:
- Kho: 1.790 video có annotation (1.333 train + 457 val). Theo đếm local, 1.500 video có mp4. Số video thật trên Drive chưa xác nhận.
- Luật không nhãn tính lại trên kho: L của R2, μ / σ của z-score. Video không có đề xuất UniAV nào thì không có đoạn R6 / R3 (`03:L178-179`).
- Ước lượng (chưa đo): ~165.000 clip Omni ≈ 11–15 giờ A100; ô chấm cần ~25 GB RAM (`03:L153`, `03:L182`).

**Câu kết luận (file 03 §8): (chờ benchmark)**

## 8. Chi phí

Phía video, trên kho dev (`results/uemr_final/dev/cost.json`; tốc độ từ `02:L144`, `02:L166`):

| Biến thể | Clip Omni / video | Vector lưu / video | Chỉ mục cả kho dev, fp16 (tính từ bảng) | Thời gian Omni / video (tính từ bảng) |
|---|---|---|---|---|
| UEMR-Omni | 33,2 | 33,2 `e_i` | 121,4 MB | ≈ 8,0 s |
| UEMR-Full | 33,2 | 33,2 `e_i` + 33,2 `h_i` | 242,8 MB | ≈ 8,0 s |
| UEMR-Hybrid | 5,9 | 5,9 `e_i` + 27,3 `h_i` | 121,4 MB | ≈ 1,4 s |
| UEMR-Lite | 0 | 33,2 `h_i` | 121,4 MB | 0 |

- Kích thước chỉ mục = số vector × 3.584 chiều × 2 byte. Giả định `h_i` cũng lưu fp16; config chỉ ghi fp16 cho vector Omni.
  Ví dụ: 16.939 × 3.584 × 2 = 121,4 MB.
- Thời gian Omni = clip / video × 0,24 s/clip (Omni chạy 0,231–0,246 s/clip trên A100).
- Mọi biến thể đều cần thêm: UniAV forward + span head 0,9–1,0 s/video (CPU local), và trích đặc trưng K400 + BEATs.
- Tốc độ trích đặc trưng **chưa đo**. Lượt SMOKE in 6,6 video-s/s trên 5 video, tính cả nạp model, nên chưa đủ để ước lượng.
- Phía câu hỏi, mọi biến thể đều chạy Omni `encode_text` một lần mỗi câu.

## 9. Ví dụ định tính

**Chưa làm.** Theo plan cần 3–4 câu lấy từ thứ hạng đã lưu:
- một câu UEMR thắng R0;
- một câu có bước bị bỏ sót;
- một câu VCMR đúng;
- một câu thất bại.

Mỗi câu kèm khung hình, đoạn trả về và đoạn GT. Hạng từng câu của dev đã có (`results/uemr_final/dev/ranks_*.json`), nhưng khung hình cần
video. Sẽ chọn cùng lúc với benchmark để có ví dụ zero-shot.

## 10. Đóng góp

Viết đúng mức của file 00 §7, không mạnh hơn.

1. **UEMR:** chỉ mục đa vector theo sự kiện, gồm cắt bằng UniAV, lấp chỗ trống và MaxSim. Một chỉ mục phục vụ cả truy hồi video lẫn định vị
   khoảnh khắc.
2. **Bộ định vị sự kiện chưng cất từ Omni truy hồi:** một lượt UniAV cho cả đoạn lẫn vector đoạn trong không gian Omni, có ngữ cảnh video và
   task. Từ đó có UEMR-Lite (không chạy Omni 7B phía video) và UEMR-Full.
   - Kiến trúc span head và loss chưng cất là của code nhóm (HanLi). Đóng góp nằm ở việc dùng chính Omni truy hồi làm thầy, tích hợp vào
     truy hồi và đánh giá. Không claim distillation là mới.
   - Chưa viết "có hiệu quả": điều kiện là K4 hoặc K5 đạt trên ít nhất một benchmark, mà benchmark chưa chạy.
3. **Giao thức đánh giá:** suy luận không rò rỉ article cho bộ định vị có điều kiện task, chấm multi-positive, dev và benchmark tách bạch,
   zero-shot.
4. **Nghiên cứu có kiểm soát** trên dev (cách cắt, cột; mục 6) và **đánh giá zero-shot** trên benchmark (mục 7, chờ chạy).

## 11. Hạn chế

- **Số dev lạc quan.** Checkpoint của cả UniAV và Omni `best` đều chọn trên `val_seen`, và θ, `L_min` / `L_fill`, β\* cũng vậy. Kết luận chỉ
  được viết từ benchmark, mà benchmark chưa chạy.
- **Trên dev, R8 chưa hơn R2 có ý nghĩa** (+1,77 [−0,53, +4,02]). Lợi ích của ranh giới UniAV cho R@1 chưa tách khỏi lợi ích của việc có
  nhiều clip. Ranh giới chỉ giúp rõ ở VCMR.
- **β\* = 1,0 ở biên trên của lưới.** Không biết β lớn hơn có tốt hơn trên dev không.
- **UniAV học trên ít video hơn Omni** (5.826 so với 6.993). Vector thầy thiếu cho 2.581 / 36.026 đoạn train (7,16 %). Có 395 video train có
  đặc trưng nhưng không có thầy (`01:L205`).
- **Article:**
  - UniAV có bảng article đóng của HT-Step. Trên benchmark, article đoán là task HT-Step gần nhất.
  - Lúc train UniAV luôn được cho article GT, lúc dùng phải đoán: mAP giảm từ 30,31 xuống 20,14 (mục 4). Chưa thử train với article đoán.
  - C3 cho thấy `h_i` nhạy với article (−22,58 khi article sai), nên article đoán sai trên miền khác có thể làm (S) yếu đi. **[giả thuyết]**
- **Dạng câu hỏi:** `best` học trên dạng câu `"<activity>: <headline>"`; câu benchmark là câu mô tả tự do của YouCook2.
- **Dữ liệu pretrain của Omni gốc** (InternVid, InternVid-FLT, Panda-70M, PVD) có thể trùng video benchmark (cùng nguồn YouTube), không kiểm
  hết được.
- **Thống kê:** một lượt, một seed. CI chỉ phản ánh biến thiên theo video, không phản ánh biến thiên theo seed train.
- **Không làm:** Context Fusion, cắt ngẫu nhiên / K-Medoids, nhiễu ranh giới, vai trò riêng của âm thanh. Code chưa chạy của các phần này đã
  bị xoá khỏi repo ở `91342d8` theo yêu cầu của user (plan ghi "không xoá"). Bản trước khi xoá còn ở commit `3cbd0a7`.
- **Protocol benchmark không commit git trước khi chấm** (plan yêu cầu commit). Thay bằng in sha256 và đẩy HF trước ô chấm; lịch sử commit
  trên HF là bằng chứng thứ tự thời gian (`03:L173-175`).
- **Lệch kế hoạch khác:** chấm dev và khoá config chạy ở local thay vì trong notebook (`02:L153-154`). Kho benchmark đổi từ split validation
  sang train + val (`03:L139-142`).
- **Miền:** UniAV và `best` chỉ học nấu ăn. Nếu sau này thêm bộ benchmark khác miền, đây là hạn chế chính.

## 12. Tái lập

**Code** (repo Fushion, nhánh `htstep`):

| Commit | Dùng cho |
|---|---|
| `0a8f2aa`, `f921c65`, `cdb2fac` | vector thầy (`teacher_best.py`, notebook 01) |
| `b6fab80` | notebook train UniAV |
| `2a751a9`, `65ed398` | kiểm UniAV mới, `uniav_best_checks.json` |
| `143e708`, `1cb2031` | `final_dev` (các dòng, chọn), notebook encode dev |
| `2d6f771` | chấm dev, khoá `configs/uemr_final.json` |
| `c7e020e`, `3cbd0a7` | `final_bench`, notebook benchmark (`FUSHION_REF = c7e020e`) |
| `91342d8` | dọn repo (code hiện tại) |

Code khác: HanLi-Retriever `77fc4fa` (không sửa); Omni-fix `ab772eb` (loader đọc record `clip`, qua `OMNI_QWENVL_DIR`).

**Checkpoint và dữ liệu (sha256):**

| Thứ | sha256 | Ở đâu |
|---|---|---|
| UniAV mới | `9447de877aad65407605a338e1b946f22c140f18690aae93cf35cec5468945f0` | HF `GraceFeel/uniav-htstep-best` |
| UniAV cũ | `a7c46353bf6635ad33d9cfecb04eb8ca275dc8fb4eab40d62457aef5c70c2616` | HF `nguyenminh04/uniav-htstep-omni` |
| Adapter `best` | `5e9899aa043191a87abdfd92f19adf89282c0b3318ab38c40c41354faa9e208a` | `GraceFeel/omni_lora_htstep` |
| Adapter pt | `31e81f1c43ad7b3b37b5155ad84db9f87d28128cef9ea6c02e6c3d22df380143` | `nguyenminh04/omni-adapters/omniretriever-7b` |
| Bộ thầy `config.json` | `f9abbce3876926d96d531de27e5469078d4f728c49ff5b0de2378df9335512b3` | HF `GraceFeel/htstep-omni-av-best` |
| Đầu ra UniAV mới trên dev (`uniav_raw.jsonl`) | `5db3e8cb69808c2b08af5c95ea3a416124bb00b9273d35bc966ef6c9af5cae87` | local `D:/Học/KL/output/uemr_chot/uniav_best/` |
| Cache dev 1 `segments.npz` / `text.npz` | `16636ad17b6990b3…` / `4c8a1c8f88b02b83…` | HF `GraceFeel/encode_omni`, `htstep_uemr/a4ca43db/` |
| Clip mới của dev (36.119) | — | HF `GraceFeel/encode_omni`, `uemr_final/dev/best/clips/` |
| Config cuối | `c9840ffbc964a4b0ed70baee157213adaaa6526178316c80a8435e7d2fcdbde3` | `configs/uemr_final.json` |
| Protocol benchmark | (chờ benchmark) | `configs/benchmark_protocol.json` |

**Bảng nào từ đâu:**

| Mục | Ô notebook / lệnh | File |
|---|---|---|
| 4, dữ liệu thầy | `uemr_chot_01_teacher_best.ipynb` ô 3–10 | log A1, A2 |
| 4, train | `uemr_chot_01_train_uniav.ipynb` | log A3 |
| 4, bảng UniAV | `run_uniav proposals / check / omni` + `uemr_htstep.uniav_best_checks` (local) | `results/uemr_final/uniav_best_checks.json` |
| 6.1 | `final_dev rows / select` (local) | `results/uemr_final/dev/segments/`, `tried.jsonl` |
| 6.1–6.4, 8 | `uemr_final_dev.ipynb` (encode) + `final_dev score` (local) | `results/uemr_final/dev/{metrics,pairs,groups,cost}.json`, `ranks_*.json` |
| 7 | `uemr_final_benchmark.ipynb` ô 3–10 | (chờ) `results/uemr_final/bench/yc2/` |

Vector `h_i` của dev chỉ có ở máy local (`D:/Học/KL/output/uemr_chot/dev/student/`), chưa đẩy HF. Muốn tái lập cột (S) / (U) của dev mà
không có máy này thì phải chạy lại `run_uniav student`.

Log Colab là PDF trong `Fushion/notebooks/uemr/log/`. Thư mục này nằm trong `.gitignore`, nên không có trong git.
