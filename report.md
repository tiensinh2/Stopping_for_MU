# Report — cấu hình thí nghiệm RALM hiện tại

> **Đây là báo cáo cấu hình, không phải báo cáo kết quả.** Tại thời điểm viết chưa có kết quả khoa học nào:
> pipeline mới chỉ được kiểm bằng 17 unit test và một lần chạy end-to-end trên **CPU với dữ liệu giả** (không phải CIFAR-10).
> Chưa có lần chạy nào trên CIFAR-10 thật hay GPU thật mà tôi quan sát được. Tôi không thấy được phiên Kaggle của bạn,
> nên **"đang chạy gì" ở đây nghĩa là "notebook được cấu hình để chạy gì"**. Trên máy phát triển không có tiến trình nào đang chạy.
> Mọi con số bên dưới được tính từ code (`ralm/config.py`, `ralm_kaggle.ipynb`), không phải đo từ lần chạy thật.

Repo: https://github.com/tiensinh2/Stopping_for_MU · Mục tiêu: kiểm tra luật dừng RALM (chọn checkpoint unlearning không cần oracle) — framing luôn là
"luật chọn k", **không phải** estimator của retrain (kịch bản §0.4).

---

## 1. Bộ dữ liệu

| Mục | Giá trị |
|---|---|
| Dataset đang cấu hình | **CIFAR-10** (profile `kaggle`), 10 lớp, ảnh 32×32 |
| Mô hình | ResNet-18 biến thể CIFAR (stem 3×3, không maxpool), width 64 |
| Train gốc | 50.000 ảnh; **test** 10.000 ảnh |
| CIFAR-100 | Có profile `kaggle_c100` nhưng **chưa nằm trong notebook** và chưa được chạy |
| Tiny-ImageNet, ViT | Chưa cài đặt |

## 2. Chia dữ liệu

Chia cố định theo §3.1, không phụ thuộc seed huấn luyện: `perm = RandomState(12345).permutation(50000)`.

| Tập | Cách tạo | Số mẫu | Dùng để |
|---|---|---|---|
| **V** (validation anchor) | 5.000 chỉ mục **cuối** của `perm` | 5.000 | anchor `S_V`, chọn k̂; **không** nằm trong D |
| **D** (huấn luyện θ0) | 45.000 chỉ mục còn lại | 45.000 | huấn luyện θ0; anchor `S_D` |
| **F** (forget) | tập con của D, xem dưới | 4.500 (ρ=0,1) hoặc 450 (ρ=0,01) | bị unlearn |
| **R** = D \ F | phần còn lại của D | 40.500 (ρ=0,1) hoặc 44.550 (ρ=0,01) | retain objective `S_R` |
| **T** (test) | test set chính thức | 10.000 | chỉ đánh giá **sau khi chọn k̂** |

Đã kiểm: D ∩ V = ∅ (0 mẫu chung); code assert F ⊂ D, F ∩ V = ∅, R ∩ F = ∅ mỗi lần dựng.

**Forget set trong notebook hiện tại** — chỉ loại `random`, chọn ngẫu nhiên không hoàn lại từ D bằng seed 12345:

| fkey | ρ | \|F\| | \|R\| |
|---|---|---|---|
| `random_rho0.1` | 10% | 4.500 | 40.500 |
| `random_rho0.01` | 1% | 450 | 44.550 |

- F(1%) là **tập con của** F(10%) (đã kiểm: `np.isin(F1, F10).all() == True`), vì `RandomState.choice` không hoàn lại lấy tiền tố của một hoán vị. Hệ quả: hai cấu hình ρ **không độc lập** — cần nhớ khi diễn giải "RALM thắng ở cả hai ρ".
- F **giống nhau cho mọi seed** (manifest `global`). Seed 0/1/2 chỉ thay đổi θ0 và quá trình unlearning, không đổi F.
- Đã cài đặt nhưng **chưa nằm trong notebook**: `random_rho0.001` (45 mẫu), `highloss_*`, `memorized_*` (dựng từ logits θ0 của từng seed), `classwise_<tên|id>` (vd `classwise_airplane`, `classwise_3`). Kịch bản coi `highloss` và `classwise` là các ca khó quyết định việc một anchor có "thắng" T0 hay không (§0A.6.2); **với cấu hình hiện tại (chỉ random) T-matrix không thể có anchor nào thắng**, vì tiêu chí đòi hỏi cả highloss và classwise.
- `subset-D` 10.000 mẫu cố định (seed 777) dùng cho anchor T5/T6; không phụ thuộc F.

## 3. Huấn luyện θ0 và θ_R (recipe §3.1, [FIXED])

| Thành phần | Giá trị |
|---|---|
| Optimizer | SGD, momentum 0,9, weight decay 5e-4 |
| lr / schedule | 0,1; warmup 5 epoch rồi cosine (cập nhật theo iteration) |
| Epochs / batch | 100 / 128 |
| Augmentation (chỉ khi train) | RandomCrop(32, pad 4) + flip ngang, chạy trên GPU |
| Độ chính xác | AMP fp16 khi train và khi đo (logits ép về fp32 trước CE/KL) |
| Gate | test acc θ0 ≥ **93%**, nếu không `base` tự dừng bằng `SystemExit` |
| θ_R (retrain oracle) | cùng recipe, huấn luyện từ đầu **chỉ trên R**; chỉ `eval_oracle`/`aggregate` được đọc |

## 4. Các method unlearning

Tất cả chạy từ θ0 với **1.000 bước**, snapshot mỗi **25 bước** (41 snapshot: step 0, 25, …, 1000), batch unlearn 128, có augmentation.
Siêu tham số lấy từ `METHOD_HP`, giữ nguyên giữa các seed (§18.5).

| Method | Có trong notebook? | Cơ chế | Siêu tham số chính | Nguồn / mức tin cậy |
|---|---|---|---|---|
| `ga` | Có | −CE(F) (gradient ascent) | lr 1e-3, momentum 0 | đúng §5.1 |
| `ngp` (NegGrad+) | Có | −CE(F) + α·CE(R) | lr 1e-3, α=1 | đúng §5.2 |
| `scrub` | Có | max-step −KL(t‖s) trên F (300 bước đầu), min-step α·KL + γ·CE trên R | lr 5e-4, mom 0,9, α=0,5, γ=1, T=4 | **tự cài** theo Kurmanji 2023; **khác** SCRUB-lite §5.3 |
| `salun` | Có | random-label trên F + CE(R), gradient mask giữ top 50% theo \|∇CE(F)\| (tính một lần ở θ0) | lr 5e-3, mom 0,9, mask 0,5 | **tự cài** theo Fan 2024 |
| `tarun` (UNSIR) | Có | học noise theo class của F → impair (100 bước, noise + R) → repair (R, lr 1e-2) | lr 2e-2, 32 noise/class | **tự cài** theo Tarun 2023 |
| `rl` | Chưa (đã cài) | như SalUn không mask | lr 5e-3 | đối chứng |
| `ft` | Chưa (đã cài) | CE(R) | lr 5e-3 | đối chứng |

**Cảnh báo:** `scrub`, `salun`, `tarun`, `rl`, `ft` **chưa được hiệu chỉnh trên dữ liệu thật**. Trong lần chạy thử dữ liệu giả (mô hình tí hon),
salun/rl/tarun làm L_R tăng rất nhanh; có thể là artefact, nhưng cần chạy lr-scan (notebook mục 2) trước khi tin kết quả của chúng.
Nếu bạn muốn dùng SCRUB đúng công thức hai hạng của kịch bản §5.3 thì phải sửa `class SCRUB` trong `ralm/unlearn.py`.

## 5. Luật dừng được so sánh

`ralm-dv` (chính), `ralm-d`, `ralm-v`, `ralm-band` (H4), `ralm-dv-acc` (A3), `retain-only`, `val-only`, `pareto-knee`,
`fixed-{50,100,200,500,1000,2000,last}`, `random-k` (5 seed). `k_min = 100`, λ_D = λ_V = 1, ε = 1e-12 [FIXED].
Với `n_steps=1000`, `fixed-1000`, `fixed-2000` và `fixed-last` trùng nhau (đều bị chặn về step cuối) — bảng sẽ có dòng lặp, không phải lỗi dữ liệu.
Oracle: `k* = argmin_k G_logit_F(k)` với `G_logit_F = mean_{x∈F} KL(p_θR ‖ p_θk)`; `Regret = G(k̂) − G(k*)`.

## 6. Ma trận chạy theo notebook

| Giai đoạn | Cấu hình | Số job |
|---|---|---|
| Smoke (mục 1) | seed 0 · `random_rho0.01` · `ga ngp scrub` | 1 base + 1 retrain + 3 trajectory |
| Chính (mục 3) | **1 seed mỗi session** (0, rồi 1, rồi 2; cộng dồn qua `RESTORE_FROM`) · `random_rho0.1`, `random_rho0.01` · `ga ngp scrub salun tarun` | mỗi seed: 1 base + 2 retrain + 10 trajectory (3 seed: 3 + 6 + 30) |

- `RETRAIN_SEEDS = 1` (kịch bản §9 cho tới 5): với n=1 không có trung vị qua seed retrain.
- Job chia cho 2 GPU T4 theo `i % 2` (mỗi GPU một tiến trình). Mọi job idempotent, bỏ qua nếu đã có kết quả.
- Mỗi snapshot forward toàn bộ 50.000 ảnh train (D ∪ V) một lần rồi lấy chỉ mục ra R/D/F/V, nên identity `L_D = (1−ρ)L_R + ρL_F` được kiểm (sai lệch < 1e-5) tại **mọi** snapshot; vi phạm sẽ làm job dừng.
- Mỗi snapshot lưu (fp16): per-sample loss trên D, logits trên V / subset-D / F / test, cùng `w_dist` (T8) và `emb_drift_V` (T9). Chỉ `aggregate` đọc logits test, sau khi k̂ đã chọn.
- Thời gian dự kiến cỡ 6–10 giờ GPU cho chạy chính — **ước lượng, chưa đo**.

## 7. Trạng thái xác minh

| Hạng mục | Trạng thái |
|---|---|
| Unit test (`tests/test_ralm.py`) | 17/17 pass trên CPU |
| End-to-end (base → 7 method → retrain → aggregate → hình) | pass trên **dữ liệu giả**, CPU |
| Cô lập oracle (§1.1) / chặn cột test (§1.2) | được test tự động |
| Chạy trên CIFAR-10 thật | **chưa** |
| Chạy trên GPU thật / đo tốc độ, VRAM | **chưa** |
| Gate θ0 ≥ 93% | **chưa biết** |
| Smoke go/no-go | **chưa chạy** |
| Notebook: các cell có `!`/`%` (clone, train, aggregate) | **chưa chạy** (chỉ các cell phân tích được thử trên dữ liệu giả) |

## 8. Chưa làm so với kịch bản

Ablation A4 (nhiễu ρ nhỏ), B1 (label noise), C1 (θ0 underfit), grid E3/E4/E5, CIFAR-100 / Tiny-ImageNet / ViT (Phase 3), U-LiRA, chế độ online (patience).
Các điểm khác kịch bản được ghi đầy đủ ở [`CHANGES.md`](CHANGES.md).

## 9. Bước tiếp theo đề xuất

1. Chạy notebook mục 0 → 1 (smoke) và xem cell go/no-go cùng hình `regret_curve_*`.
2. Nếu GO: chạy mục 2 (lr-scan) cho `salun`/`tarun`/`scrub`, cập nhật `METHOD_HP` **một lần**, rồi mới chạy mục 3.
3. Muốn T-matrix và H3 có ý nghĩa: thêm `highloss_rho0.01` và `classwise_airplane` vào `FKEYS`.
4. Nếu bất kỳ cell nào lỗi, gửi log; không đổi recipe huấn luyện (§18).
