# CHANGES.md — các điểm khác / bổ sung so với kịch bản

Theo kịch bản (đầu file): mọi thay đổi khỏi [FIXED] hoặc diễn giải phải ghi ở đây. Không có thay đổi nào chạm vào mục [FIXED]
(ε, λ_D=λ_V=1, k_min=100, δ_F, seed split 12345, recipe §3.1, định nghĩa KL/oracle §10).

## Phương pháp được thêm theo yêu cầu của bạn (kịch bản chỉ có GA, NG+, SCRUB)
| Phương pháp | Nguồn | Ghi chú |
|---|---|---|
| `salun` | Fan et al. 2024 | random-label trên F + CE(R), gradient bị mask theo \|∇CE(F)\| (giữ top 50%). Mask tính một lần ở θ0. |
| `rl` | đối chứng của SalUn | như `salun` không mask |
| `tarun` | Tarun et al. 2023 (UNSIR) | học noise **theo class của F** tối đa hoá CE → impair (noise + R, `impair_steps` bước) → repair (R, lr thấp hơn). |
| `ft` | đối chứng | fine-tune trên R |

**Lưu ý trung thực:** `tarun`, `salun`, `scrub` là bản **tự cài đặt rút gọn** theo mô tả paper, không phải code chính thức.
Ví dụ `scrub` ở đây theo cấu trúc max-step/min-step của Kurmanji (có nhiệt độ T=4), **khác** công thức 2 hạng "SCRUB-lite" ở §5.3 của kịch bản
(`KL(s‖t)` + `γ·KL_forget`). Tôi chọn bản gốc vì nó là cái thường được trích dẫn; cần đổi thì sửa `class SCRUB` trong `unlearn.py`.
SCRUB giữ `max_iters=300` bước max-step đầu rồi chỉ còn min-step (đúng tinh thần "dừng max-step sớm").

## Các lựa chọn tôi đã tự quyết (hãy xem lại)
1. **Hyper-parameter chưa hiệu chỉnh trên dữ liệu thật** cho salun/tarun/rl/scrub/ft (xem `METHOD_HP`). Có bước lr-scan trong `KAGGLE.md §2`.
   Chỉ GA (lr 1e-3, momentum 0) và NG+ (α=1) là đúng giá trị §5.1/§5.2.
2. **Track A thay cho Track B nguyên bản:** không dùng repo `deepunlearn`/`SalUn`; tôi dựng pipeline tối giản tự viết (Track B §2) vì cần lưu per-sample loss để làm T-series.
   Hệ quả: **số liệu không đối chiếu trực tiếp** được với các repo benchmark.
3. **Profile `kaggle`:** snapshot mỗi **25** bước, **1000** bước (đúng Track A §0B.3). Profile `paper` = 10 bước/2000 bước (stage I). `k_min=100` giữ nguyên.
   Với snapshot 25 bước thì "±100 bước" ở gate go/no-go tương ứng 4 snapshot.
4. **θ0 trên D** (45.000 mẫu, V = 5.000 cuối của permutation seed 12345) đúng §3.1. Augmentation chạy trên GPU (random crop pad 4 + flip, per-sample).
   Train dùng AMP fp16; **đo loss/logit cũng dùng autocast fp16** (logits ép về fp32 trước CE/KL). Sai số fp16 nhỏ hơn 1e-5 của identity vì identity là tính chất của phép trung bình (cùng một vector per-sample loss). Nếu bạn muốn đo fp32 hoàn toàn: đặt `amp=False` trong profile.
5. **Trajectory không lưu model `.pth` mặc định** (tiết kiệm đĩa Kaggle): lưu per-sample loss (D), logits (V, subset-D 10k, F, test) mỗi snapshot dạng fp16 trong `.npz` — đủ để tính mọi metric oracle/anchor offline. Bật checkpoint bằng `--save_ckpt_every 50`.
   Logits **test** được ghi vào cùng `.npz` nhưng `score_ralm.load_arrays` **không nạp** chúng (whitelist) và `load_traj` từ chối cột/`post_selection` test (§1.2) — chỉ `aggregate`/`eval_oracle` đọc, sau khi k̂ đã được chọn.
6. **Retrain seed:** mặc định 1 seed (`--retrain_seeds 1`) để vừa ngân sách; §9 yêu cầu tới 5. Với n=1 không có median qua retrain-seed — hãy tăng khi còn compute.
7. **`retain-only` cũng áp k_min** và (nếu bật `--delta_f`) cổng δ_F như các rule khác. Chi phí online tính theo số ảnh forward/snapshot (ước lượng; `pareto-knee` ghi gần đúng).
8. **`ralm-band`, `ralm-dv-acc` (A3), T0–T10, MIA AUC** được thêm vì kịch bản yêu cầu (§0A.4–0A.8, §13, §10.3). Anchor `T4` (KL) dùng KL tuyệt đối (θ0 → 0) nên nằm cùng thang với S_R; trọng số λ=1 áp lên mọi anchor mà **không** chuẩn hoá lại (kịch bản §0A.6.1).
9. **Highloss/memorized** dựng từ logits θ0 của từng seed (`basis: per-seed`), kiểm bằng test: mọi mẫu F có CE ≥ mọi mẫu R.
10. Sanity `L_F(step 0)` khớp manifest dùng dung sai tương đối **1e-3** (kịch bản: 1e-4) vì đo ở AMP.

## Chưa làm / ngoài phạm vi lần này
- Chưa chạy trên CIFAR-10/GPU thật (không có môi trường); chưa có bất kỳ kết quả khoa học nào.
- Chưa có: A4 (nhiễu ρ nhỏ qua 5 fold), B1 (label noise), C1 (θ0 underfit), E3/E4/E5 grid, CIFAR-100/Tiny-ImageNet/ViT (Phase 3), U-LiRA. Các ablation re-analysis (A1–A3, E1/E2 qua `--ema`, T0–T10, band H4) có trong `aggregate`; E1 (`--k_min`), E2 (`--ema`) chạy lại bằng cờ CLI.
- Online/patience mode chưa cài; chỉ offline (mặc định theo §7).
