# Kịch bản thí nghiệm RALM — bản chi tiết cho AI agent

Tài liệu này là **kịch bản thi hành (execution protocol)** viết cho một AI agent chạy thí nghiệm tự động. Agent phải đọc toàn bộ tài liệu trước khi viết dòng code nào. Các thiết lập đánh dấu **[FIXED]** không được thay đổi; mọi thay đổi khác phải ghi vào `CHANGES.md` kèm lý do. Nếu gặp mâu thuẫn giữa các mục, mục có [FIXED] thắng.

---

## 0. Bối cảnh, định nghĩa và giả thuyết

### 0.1 Ký hiệu

| Ký hiệu | Ý nghĩa |
|---|---|
| `D` | toàn bộ training set dùng khi huấn luyện θ0 |
| `R` | retain set = D \ F |
| `F` | forget set |
| `V` | validation anchor, tách khỏi D **trước khi** huấn luyện θ0 |
| `T` | test set chính thức, giữ kín đến cuối |
| `ρ` | |F| / |D| |
| `θ0` | base model đã huấn luyện trên D |
| `θ_R` | retrain oracle, huấn luyện lại từ đầu chỉ trên R |
| `θ_k` | checkpoint thứ k của trajectory unlearning |
| `L_X(θ)` | cross-entropy trung bình trên tập X tại θ |
| `A_X(θ)` | top-1 accuracy trung bình trên tập X tại θ |
| `^0` | giá trị ghi nhận tại θ0 trước khi unlearning |
| ⋆ Các loss đều là **cross-entropy trung bình (reduction="mean")**, đo bằng full forward trên toàn tập X, không phải trung bình mini-batch. |

### 0.2 Điểm số RALM

```
S_R(k) = L_R(k) / (L_R^0 + ε)                      # retain objective, nhỏ tốt hơn
S_D(k) = |L_D(k) − L_D^0| / (L_D^0 + ε)            # full-training anchor
S_V(k) = |L_V(k) − L_V^0| / (L_V^0 + ε)            # generalization anchor

J_RALM-DV(k) = S_R(k) + λ_D·S_D(k) + λ_V·S_V(k)    # phương án chính
J_RALM-D(k)  = S_R(k) + λ_D·S_D(k)                 # ablation E
J_RALM-V(k)  = S_R(k) + λ_V·S_V(k)                 # ablation F

k̂ = argmin_{k ≥ k_min} J(k)
```

- `ε = 1e-12` [FIXED] chỉ để chống chia 0.
- `λ_D = λ_V = 1` [FIXED] cho mọi thí nghiệm chính. Không grid search λ trong giai đoạn chính; λ-grid chỉ chạy trong ablation E3 để chẩn đoán.
- `k_min = 100` (stage I) [FIXED]. Nếu ≥ 50% run chọn `k̂ = k_min`, nâng `k_min = 200` và ghi CHANGES.md.
- **Dạng tổng quát (thí nghiệm anchor thay thế, mục 0A):** `J(k) = S_R(k) + Σ_i λ_i·A_i(k)`, mỗi `A_i` là một anchor thay cho `S_D` (hoặc `S_V`). RALM chuẩn = trường hợp `A_1 = S_D`, `A_2 = S_V`. Chi tiết lý thuyết và ma trận T0–T9 ở mục 0A.

### 0.3 Quan hệ đại số nền tảng

`L_D = (1−ρ)·L_R + ρ·L_F` là đẳng thức chính xác (cùng reduction, cùng tập). Hệ quả: khi giữ `L_D ≈ L_D^0` và giảm `L_R`, forget loss bị ép tăng theo `ΔL_F = ((1−ρ)/ρ)·|ΔL_R|`. Đây là cơ chế nhạy của RALM, đồng thời là nguồn khuyếch đại nhiễu 1/ρ — ablation A4 kiểm tra trực tiếp điều này.

### 0.4 Giả thuyết

- **H1 (chính):** checkpoint do RALM chọn có regret (định nghĩa §7) thấp hơn `fixed-200`, `retain-only`, và `random-k`, không dùng oracle khi chọn.
- **H2:** thành phần anchor `S_D` mang thông tin thật, không chỉ neo model về θ0. (Kiểm: ralm-d vs retain-only, ablation A1.)
- **H3:** RALM có **ranh giới hiệu lực dự đoán được**: suy giảm khi ρ lớn, forget set influential, hoặc θ0 underfit. Sự suy giảm này là kết quả khoa học, không phải thất bại cần che giấu.
- **Anti-claim [FIXED]:** RALM là luật chọn checkpoint không cần oracle, **không phải** estimator của retrain target. Không dùng câu "RALM ước lượng retrain" trong bất kỳ báo cáo nào.

### 0.5 Câu hỏi nghiên cứu

- RQ1: kiểm tra H1 (so với baselines trong §6).
- RQ2: biến thiên theo ρ và loại forget set (H3).
- RQ3: chuyển giao hằng số (λ, k_min, patience) qua dataset/architecture (điều kiện thành công số 4).
- RQ4: mỗi anchor (D, V) đóng góp gì (H2).
- RQ5: checkpoint regrets thấp có loại bỏ được membership leakage trên F (privacy proxy) không.

---

## 0A. Nền tảng lý thuyết — tại sao anchor là L_D, và các hàm anchor thay thế

Mục này là phần lý thuyết chi tiết. Agent phải đọc kỹ trước khi code `score_ralm.py`, vì §0A.4–0A.6 yêu cầu lưu thêm dữ liệu ngay từ Phase 1 (mục 0A.6.3).

### 0A.1 Đẳng thức bảo toàn — phát biểu, chứng minh, điều kiện

**Định lý (conservation identity).** Nếu `L_X(θ)` là trung bình cộng của per-sample loss trên tập X, và D = R ⊔ F (disjoint union), thì với mọi θ:

```
L_D(θ) = (1/|D|) Σ_{x∈D} ℓ(x;θ)
       = (1/|D|) [ Σ_{x∈R} ℓ(x;θ) + Σ_{x∈F} ℓ(x;θ) ]
       = (|R|/|D|)·L_R(θ) + (|F|/|D|)·L_F(θ)
       = (1−ρ)·L_R(θ) + ρ·L_F(θ)
```

Chứng minh 2 dòng, nhưng **đẳng thức chỉ đúng khi thỏa đồng thời 6 điều kiện** — mỗi điều kiện là một bug tiềm ẩn của pipeline:

1. Loss dùng reduction="mean" trên cả ba tập (không phải sum, không phải balanced-weight).
2. Trọng số per-sample đều (không class-balanced, không sampler-weighted).
3. R ∩ F = ∅ và F ⊂ D cùng index gốc (không duplicate qua augmentation offline).
4. Cùng một θ_k được forward trên cả D và trên R, F riêng (không R forward bằng batch-shuffle khác).
5. Không augmentation khi đo eval [§1.5] (nếu có, L_D ≠ tổ hợp tuyến tính của L_R, L_F theo nghĩa số học).
6. Cùng số lớp logit được dùng khi tính loss trên cả ba tập.

Pipeline phải tự kiểm tra identity sau mỗi snapshot (sai lệch < 1e-5, xem §6). Identity này là *tính chất của trung bình cộng*, không phải của CE — nó vẫn đúng cho bất kỳ per-sample loss nào (MSE, hinge, focal) miễn reduction là mean. Nói cách khác: **đẳng thức là thuộc tính của phép lấy trung bình, không thuộc tính của hàm loss.** Đây là điểm then chốt để hiểu hạn chế ở 0A.3.

**Hệ quả amplifier:** đẩy `L_D(k) → L_D^0` và `L_R(k) = L_R^0 − Δ_R` kéo theo `ΔL_F = ((1−ρ)/ρ)·Δ_R`. Hệ số `(1−ρ)/ρ ≈ 1/ρ` với ρ nhỏ — cùng một cơ chế làm signal nhạy cũng làm nhiễu đo khuyếch đại (ablation A4 định lượng).

### 0A.2 Động lực học từng bước — vì sao anchor "làm việc"

Đẳng thức 0A.1 là tĩnh. Cần xem động lực: một bước GA (ascent trên F, lr η, gradientforget ∇L_F) cho:

```
ΔL_F ≈ +η·‖∇L_F‖²              (luôn dương: ascent luôn tăng forget loss)
ΔL_R ≈ −η·⟨∇L_R, ∇L_F⟩·|F|/|R|  (dấu tùy góc giữa hai gradient)
ΔL_D ≈ ρ·η·‖∇L_F‖² + (1−ρ)·ΔL_R
```

Ba pha có thể quan sát trên trajectory, và **đây là lý do giải thích cấu trúc của J:**

- **Pha I (ascent thuần):** ⟨∇L_R, ∇L_F⟩ ≥ 0, tức-gradient forget trùng hướng gradient retain → L_D **tăng**. S_D bắt J phạt ngay. Đây là cơ chế phát hiện over-unlearning nhanh nhất.
- **Pha II (collateral damage):** L_F tăng chậm, L_V bắt đầu tăng, L_D tăng do generalization drift. S_V bắt sóng trước S_D (V loss nhạy hơn train loss vì không memorize).
- **Pha III (retain recovery, chỉ với NG+/SCRUB/retain-finetune):** L_R giảm thật sự dưới L_R^0 → theo identity, L_F bị ép tăng. Đây là pha duy nhất mà cơ chế conservation "sống", và J đạt min trong pha này hoặc pha II muộn.

**Suy ra dự đoán kiểm chứng được (đưa vào report phase 1):** với GA thuần, `argmin J` thường rơi ở vùng chuyển pha II→III; nếu trajectory không bao giờ vào pha III (GA không bao giờ giảm L_R), thì anchor D chủ yếu đang làm việc như "phát hiện L_D tăng", tức gần giống `argmin |ΔL_D|`, và giá trị phụ thuộc của nó giảm — kiểm chứng bằng cách so sánh corr(S_D(k), S_V(k)) qua trajectory.

### 0A.3 Ba lập luận cho anchor L_D, và ba lỗ hổng

**Lập luận 1 — Exact và miễn phí:** L_D có sẵn trong pipeline, không cần thêm forward, và qua identity nó *xác định* L_F khi biết L_R: một ràng buộc, hai đại lượng.

**Lập luận 2 — Neo manifold huấn luyện:** mọi θ thu được bằng bất kỳ thuật toán nào huấn luyện trên D đều có xu hướng nằm "gần" L_D ≈ L_D^0 (trong khoảng noise floor của SGD). Retrain thoát khỏi vùng này theo hướng riêng của nó: từ công thức tổng, độ lệch mục tiêu của anchor là

```
L_D(θ_R) − L_D^0 = (1−ρ)·[L_R(θ_R) − L_R^0] + ρ·[L_F(θ_R) − L_F^0]
```

với `L_F(θ_R) > L_F^0` gần như chắc chắn (θ_R chưa từng thấy F). Anchor L_D do đó **mục tiêu lệch hệ thống (misspecified)** so với θ_R: nó kéo θ_k về phía vùng huấn luyện-có-thể, không về θ_R. RALM không sửa được điều này — nó chỉ chọn điểm thương lượng (trade-off) tốt nhất dọc trajectory. Mọi run phải đo và report `L_D(θ_R) − L_D^0` làm thước của misspecification (đã ghi §9).

**Lập luận 3 — Không cần oracle:** cả ba đại lượng đều có trước unlearning.

**Lỗ hổng 1 — Projection blindness (quan trọng nhất cho hướng "hàm khác"):** L_D là **phép chiếu 1 chiều** của toàn bộ phân phối per-sample loss {ℓ(x;θ)}_{x∈D} xuống trung bình của nó. Hai mô hình có L_D bằng nhau có thể khác nhau tùy ý ở *hình dạng* phân phối — đặc biệt ở **đuôi trên**, nơi các mẫu memorized và các mẫu MIA phát hiện được nằm. Population/U-LiRA thắng thua ở đuôi, mean thì mù đuôi. Đây là lập luận lý thuyết trực tiếp đòi hỏi anchor phi tuyến (0A.4).

**Lỗ hổng 2 — Redundancy đại số:** với anchor là hàm g(L_D) (bất kỳ), vì L_D = (1−ρ)L_R + ρL_F và khi L_D bị neo, mọi thông tin của anchor chỉ còn phụ thuộc L_R qua đường: `g((1−ρ)L_R + ρL_F)` — tức anchor dạng "hàm của trung bình" là *dư thừa hoặc gần dư thừa* với S_R. Cụ thể: khi ρ nhỏ, L_D = (1−ρ)L_R + ρL_F có biến thiên chủ yếu theo L_R (biến thiên qua L_F bị nhân ρ, rất nhỏ) → corr(S_D, S_R) cao và S_D thêm rất ít thông tin mới. Một anchor chỉ chứa thông tin mới khi nó phụ thuộc **hình dạng/tail/phân phối** của per-sample loss hoặc lời dự đoán — không phụ thuộc số trung bình một mình.

**Lỗ hổng 3 — 1/ρ amplification:** với ρ nhỏ, sai số đo L_D phản chiếu sang sai số suy diễn L_F gấp ~1/ρ lần (ablation A4). Không có thay đổi hàm anchor nào sửa được *độ chính thu đo*, chỉ có thể chọn anchor *nhạy đúng chỗ* (tail) thay vì nhạy cả nhiễu mean.

### 0A.4 Họ hàm anchor thay thế (pluggable anchors)

Dạng tổng quát, thay cho 1 anchor:

```
J(k) = S_R(k) + Σ_i λ_i · A_i(k),     A_i(k) = u_i( m_i(k) ) / ( u_i(m_i^0) + ε )
```

trong đó `m_i` là một thống kê của **per-sample loss trên D** (hoặc logits/dự đoán trên D, V). Bảng anchor — mỗi dòng cần lưu số liệu gì (xem 0A.6.3):

| ID | m_i | Định nghĩa | Nhìn thấy gì mà mean không thấy | Chuẩn hoá A_i | Cost |
|---|---|---|---|---|---|
| T0 | mean | `(1/n)Σℓ_i` trên D | (baseline hiện trạng) | `\|m−m^0\|/(m^0+ε)` | 1 forward (đã có) |
| T1 | variance | `Var({ℓ_i})` trên D | dispersion; unlearn làm đuôi dày hơn | `\|m−m^0\|/(m^0+ε)` | từ per-sample loss |
| T2 | CVaR-q | mean của top-q% loss lớn nhất trên D, q ∈ {0.5, 1, 5} | **đuôi trên = vùng memorization/MIA** | ratio như trên | từ per-sample loss |
| T3 | quantile | P50, P90, P99 của {ℓ_i} trên D | hình dạng phân phối thay vì trung bình | ratio | từ per-sample loss |
| T4 | KL-forward | `mean_x KL(p_θk(·\|x) ‖ p_θ0(·\|x))` trên V | phân phối output đầy đủ, không chỉ loss scaler | giá trị KL thô (KL(θ0,θ0)=0, tự chuẩn hoá theo nats) | cần logits θ0 (đã lưu §3.2) + logits θk trên V mỗi snapshot |
| T5 | KL trên D | như T4 nhưng trên D | hành vi drift trên đúng tập được neo | KL thô | cần logits θk trên D mỗi snapshot (subset 10k nếu disk thiếu) |
| T6 | confidence | mean max-softmax trên D và V | memorization thể hiện qua confidence, không loss | ratio | từ logits đã lưu |
| T7 | p-norm | `(1/n Σ ℓ_i^p)^{1/p}`, p ∈ {2, 4} | nhạy tail trong khoảng giữa mean và CVaR | ratio | từ per-sample loss |
| T8 | weight-norm | `‖θ_k − θ0‖₂ / sqrt(n_params)` | implicit regularization; cost 0 (không forward) | ratio (m^0 = 0 → dùng giá trị thô) | từ state_dict |
| T9 | embedding | mean² khoảng cách penultimate-feature giữa θk và θ0 trên V (normalized) | drift biểu diễn, bền với recalibration head | giá trị thô (m^0 = 0) | same forward, lấy feature |

**Nhận xét lý thuyết về identity theo từng anchor:** T0, T1, T7 là *hàm của trung bình tổng quát hoá* — identity tuyến tính chỉ còn cho T0. T2/T3 (CVaR, quantile) **phá vỡ identity**: chúng không suy ra được từ L_R và L_D, tức chúng thêm thông tin thật về cấu trúc phân phối — đúng mục tiêu của hướng "hàm khác". T4/T5, T6, T9 phụ thuộc logits/nhãn-predictions, cũng ngoài hệ identity. **Kỳ vọng khoa học:** anchors ngoài họ mean (T2, T4/T5, T6) sẽ có corr với S_R thấp hơn và regret tốt hơn ở forget set influential, nhưng tốn chi phí lưu/logistic nhẹ.

**Cảnh báo chuẩn hoá:** T1 có m^0 nhỏ (variance vốn nhỏ) → ratio dễ bị khuyếch đại nhiễu; dùng tốt hơn dạng `|m − m^0| / (P99(loss^0) + ε)` nếu ratio gốc không ổn — quyết định sau khi xem A4, không chỉnh giữa chừng.

### 0A.5 Đo mức dư thừa của từng anchor (phần chẩn đoán bắt buộc)

Với mỗi run, tính trên trajectory:

1. **corr(A_i(k), S_R(k))** qua mọi checkpoint k: nếu |corr| > 0.9 → anchor gần như trùng lặp với retain objective (kỳ vọng: T0 cao; T2/T4/T6 thấp hơn).
2. **corr(A_i(k), G_logit_F(k))** (chỉ tính offline, sau khi có θ_R): chỗ này chính là câu trả lời cho H2 mở rộng — anchor nào track hành vi-forget-deep nhất.
3. **Scatter plot A_i(k) vs G_logit_F(k)** per config, đưa vào report.

Kết luận T-matrix dựa trên (2), không phải (1); (1) dùng để giải thích tại sao anchor nào vô dụng.

### 0A.6 Ma trận thí nghiệm T — các anchor như ablations

**Quan trọng về chi phí:** toàn bộ T-series (T0–T7, T9) là **re-analysis trên cùng trajectories** của Phase 1 — không cần chạy unlearning thêm, chỉ cần dữ liệu per-snapshot được lưu đúng từ đầu (0A.6.3). Chi phí thực của T-series chỉ là compute CPU/pandas.

0A.6.1 **Cấu hình T:** mỗi anchor chạy như một stopping rule ralm-*Anchor*: `J = S_R + λ·A_i` và `J = S_R + λ·A_i + λ·A_V-anchor` (khi anchor thay S_D, giữ S_V nguyên; ngược lại giữ T0 làm anchor D nếu đổi S_V). So sánh với baseline T0 (ralm-dv chuẩn).

0A.6.2 **Tiêu chí một anchor "thắng":** median regret giảm ≥ 20% so với T0 trên (a) highloss và (b) classwise forget sets, không tệ hơn > 10% trên random. Nếu chỉ thắng trên random → anchor chỉ đo generalization drift, không thêm thông tin memorization.

0A.6.3 **Yêu cầu lưu dữ liệu ngay từ Phase 1 [FIXED — không lưu thì T-series không làm được]:**

- Mỗi snapshot: lưu vector **per-sample loss trên toàn D** dạng npy fp16, `ckpt/traj/.../stepNNNN_dperloss.npy` (45.000 × 4 bytes × 200 snapshot ≈ 36 MB/run — chấp nhận được).
- Mỗi snapshot: lưu **logits trên V** (fp16, 5.000×C ≈ 0.1 MB) và **logits trên subset-D 10.000 mẫu cố định (chỉ mục cố định manifest)** phục vụ T5.
- θ0 logits trên D đã lưu từ §3.2 — dùng cho mọi T4/T5/T6.
- Mỗi state_dict checkpoint (mỗi 50 step) dùng cho T8 — không cần thêm gì.

0A.6.4 **Trình tự:** Phase 1 xong → chạy T-series offline → nếu có một anchor ngoài họ mean thắng T0 theo 0A.6.2 ở cả highloss và classwise → mới đưa anchor đó vào Phase 3 (transfer test) song song với T0. Không chọn anchor dựa trên kết quả random-forget một mình (nguy cơ chỉ đo drift tổng quát).

### 0A.7 Giả thuyết "khoảng đích ổn" (admissible band) — H4

**Phát biểu của lập luận:** các anchor không xác định được *điểm* đích (vì ta không biết per-sample/per-class target của retrain, cũng không thấy bên trong không gian feature), nhưng chúng giới hạn trajectory vào một **khoảng đích ổn** — tập các checkpoint mà mọi anchor đều nằm trong dung sai. Chọn điểm tốt nhất trên retain *bên trong khoảng này* có thể đủ gần oracle, dù khoảng không containment đầy đủ.

**Chính thức hóa:**

```
B(ε) = { k : S_D(k) ≤ ε_D  và  S_V(k) ≤ ε_V }        # khoảng chấp nhận được
k̂_band = argmin_{k ∈ B(ε), k ≥ k_min} S_R(k)          # chọn trong band
```

Đây tức là phiên bản constrained stopping (§6 cũ, phương án "phiên bản mạnh hơn") — giờ đọc như một **funnel filter**: anchor lọc thô, S_R chọn tinh. Dạng band có hai ưu điểm lý thuyết so với dạng cộng:

- Sai số anchor không cộng dồn tuyến tính vào selecting score; band chỉ quyết định *thu hẹp tập ứng viên*.
- Mỗi anchor có ngưỡng riêng giải thích được (`ε_D = 2.5%` = "ch cách lệch train loss quá 2.5%") — dễ defend trong paper hơn λ.

**Ba chỉ số kiểm chứng H4 (tính offline sau mỗi run, đưa vào report):**

1. **Coverage:** tỷ lệ seed mà `k* ∈ B(ε)` (hoặc |k* − k gần nhất trong B| ≤ 100 steps). H4 cần coverage ≥ 80% trên random-forget; nếu coverage thấp, band đang cắt nhầm vùng chứa oracle.
2. **Path-width:** số checkpoint liên tục trong B. Band quá hẹp (width ≈ 1 snapshot = 10 steps) nghĩa là anchor đang hoạt động như điểm-ước lượng giả tạo, nhạy nhiễu; band quá rộng (> 50% trajectory) nghĩa là ràng buộc vô dụng. Kỳ vọng vùng tốt: width 5–30 snapshots.
3. **In-band dispersion:** phân phối của `G_logit_F(k)` **giới hạn trong B** so với ngoài B. Nếu mean và variance của G_logit_F trong B thấp hơn rõ rệt ngoài B → band thực sự "ổn" (đúng tên gọi). Đây là kiểm chứng trực tiếp nhất cho H4, không cần RALM thắng: chỉ cần band chứa các checkpoint tốt.

**Chọn ε:** [FIXED] xuất phát `ε_D = 0.05, ε_V = 0.05`; sau A4, tinh chỉnh sao cho path-width rơi vào 5–30 snapshots trên random-1% (đây là calibration **trên retain-metric band width**, không dùng retrain — hợp lệ). Grid ε chỉ ở dạng ablation E5.

**Quan hệ với H1:** thành công của H1 (RALM-DV dạng cộng) ngụ ý H4 nhưng ngược lại chưa chắc. Kỳ vọng khoa học: band-version bền hơn ở ρ nhỏ (không bị khuyếch đại nhiễu cộng dồn), và nếu band thắng version cộng ở random-ρ=0.001, đó là bằng chứng cấu trúc, không phải noise.

### 0A.8 Anchor per-class (trả lời trực tiếp lo "chỉ biết loss chung")

Với D, ta *có* nhãn của mọi mẫu — nên ta đo được thống kê per-class của chính θk, thiếu chỉ là target:

```
A_class(k) = sqrt( (1/C) Σ_c [ (m_c(k) − m_c^0) / (m_c^0 + ε) ]² )
```

với `m_c(k)` = mean per-sample loss của class c trên D tại θk (tính từ vector per-sample loss đã lưu mỗi snapshot, không tốn thêm forward). Đây là khoảng cách Euclidean chuẩn hóa trong không gian {per-class mean loss}, tức anchor **C chiều thay vì 1 chiều**; nó phát hiện trường hợp mean D "ổn" nhưng một class hiếm bị phá (đúng blind spot mà mean không nhìn thấy). Kỳ vọng: A_class hữu ích nhất với forget set classwise — chính là ca mà T0 yếu nhất. Đưa vào T-series với ID **T10**, cùng hit-criteria 0A.6.2.

---

## 0B. TRACK A — đường nhanh: đứng trên repo benchmark có sẵn, chỉ thêm lớp dừng

Đây là **path được chạy trước tiên** khi user nói "thí nghiệm đơn giản": không dựng pipeline từ đầu, chỉ gắn RALM scoring lên code unlearning sẵn có. Track B (các mục 1–18) chạy sau nếu Track A cho tín hiệu khả quan.

### 0B.1 Substrate

- **Chính:** [`xcadet/deepunlearn`](https://github.com/xcadet/deepunlearn) — official implementation của Deep Unlearn: Benchmarking Machine Unlearning for Image Classification. Repo đã có sẵn: huấn luyện original model, retrained model, nhiều phương pháp unlearning (GA, NG+, SRL, v.v.), và MIA/U-LiRA evaluation. Note quan trọng từ chính benchmark này: GA/SRL là baseline yếu, NG+ tune đúng mới là baseline mạnh — nên phương pháp chọn để test RALM phải gồm NG+.
- **Phụ (nếu deepunlearn khó chạy):** [`meghdadk/SCRUB`](https://github.com/meghdadk/SCRUB) (official SCRUB, NeurIPS 2023) và [`OPTML-Group/Unlearn-Saliency`](https://github.com/OPTML-Group/Unlearn-Saliency) (SalUn, ICLR 2024, thư mục Classification có sẵn script `main_random.py` cho ResNet-18 + CIFAR-10 + 10% forget). SalUn repo là lựa chọn dễ chạy nhất vì README có lệnh mẫu trực tiếp.
- **Quyết định:** agent thử deepunlearn trước (đủ method + eval); nếu setup > 1 ngày không chạy được, chuyển sang SalUn repo và lấy SCRUB từ meghdadk/scrub-reimplemented hoặc RoyRin/scrub-reimplemented. Ghi quyết định vào RUN_NOTES.md.

### 0B.2 Thay đổi patch tối thiểu (giữ nguyên phần còn lại của repo)

Chỉ 2 patch, không hơn:

1. **Trajectory hook:** sau mỗi `eval_interval` bước unlearning (đặt 25 bước, hoặc 1 epoch nếu method chạy theo epoch), tính `L_R, L_D, L_V, A_R, A_D, A_V` bằng hàm eval có sẵn của repo trên **full** tập tương ứng, append vào `traj.jsonl` (schema §6), và `torch.save` checkpoint thứ k vào `ckpt/`. Với SalUn repo: hook vào `unlearn()` loop; với deepunlearn: hook vào vòng unlearn tương ứng. Patch phải là thêm-mới (không sửa logic loss/optimizer của method gốc).
2. **RALM scorer:** module độc lập đọc `traj.jsonl`, chạy đúng công thức §0.2 (offline mode, EMA off mặc định), xuất `selected_k` cho 3 biến thể ralm-d/v/dv + baselines §8.

**Không patch:** ρ ngầm mặc định của repo, recipe gốc của method, MIA code. Sau patch, chạy `git diff` phải cho thấy method gốc bất động ngoài hook.

### 0B.3 Ma trận chạy tối thiểu của Track A

| Thành phần | Giá trị |
|---|---|
| Dataset / model | CIFAR-10, ResNet-18 (chuẩn của cả 2 repo) |
| Forget set | random 10% (mặc định repo, dễ đối chiếu số) + random 1% (nơi 1/ρ amplification hiện ra) |
| Methods | GA, NG+, SCRUB (3 trajectory-type khác nhau: ascent-only, mixed, teacher-student) |
| Seeds | 3 base seed (0,1,2) × 3 method × 2 forget ratio; retrain 3 seed mỗi config (repo có sẵn pipeline retrain) |
| Eval interval | 25 bước unlearning, tối đa 500–2000 bước theo method |
| Stopping rules | ralm-dv (chính), ralm-d, retain-only, val-only, fixed-{100,500,last}, random-k |
| Test set | mở một lần duy nhất cuối Track A |

Ước tính GPU Track A: base models 3 × ~1.5h; trajectories 18 × ~0.5–1h; retrains 12 × ~1.5h → **~35–50 giờ một GPU**. Nếu thiếu, cắt theo đúng thứ tự: seeds 3→2, SCRUB giữ (không cắt, vì là method ít bị over-forget nhất).

### 0B.4 Gate của Track A (đơn giản hóa từ §12)

1. Patch chạy end-to-end 1 config (GA + random 10%) không crash; traj.jsonl khớp identity `L_D=(1−ρ)L_R+ρL_F` ±1e-5.
2. 1 file config đầy đủ, ralm scorer ra `selected_k` đúng nơi min của J (unit test mini 5 điểm như §7).
3. Xem plot regret curve + overlay J (như smoke §12, go/no-go theo cùng tiêu chí: k̂ trong ±100 steps quanh k* hoặc regret < fixed-100).
4. Pass → chạy toàn bộ ma trận 0B.3 → `results/summary_trackA.csv` + report theo template §15, nhưng cho phép rút gọn: chỉ cần (a) bảng regret theo rule, (b) scatter k̂ vs k*, (c) câu trả lời RQ1 + câu H4-band coverage (§0A.7 chỉ số 1 và 3).

### 0B.5 Đầu ra của Track A quyết định gì

- ralm-dv thắng hoặc ngang baselines ở cả ρ=1% và 10% → tin cậy để mở Track B đầy đủ (Phase 1–3, T-series, ablations).
- ralm-dv chỉ thắng ở ρ=10% hoặc chỉ ở GA → kết quả dọc H3, mở Track B ở settings đó trước.
- ralm-dv thua `retain-only` ở cả 2ρ → NO-GO theo §12, viết report_smoke và dừng.

---

## 1. Quy tắc bất di bất dịch [FIXED toàn mục]

1. **Retrain-only-for-eval:** θ_R chỉ tồn tại trong module đánh giá. Nghiêm cấm: dùng θ_R để chọn k̂, chọn λ, dừng sớm, hoặc tính J. Tách code vật lý: `retrain.py` và `eval_oracle.py` không được import trong `collect_traj.py` / `score_ralm.py`; thêm assert kiểm tra trong đầu mỗi file này.
2. **Test set chỉ mở một lần** cuối mỗi phase, ghi dấu giờ mở vào log. Metrics dùng test đặt cờ `post_selection=true` trong jsonl, và module `score_ralm.py` có assert từ chối đọc cột này (nếu phải đọc, raise lỗi).
3. **Log liên tục:** mọi số liệu append jsonl ngay khi tính xong, flush mỗi snapshot. Không giữ full trajectory trong RAM đến cuối.
4. **Reproducibility:**
   - F/R split cố định theo manifest (§5), không phụ thuộc seed.
   - Seed chạy = (seed_train_base, seed_unlearn, seed_retrain). Giai đoạn I dùng seed ∈ {0,1,2,3,4}; seed_unlearn = seed_train_base để giảm tổ hợp.
   - `torch.use_deterministic_algorithms(True)` nếu không làm chậm quá 2×; nếu có nondeterminism, định lượng nó trong report (chạy 2 lần same seed, báo max절대 lệch metrics).
   - Lưu `torch.save(model.state_dict())` của θ0, các checkpoint trajectory (nếu disk đủ, gate mỗi 50 steps; từ checkpoint giữa chỉ lưu số liệu), và θ_R mỗi retrain seed.
5. **Loss measurement:** full forward qua toàn tập X, `torch.no_grad()`, batch lớn nhất vừa GPU, `model.eval()` (không dropout... ResNet-18 chuẩn không có dropout nhưng giữ thói quen), **không augmentation** khi đo loss anchor. Ghi rõ trong code comment: "augmentation chỉ khi train".
6. **OOM:** giảm eval batch × ½, tối đa 3 lần; nếu vẫn fail, giảm snapshot frequency thay vì đổi recipe. Không bao giờ đổi lr/batch recipe để sửa OOM.
7. Mọi kết luận trong report phải dẫn tới file log + dòng dữ liệu (vd `logs/traj_ga_c10_10pct_seed0.jsonl:L520`).
8. **Không tự "cứu" kết quả xấu** bằng cách đổi tham số cho đến khi đẹp. Kết quả xấu im lặng vẫn là kết quả.

---

## 2. Cấu trúc repo và hạ tầng

```
ralm/
  data/
    forget_manifest_*.json   # §5,ingerprint của từng F set
    base_stats_*.json        # L^0, A^0 của θ0 cho từng seed
  ckpt/
    base/seed{s}.pth
    traj/{method}/{dataset}/f{type}_rho{r}_seed{s}/step{0100..}.pth   # gate: giảo lưu mỗi 50 thì lưu model, mỗi 10 thì chỉ lưu metrics
    retrain/{dataset}/f{type}_rho{r}_seed{s}.pth
  logs/
    traj_*.jsonl             # trajectory metrics theo schema §7
    retrain_combined.jsonl
  results/
    summary_{phase}.csv
    report_smoke.md / report_phase1.md ...
    figs/
  src/ (xem §2.1)
  RUN_NOTES.md
  CHANGES.md
```

### 2.1 Module và API bắt buộc

```
src/
  config.py            # mọi hằng số [FIXED] ở đây, không hard-code rải rác
  manifest.py          # sinh/load forget manifest, Built H2 lệch split FALLBACK
  datasets.py          # cifar10/cifar100/tinyimagenet + split trả training(D đậm F), V sẵn manifest F
  train_base.py        # huấn luyện θ0, chạy base_eval, ghi base_stats
  unlearning/ga.py     # gradient ascent trajectory
  unlearning/ngp.py    # negative gradient plus
  unlearning/scrub.py  # SCRUB
  collect_traj.py      # orchestrator: load θ0 + manifest, chạy unlearning method, snapshot
  eval_core.py         # eval_losses(model, X, batch) -> dict; eval_logits; kl_divergence_between(modelA, modelB, X)
  score_ralm.py        # từ jsonl → J variants → k̂; cả offline lẫn hand-off online engine
  stopping/rules.py    # interface StoppingRule.select(trajectory_metrics) -> k̂
  eval_oracle.py       # từ θ_R + trajectory metrics → regret, G_logit (CHỈ file này được load θ_R)
  mia.py               # population MIA + confidence gap (§10.3)
  aggregate.py         # csv + plots
  tests/test_score_ralm.py
  tests/test_manifest.py
  tests/test_eval_core.py
```

### 2.2 Environment

- Python 3.10+, PyTorch ≥ 2.1, torchvision, `numpy`, `pandas`, `matplotlib`, `scipy`, `tqdm`. Cài qua yêu cầu `requirements.txt` do agent tự sinh.
- GPU hiện có: ghi vào RUN_NOTES.md số GPU, VRAM, CUDA version. **Ước lượng runtime trước khi chạy** (§13) và báo user trước khi nhấn chạy các giai đoạn > 12 giờ GPU.
- Log training bằng `print` + file, không cần wandb/tensorboard (optional).

### 2.3 Dựng pipeline từ đâu

- **Lựa chọn ưu tiên 1 (Track A, §0B):** bắt đầu từ [`xcadet/deepunlearn`](https://github.com/xcadet/deepunlearn), chỉ thêm 2 patch (trajectory hook + RALM scorer) như §0B.2. Fallback: [`OPTML-Group/Unlearn-Saliency`](https://github.com/OPTML-Group/Unlearn-Saliency) (SalUn) và [`meghdadk/SCRUB`](https://github.com/meghdadk/SCRUB), hoặc bản reimplement [`RoyRin/scrub-reimplemented`](https://github.com/RoyRin/scrub-reimplemented) nếu official SCRUB không chạy nổi.
- **Lựa chọn 2 (Track B):** dựng pipeline tối giản tự viết (mục dưới đây) khi cần kiểm soát đầy đủ (T-series per-sample loss, band-check, per-class anchor T10).

---

## 3. Huấn luyện θ0 (base model)

### 3.1 Recipe [FIXED cho stage I–II]

| Thành phần | Giá trị CIFAR-10/100 | Tiny ImageNet |
|---|---|---|
| backbone | ResNet-18 | ResNet-18 |
| optimizer | SGD, momentum 0.9, wd 5e-4 | như bên |
| lr / schedule | 0.1, cosine, warmup 5 epoch | 0.1, cosine, warmup 5 |
| epochs | 100 | 100 |
| batch | 128 | 128 |
| augmentation | RandomCrop(32, pad=4), RandomHorizontalFlip | RandomCrop(64, pad=4), flip |
| label smoothing | 0 | 0 |
| input norm | chuẩn CIFAR (cifar10_mean/std) | chuẩn chung |
| chia V | **index cuối **[40.000:45.000]** của CIFAR-10 train sau shuffle cố định seed=12345 | tương tự |

Chỉ coi θ0 đạt yêu cầu nếu:
- test acc ≥ 93% (CIFAR-10) / ≥ 70% (CIFAR-100); nếu không đáp ứng, **dừng và báo user**, không tự đổi recipe.
- Ghi nhận exact test acc vào base_stats.

### 3.2 Flow chi tiết `train_base.py` (pseudo-code)

```python
def main(seed, dataset):
    set_all_seeds(seed)
    full = load_dataset(dataset, train=True)
    # SPLIT FIX: v_frozen = 5000 cuối theo permutation seed=12345
    perm = np.random.RandomState(12345).permutation(len(full))
    v_idx, d_idx = perm[-5000:], perm[:-5000]
    D = Subset(full, d_idx); V = Subset(full, v_idx)
    model = resnet18(num_classes).to(dev)
    train_loop(model, D, epochs=100)           # recipe §3.1
    save(state_dict, f"base/seed{seed}.pth")
    stats = eval_full(model, {"D": D, "V": V, "T": test})
    # stats lưu: L_X, A_X, mean_logit_margin_X, per-sample logits+targets (fp16, một lần)
    stats.save("base_stats_seed{seed}.json")
```

**Bắt buộc:** snapshot per-sample logits + labels của θ0 trên toàn D (45.000 mẫu) → file `d_logits0_seed{seed}.npy`. Đây là nguồn chọn forget set high-loss / memorized, và là nguồn đáng tin để tính lại `L^0` chính xác ở mọi thời điểm.

### 3.3 base_stats.json schema

```json
{
  "seed": 0,
  "dataset": "cifar10",
  "L_D0": 0.0518, "L_V0": 0.3001, "L_T0": 0.3122,
  "A_D0": 1.0000, "A_V0": 0.9198, "A_T0": 0.9312,
  "mean_margin_F": null,
  "per_sample_logit_file": "d_logits0_seed0.npy",
  "timestamp": "...", "resume_recipe": "§3.1"
}
```

Có thể có `L_I0` (forget set ở phase I: L_F0) nul, được tính sau khi manifest chốt.

---

## 4. Forget set manifest — cách dựng chi tiết

### 4.1 Bảng dựng (CIFAR-10)

| Loại file | Cách dựng chính thức | ρ áp dụng | Vì sao cần |
|---|---|---|---|
| `random` | chọn uniform không replacement từ D | 0.001, 0.01, 0.1 | baseline; original ≈ retrain gần |
| `highloss` | lấy top-ρ% | của CE loss per-sample tại θ0, sort giảm dần | influential khó (lưu ý: samples loss cao thường là hard/noisy) |
| `memorized` | top-ρ% theo train margin \( y_{true}·z_{true} − max_{c≠true} z_c \) **cao nhất** tại θ0 | 0.01 | samples bị memorize sâu; dấu vết cốt lõi của privacy |
| `classwise` | toàn bộ sample của class `airplane` | ≈0.1 (45,000/10≈4.500) | stress test large-ρ |
| `classwise-hard` | toàn bộ class `cat` | ≈0.1 | class khó, để phân biệt do class在那里 hay không |

CIFAR-100 thêm: `classwise` (1 class, ρ≈0.01) và `superclass` (1 superclass, 5 class, ρ≈0.05).

### 4.2 Định dạng manifest

```json
{
  "dataset": "cifar10",
  "type": "highloss",
  "rho": 0.01,
  "base_model_seed": 0,
  "source_logits": "d_logits0_seed0.npy",
  "n_forget": 450,
  "d_indices": [19023, 3381, ...],
  "v_indices_intersect": [],
  "seed": 12345,
  "created_from_recipe": "§4.1"
}
```

- Khai báo `v_indices_intersect` luôn [] (verify bằng assert **F∩V=∅** và **F⊂D** khi load).
- Cùng 1 manifest dùng cho mọi seed unlearning, chỉ con build dựa trên θ0 của seed tương ứng (có nghĩa: với highloss/memorized, mỗi seed có một manifest riêng vì θ0 khác nhau; random/classwise dùng chung cho śawait mọi seed). Đánh dấu rõ `basis: "per-seed"` hay `"global"`.

### 4.3 Sanity checks khi build (bắt buộc, assert trong `manifest.py`)

1. F ∩ V = ∅, F ⊂ D, |F| = round(ρ·|D|).
2. Với classwise: kiểm tra class分布 của R không lệch quá 2 điểm % so với khi bỏ ngẫu nhiên (in báo cáo thôi, không block).
3. L_F0 tính tại θ0 trên các sample của manifest phải khớp mean per-sample loss từ `d_logits0` (tolerance 1e-4).
4. Kiểm tra full check trùng: F không được chứa bất kỳ sample nào mà V dùng (trùng index, không phải trùng ảnh).

---

## 5. Thuật toán unlearning — chi tiết từng phương pháp

### 5.1 GA (Gradient Ascent) — phương pháp chính cho stage I

```python
optimizer = SGD(model.parameters(), lr=args.lr_ul, momentum=0.0)
for step in range(n_steps):                 # n_steps = 2000
    x, y = next_forget_batch(F, shuffle_seed=seed_ul)
    loss = -CE(model(x), y)                 # ASCENT trên F
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    if (step+1) % 10 == 0:
        snapshot(model, step+1)             # eval §7, append jsonl
```

- `lr_ul`: dùng `1e-3` [FIXED stage I]. Dành stage smoke: quét {1e-4, 1e-3, 1e-2} trên 1 seed chọn theo tiêu chí **V-loss không tăng quá 20% tại step 200** (chọn bằng V, không phải test, không phải retrain).
- Batch forget: 128. Với ρ=0.001 (135 mẫu), batch 128 gần như full-batch trên F gần như = F — GIỮ nguyên, nhưng note trong report rằng GA gần như sa chờ đợi chút ít access pattern.
- Momentum 0 (một vài paper dùng 0.9; giữ 0 để trajectory đơn giản hơn).

### 5.2 NG+ (Negative Gradient Plus)

```python
for step in ...:
    loss = -CE(model(xF), yF) + α·CE(model(xR_batch), yR_batch)
```

- `α = 1.0`, batch R 128, batch F 128 (pairing per-step); lr điều chỉnh giữ cùng khoảng cách bước so với GA: scan nhanh {1e-4, 1e-3} trên 1 seed nếu smoke §12 không chạy nổi.
- Snapshot cùng cadence.

### 5.3 SCRUB (tối giản gọn)

```python
max_steps = 2000, max_epochs_s = 3 (số vòng sweep R)
for epoch in range(3):
    for (xR, yR), (xF, yF) in zip(loader_R_shuffle, loader_F_cycle):
        L_m = KL(p_s(xR) || p_t(xR)) + CE_s(xR, yR)          # retain match
        L_kl = KL(p_s(xF) || p_t_adv(xF).detach())           # forget push
        loss = L_m + γ·L_kl
```

- `γ = 0.5`, lr 5e-4, teacher là θ0 copy (không gradient), không EMA teacher để tránh khớp nhiều dựa vào teacher quá.
- Follow as close as possible SCRUB reference; nếu code official quá phức tạp, agent implement bản "SCRUB-lite" trên 2 loss bên trên, ghi chú rõ vào RUN_NOTES.md rằng không bản chính thức.

### 5.4 Quy định snapshot

- Cứ 10 bước unlearning → 1 snapshot metrics. Cứ 50 bước → snapshot model `.pth` (stage I chỉ; stage II có thể thưa hơn).
- Với 2000 steps: 200 snapshot metrics, 40 model checkpoints.
- Mỗi snapshot metrics tính được các mảng Sau: D, R, F, V, T (T chỉ khi cờ `allow_test=True`).

---

## 6. Trajectory metrics — schema jsonl chi tiết

Mỗi dòng log (`logs/traj_{method}_{dataset}_{ftype}_rho{r}_seed{s}.jsonl`):

```json
{
  "run_id": "ga_cifar10_highloss_rho0.01_seed0",
  "step": 120,
  "wall_time_s": 93.4,
  "L_R": 0.2645, "L_D": 0.3512, "L_F": 2.1041, "L_V": 0.3098,
  "A_R": 0.9901, "A_D": 0.9212, "A_F": 0.1503, "A_V": 0.9155,
  "L_T": null, "A_T": null,
  "mean_conf_F": 0.2231,
  "grad_norm_last_step": 0.9112,
  "lr": 0.001,
  "batch_indices_last_step": [12, 888, ...],
  "post_selection": false
}
```

- **Bắt buộc:** thêm dòng step=0 (chính là θ0/nilai ref) như dòng đầu tiên.
- `L_T/A_T` chỉ có trong file `*_test_metrics.jsonl` riêng, mở ở cuối phase.
- **module `eval_core.eval_full`** phải tính theo đúng thứ tự: R → D (sau R và F tự có được từ D + F) → F → V. Việc tính L_D trực tiếp trên D và tính L_R + (1−ρ)L_R + ρL_F phải sai lệch < 1e-5 (sanity tự động, raise nếu lệch hơn). Đây là mẹo bắt lỗi cho cả pipeline split.

### 6.1 Test online

- Trong run online (patience-based), snapshot cadence 10 steps, sau khi dừng, lấy k̂ đã chọn rồi tính test metrics riêng cho k̂ duy nhất đó. Không tính test trên độc checkpoint chưa được chọn.

---

## 7. RALM scoring engine — pseudo-code đầy đủ

```python
# src/score_ralm.py

def compute_scores(traj: list[dict], base: dict, ema_beta: float | None = 0.9,
                   k_min: int = 100, lamD: float = 1.0, lamV: float = 1.0) -> dict:
    eps = 1e-12
    LR0, LD0, LV0 = base["L_R0"], base["L_D0"], base["L_V0"]

    # 1) optional EMA smoothing
    if ema_beta:
        for key in ["L_R", "L_D", "L_V"]:
            traj = apply_ema(traj, key, beta=ema_beta)   # EMA theo thứ tự step

    # 2) component scores
    for m in traj:
        m["S_R"] = m["L_R"] / (LR0 + eps)
        m["S_D"] = abs(m["L_D"] - LD0) / (LD0 + eps)
        m["S_V"] = abs(m["L_V"] - LV0) / (LV0 + eps)
        m["J_DV"] = m["S_R"] + lamD*m["S_D"] + lamV*m["S_V"]
        m["J_D"]  = m["S_R"] + lamD*m["S_D"]
        m["J_V"]  = m["S_R"] + lamV*m["S_V"]

    # 3) selection with k_min + optional delta_F gate
    cand = [m for m in traj if m["step"] >= k_min]
    out = {}
    for jname in ["J_DV", "J_D", "J_V"]:
        out[jname] = min(cand, key=lambda m: m[jname])["step"]
    return out
```

- Ở **offline mode** (chính cho giai đoạn đánh giá): chạy trên jsonl trọn vẹn trajectory sau cùng.
- Ở **online mode** (patience): cùng một module phát event; agent implement cả 2, nhưng offline là mặc định để phân tích nhân quả.

- **Unit test bắt buộc `tests/test_score_ralm.py`:**
  1. Kèm trajectory hằng số J (flat) → chọn k lớn nhất hoặc ổn định (không crash).
  2. Trajectory có min ở trước k_min → chọn đúng min sau k_min.
  3. Chọn kỳ vọng J như tính tay trong 1 case mini (5 điểm) khớp 6 số thập phân.
  4. EMA(β=0.9): kiểm tra tay với dãy 5 số.
  5. δ_F gate: khi không checkpoint民航 vượt δ_F → trả về None + báo log cảnh báo (không crash).

---

## 8. Stopping rules — interface và các baselines

```python
# src/stopping/rules.py
class StoppingRule:          # chạy online trên trajectory streaming
    def on_snapshot(self, m: dict) -> None: ...
    def should_stop(self) -> bool: ...
    def selected_step(self) -> int: ...
```

| Tag | Rule | Định nghĩa chuẩn (viết như thế này trong code) |
|---|---|---|
| `fixed-{s}` | checkpoint cố định | `selected = s` (không eval gì thêm), s ∈ {50,100,200,500,1000,2000} |
| `retain-only` | `argmin_k L_R(k), k ≥ k_min` | baseline B |
| `val-only` | `argmin_k L_V(k), k ≥ k_min` | baseline C |
| `pareto-knee` | chuẩn hóa cả (L_F rising, L_V rising); chọn knee = điểm max khoảng cách từ đường (k-model, min-max curve) lên đường chéo | baseline D. Viết chính xác: knee trên đường cong (u_k = norm(L_V(k)), v_k = norm(max(0, L_F(k)−L_F0+δ))) bằng hàm `kneedle`. Nếu knee-detection thất bại → dùng `max(u+v)`. Ghi rõ version knee code. |
| `random-k` | seed=0..4, chọn step uniform trên {k_min..2000} | control, 5 mẫu/seed cấu hình |
| `ralm-d`, `ralm-v`, `ralm-dv` | §7 | đề xuất chính |

- `patience p = 5` snapshots, đánh giá mỗi 10 steps [FIXED]; grid p ∈ {3,5,10} chỉ ở ablation E4.
- Cost cả các Online dùng chung: muối số forward pass của `should_stop` được đếm và ghi vào summary. Retain-only/val-only chỉ cần 1−2 tập; **ralm full cần D + V mỗi snapshot → đắt hơn ~3×.** Đây là một đánh trade-off cụ thể mà phải bỏ vào report.

- **warm-up `[δ_F]`:** chỉ nhận k nếu `L_F(k) ≥ L_F(0) + δF` với `δF = 0.05·L_F^0` [FIXED]. Được bật/tắt theo config; OFF cho thuốc chính khi chờ ablation; nếu bật và mọi k không pass, chọn checkpoint có `L_F` lớn nhất (nêu trong report).
  - Chú ý: RALM **không sử dụng L_F trong J**; δF chỉ dùng như cổng sanity chống k̂≈0. Từ stage II sử dụng cấu hình Default ON.

---

## 9. Retrain oracle — protocol chi tiết

```python
# src/retrain.py
for rseed in [0, 1, 2, 3, 4]:   # stage III có thể 3 seed
    set_all_seeds(rseed)
    model = resnet18(num_classes)
    train(model, R_split, recipe = §3.1, epochs=100)
    eval_full → per-set metrics lg G_logit so với từng trajectory ckpt
    save retrain/{dataset}/f{type}_rho{r}_seed{rseed}.pth
```

- Batch size có thể lớn hơn 256 nếu GPU cho phép; ghi rõ nếu đổi.
- **Chú ý đánh giá θ_R:** tính đủ L_R, L_F, L_V, L_D, L_T và **logits** trên {F, V, T, một phần ngẫu nhiên R 10.000}. Đầu ra `retrain_logits_*.npy` phục vụ G_logit batch.
- Ghi rõ `L_D(θ_R)` sẽ **khác** `L_D^0`, thường tăng vì model chưa thấy F. Đây chính là điểm vòng vòng của ý tưởng mà mình cần defensively đo: mỗi run phải in `L_D(θ_R) − L_D^0`; nếu anchor D kéo trajectory về gần θ0 xa θ_R thì đây chính là bằng chứng shores cho H3.

---

## 10. Đánh giá — nơi kết nối θ_R (chỉ `eval_oracle.py` có thể load θ_R)

### 10.1 Regret gap metrics

Tại mỗi ckpt k ∈ trajectory (cả 10-step grid metrics, dùng model `.pth` tại 50-step gate; metrics 10-step từ logits nếu có lưu): 

| Metric | Công thức |
|---|---|
| `G_R` | \|L_R(θ_k) − L_R(θ_R)\| (khớp mean qua retrain seeds: dùng trung vị retrain seeds) |
| `G_F` | \|L_F(θ_k) − L_F(θ_R)\| |
| `G_test_loss` | \|L_T(θ_k) − L_T(θ_R)\| |
| `G_test_acc` | |A_T(θ_k) − A_T(θ_R)| (điểm %) |
| `G_logit_F` | mean_{x∈F} KL(p_{θ_R} ‖ p_{θ_k}) [FIXED làm oracle selection metric] |
| `G_logit_V` | như trên với V |

**Chi tiết KL tính [FIXED]:** dùng mật độ soft (softmax với p=1, không tối đêm), tính per-sample KL(θ_R‖θ_k) = Σ_c p_R,c · (log p_R,c − log p_k,c), trung bình trên sample. Không temperature. logits giữ float32 khi tính KL, dù lưu disk dạng fp16.

### 10.2 Oracle và regret

- `k* = argmin_k G_logit_F(θ_k)` trên trajectory [FIXED].
- `Regret(k̂) = G_logit_F(θ_k̂) − G_logit_F(θ_k*)`.
- `E_step = |k̂ − k*|`. (Kèm thêm `|rank(k̂) − rank(k*)|/n_snapshots` vì trục step không uniform ý nghĩa.)
- **Median + IQR trên 5 seed** cho mọi số liệu tổng hợp; không chỉ mean (tránh outlier seed φ).

### 10.3 Privacy proxy (đơn giản nhất chấp nhận được)

- **Population MIA:** classifier là logit-gap θ(x, y) trên F vs một tập holdout random ngoài D (gap = z_y − max_{c≠y} z_c). Train a small logistic classifier hoặc chỉ report AUC của gap trực tiếp. Báo cáo **AUC của gap** làm metric chính.
- **U-LiRA (nếu compute cho phép):** dùng code từ implement tham khảo theo paper U-LiRA hoặc LiRA per-sample (n shadows) trên 500-1000 forget samples; nếu không chạy nổi, đánh dấu rõ `privacy: population-only` trong summary.
- Đưa thêm: `mean_conf_F` và `A_F từ jsonl` làm quick red flag (phải vào report, không phải oracle).

---

## 11. Ma trận chạy và ước lượng compute

### 11.1 Stage I (cứng, bắt buộc)

| Thành phần | Số run | Ước tính |
|---|---|---|
| base θ0 train | 5 seed × 100 ep | 5 × ~1.5 h GPU |
| forget manifests | 5 loại × vài ρ | không tốn GPU |
| GA trajectory | 3 random-ρ + high-loss + memorized + classwise(×2 class) = 7 config × 5 seed = **35 runs** | 35 × ~2.000 steps × (train step + eval D+R+F+V ~30s/batch lớn) |
| retrain | 7 config × 5 seed × 100 ep on R-only | ~7 × 5 × 1.5 h GPU |

Tổng GPU stage I ước tính: **~ 35-60 giờ** trên 1 GPU (đánh số chính xác từng run trước khi commit). Nếu chỉ có 1 GPU < 24 h budget → cắt: chỉ giữ random-1% + highloss + classwise-airplane trước, random-0.1% chuyển stage II.

### 11.2 Stage II

+ NG+ và SCRUB trên random-ρ ∈ {0.001, 0.01, 0.1} và highloss. +7 config × 5 seed, thêm retrain mirrors.

### 11.3 Stage III

- CIFAR-100 (backbone ResNet-18): random/classwise/superclass.
- Tiny ImageNet (64×64, 200 class, ResNet-18): random-store 5%, ρ=0.01.
- CIFAR-100 + ViT-Tiny (nếu compute còn): random, classwise.
- Mục đích chính: chuyển giao λ, k_min, patience (điều kiện thành công #4). Giữ cùng helper code; chỉ đổi dataset config.

---

## 12. Kế hoạch giai đoạn và các gate

### Gate 0 — setup (bài kiểm tra 2-3 giờ compute)

1. Mọi module sanity + unit tests xanh (pytest). Chạy: `pytest src/tests -x -q`.
2. Train-base 2 epochs để xác nhận end-to-end không crash, base_stats xuất ra đúng schema.
3. Sinh 1 manifest random ρ=0.01, verify sanity checks §4.3.
4. Chạy GA 100 steps snapshot 10 → jsonl schema đúng 100%, các L khớp identity 6.4 (±1e-5).
5. score_ralm chạy tiếp được trên file jsonl đó.

**Nếu gate 0 fail: dừng trước khi dùng GPU thật.**

### Gate 0.5 — SMOKE TEST (go/no-go, bắt buộc)

- Config [FIXED]: seed=0, CIFAR-10, ResNet-18, forget=random ρ=0.01, GA lr=1e-3, 1.000 steps, snapshot 10 → 100 snapshot; retrain 1 seed.
- Trả lời bằng plot:
  1. Regret curve `G_logit_F(k)` (cần θ_R) — hình dạng tiêu chuẩn sẽ thấp lên rồi ổn, hoặc tăng đơn điệu sau khi quá quên.
  2. Overlay J_DV(k), J_D(k), J_V(k), L_R(k), L_V(k) với min được đánh dấu.
- **GO** nếu: `k̂(ralm-dv)` nằm trong ±100 steps của `min(G_logit_F)` hoặc regret của ralm-dv nhỏ hơn regret của `fixed-200` và `random-k`.
- **NO-GO** nếu: ralm-dv chọn k̂ = k_min luôn, hoặc correlation giữa J_DV(k) và G_logit_F(k) là dương (J thấp = tệ).
- **Khi NO-GO:** viết `results/report_smoke.md` ≥ 500 từ, liệt kê 3 giả thuyết (đo lường lỗi / giả thuyết sai / metric không tốt) kèm chứng cứ, báo user, **không chạy tiếp stage I.**

### Phase 1 — full GA grid (stage I)

Chạy ma trận §11.1, aggregate lên `summary_phase1.csv`, report trả lời RQ1, RQ2. Plot: regret-curve, k̂-scatter, box plot.

### Phase 2 — NG+, SCRUB (stage II) + ablations A1–A4, E1–E4 trên random ρ=0.01, seed={0,1,2}. Phase 1 phải pass điều kiện thành công ≥ 2/4 trước khi mở Phase này.

### Phase 3 — CIFAR-100 + Tiny ImageNet + transfer (stage III). Phase 2 phải có kết luận rõ ràng (dù là h3 hoàn toàn) trước khi mở.

---

## 13. Ablations chi tiết

| ID | Câu hỏi | Thiết lập | Số run |
|---|---|---|---|
| A1 | S_D có thông tin? | ralm-d vs retain-only | 0 (chỉ re-analysis) |
| A2 | S_V có thông tin? | ralm-dv vs ralm-d | 0 (chỉ re-analysis) |
| A3 | Loss vs Accuracy | ralm-dv-acc: thay L bằng A trong cả 3 thành phần (A không reduced trừ khi loại) | 0 (re-analysis trên A已有的 trong jsonl) |
| A4 | Nhiễu ở ρ nhỏ | ρ=0.001, tính σ của (L_D(k) − L_D^0) qua 5 measurement subsample mỗi ckpt (chia D thành 5 folds không trùng + random); so σ vs ngưỡng anchor | 3 seed |
| B1 | label noise trong F | flip 10% / 20% labels trong F trước unlearning | random ρ=0.01, 2 seed |
| C1 | θ0 underfit | 25% / 50% / 100% / 150% epochs | 3 seed × 4 recipe |
| E1 | k_min | {50, 100, 200} | re-analysis |
| E2 | EMA beta | {none, 0.9, 0.99} | re-analysis |
| E3 | λ (chẩn đoán) | {0.5, 1, 2} λ_D=λ_V | 1 seed |
| E4 | patience | {3, 5, 10} | 3 seed random-1% |
| E5 | band tolerance ε | ε_D, ε_V ∈ {0.025, 0.05, 0.10}, chọn theo path-width 0A.7 | re-analysis |
| T0–T10 | Anchor nào chứa thông tin? | họ anchor phi tuyến §0A.4 + anchor per-class T10 (§0A.8), re-analysis trên trajectories Phase 1, tiêu chí 0A.6.2; gồm cả corr diagnostic §0A.5 và band-check H4 §0A.7 | re-analysis (0 GPU chạy mới) |

Mỗi ablation kết thúc một section nhỏ trong `report_phase2.md` với 1-2 câu kết luận + plot/link log.

---

## 14. Tiêu chí thành công / bác bỏ (đánh như vậy trong report)

### 14.1 Thành công (cả 4, tự động tính trong `aggregate.py`)

1. **Regret:** median regret ralm-dv ≤ 10% (max−min) của G_logit-F trên trajectory; và ≤ median regret của `fixed-200`, `retain-only`, `random-k` (so từng cặp với sign test n=5, threshold p<0.1 como level rough).
2. **Utility:** |A_T(k̂) − A_T(θ_R)| ≤ 1 điểm % (CIFAR-10) / ≤ 2 (CIFAR-100).
3. **Forget behavior:** median G_F(k̂) ≤ median G_F toàn trajectory (theo method).
4. **Transfer:** giữ λ=1, k_min=100, p=5 nguyên vẹn, median regret trên CIFAR-100 có σ tăng ≤ 2× so với CIFAR-10.

### 14.2 Bác bỏ (bất kỳ điều nào)

- ≥ 50% run chọn `k̂ = k_min`;
- median regret ralm-dv **không** tốt hơn `random-k`;
- `G_logit_F(k̂) > 3 × G_logit_F(fixed-200)` trung bình (RALM giữ utility nhưng còn memory trace, trong khi fixed-200 không);
- kết론 transfer cần λ hoặc k_min khác > 2× so với CIFAR-10.

### 14.3 Đánh giá nửa vời (partial success — vẫn là kết quả đáng báo cáo)

- RALM thắng ở ρ ≤ 1% random, nhưng thua ở classwise/ influential (dự đoán từ H3). **Báo cáo phân giới này là kết quả, không phải failure.**

---

## 15. Sản phẩm đi vào report — template

```markdown
# Report Phase 1 — RALM trên CIFAR-10 + GA

## 1. Tổng quan
...số runs, seed, thời gian, hardware...

## 2. Bảng chính
| Rule | median regret | IQR | median k̂ | % k̂=k_min | online cost (fwd pass) |
| (mỗi dòng 1 rule) |

## 3. Kết luận từng RQ
- RQ1: ...
- RQ2: ...  (có bảng theo ρ và forget type)

## 4. Mối lo pipeline
- có run nào đó bất thường, data leakage check, identity L_D check pass không.

## 5. Hạn chế & next steps
```

Bảng plot chuẩn (`results/figs/`):
- `regret_curve_{config}.png` — từng rule là một marker dọc tại k̂ của nó.
- `scatter_khat_vs_kstar_{config}.png` — đường y=x reference, các điểm bị tụ về k_min toả màu đỏ.
- `box_regret_by_rule_{phase}.png`.
- `sd_vs_glogit_{rho}.png` — scatter S_D(k) vs G_logit_F(k) tại ρ nhỏ, có correlation.

---

## 16. Troubleshooting và mấy con mắt cảnh giác

- **Regret ≈ 0 kì lạ / k̂ = k* luôn:** kiểm tra data của checkpoint metrics có phải bị tính sau khi load θ_R không (vi phạm quy tắc §1.1); kiểm tra KL có đang tính θ_k chính nó.
- **J chỉ giảm monotonic, k̂ = 2000:** xem anchor `S_D` có bị trôi vì L_D thay đổi quá chậm so với LR; đây là case H3 tăng cường — report và thử ablation E3 (λ diagnostic).
- **Mất memory khi D=45.000 forward:** dùng `torch.inference_mode()`, eval batch 512, dim float32 logits chỉ giữ mean cần; per-sample logits chỉ trên F + sample 2k của D/R.
- **L_F tăng quá mạnh ngay từ step 20 (GA ascent quá mạnh, khả năng lr quá cao):** vẫn không dừng — chỉ ghi. Đây là đặc trưng của GA; RALM đương nhiệm là chọn min J, không phải ngăn over-unlearn.
- **Identity \(L_D = (1−ρ)L_R + ρL_F\) lệch > 1e-5** : nghĩa là có bug trong DataSplit — dừng ngay, fix trước khi tiếp tục. (đây cũng chính là lý do giữ identity này làm bài kiểm tra logic nội bộ của pipeline.)
- **Mọi nơi có hard-code number** phải đi qua `config.py`; CI đơn giản: grep sơ sài就医 quy tắc.

---

## 17. Mẫu câu agent dùng khi báo cáo

- "RALM-DV chọn k̂ = X, cách oracle-best k* = Y bước, median regret Z (IQR A-B) so với random-k median W."
- "Với ρ=0.001, σ của ΔL_D measurement = ±c, trong khi tín hiệu cần phân biệt ≈ d ⇒ anchor D [đủ/thiếu] precision (ablation A4)."
- "ralm-dv giữ G_test tốt nhưng G_logit_F = 3.2× fixed-200 ⇒ hành vi nhớ F chưa xóa ⇒ đây là case báo H3 (partial success)."
- "Framing chuẩn: RALM là chọn-k rule không cần oracle, **không phải** estimator của retrain."

## 18. Điều agent không được làm (liệt kê nghiêm cấm)

1. Grid search λ, lr, k_min rồi chọn cấu hình đẹp nhất như kết quả "chính".
2. Dùng test set trong bất kỳ vòng chọn nào trước Gate "mở test".
3. Bỏ qua một eval snapshot bị fail — phải cố xảy ra lỗi, không skip im lặng.
4. Báo cáo mean không kèm median/IQR.
5. Đổi recipe GA / NG+ / SCRUB giữa các seed cùng config.
6. Nói "RALM xấp xỉ retrain" — sai framing [FIXED §0.4 anti-claim].