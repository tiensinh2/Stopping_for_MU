# PRE_STOP.md - ghi chú cài đặt kịch bản PRE-stop

Package `pre_stop/` cài đúng kịch bản mới (hai giai đoạn, đích = loss trước khi học F). Package `ralm/` và kết quả 3 seed cũ giữ nguyên, không bị sửa.
Mục dưới đây là những chỗ **tôi tự quyết hoặc chưa làm**; đọc trước khi chạy.

## Cách chạy
```
python -m pre_stop.run pre     --profile kaggle --out ./pre_out --seeds 0 1 2     # giai đoạn 1
python -m pre_stop.run cfg     ... --ftypes random-450 random-1350 class-0 hard-450  # giai đoạn 2, P, oracle
python -m pre_stop.run gate1   ...                                                # Cổng 1; mã thoát 2 nếu tiền đề sụp
python -m pre_stop.run lrscan  ... ; python -m pre_stop.run traj ... ; python -m pre_stop.run analyze ...
```
Notebook: `pre_stop_kaggle.ipynb`. Kiểm thử: `pytest tests/test_pre_stop.py` (13 test, CPU, dữ liệu giả).

## Đã cài (theo kịch bản)
- R0, R1 (rollback, tách riêng), R2-K, R3, R4, R5, R6-R9 (x 2 đích `none`, `V`), R10, R11, R12, R14, R15.
- **R7g, R8g (thêm theo yêu cầu, không có trong kịch bản):** `argmin_k J(k)` thuần trên toàn quỹ đạo, offline, không rào utility, không patience. R7/R8 vẫn là bản online theo §8.
- **Cổng 1 chỉ cảnh báo, không chặn pipeline** (kịch bản §10 yêu cầu dừng khi median e_pre/nu > 10; đã bỏ theo yêu cầu "cứ chạy rồi báo cáo"). Cổng theta_pre >= 90% vẫn chặn.
- Snapshot mỗi 10 bước unlearning (không theo epoch): 201 snapshot/quỹ đạo. Mỗi snapshot đo R_sub, F, V và cả test (file niêm phong).
- Rào utility G_R, G_V; dự phòng "chấp nhận được cuối cùng" rồi k=0 kèm cờ `no_admissible`.
- P đóng băng (hash + dấu thời gian) trước giai đoạn 2; test kiểm tra thứ tự thời gian và hash.
- Metric mục 9: d_hat, regret, Delta0, g_T, e_pre, e_pre,agg, e_R, G_KL, AUC proxy; median + IQR; kiểm định dấu.
- Các biến thể R6-R9 còn chạy thêm đích `log` (P-log, ablation A1) để phân tích lại.

## Chưa cài / khác kịch bản (cần bạn biết)
1. **R13 (MU-Mis) chưa cài.** Cần độ nhạy gradient theo mẫu tại mỗi snapshot; kịch bản tự ghi đây là chuyển giao ngoài thiết kế. Làm sau nếu cần.
2. **Chỉ có Phase 0 và Phase 1.** Chưa có scrub (Phase 2), ablation A2-A14, oracle từ đầu θ_Rscr (Phase 3), CIFAR-100.
3. **Mục 8-11 và 18-19 của kịch bản mới là bản bạn đã sửa; mục 0-7 chỉ còn trong lần đọc đầu của tôi.** File kịch bản hiện tại bắt đầu từ mục 8. Tôi cài mục 0-7 theo bản đã đọc; nếu bạn đã đổi chúng, báo tôi.
4. **ft chỉ chạy trên random-450** (mục 10: "thêm ft làm đối chứng trên random-450").
5. **R11 (SCRUB+R):** Err_ref lấy ở snapshot cuối quỹ đạo, histogram lớp của F áp lên V. Đây là cách đọc của tôi về mục 18.2, không phải code gốc SCRUB+R.
6. **R10 (Pareto-knee):** chuẩn hoá theo giá trị lớn nhất trên cả quỹ đạo, nên là luật offline (đúng như bảng 18.2 ghi).
7. **R2-K khi K lớn hơn quỹ đạo** thì cắt về snapshot cuối và gắn cờ `clipped`.
8. **R15 cần >= 2 seed** (leave-one-seed-out); với 1 seed nó không xuất hiện trong bảng.
9. **Seed unlearn:** 1 mỗi cấu hình (`--us 0`); tăng bằng `--us 0 1 2` nếu còn compute.
10. **BN ở chế độ train** khi unlearn (kịch bản cho phép, yêu cầu ghi rõ; đã ghi vào meta mỗi quỹ đạo). Chưa có tuỳ chọn đóng băng BN.
11. **AMP:** bật trên GPU (như `ralm/`). Cổng "eval tất định" kiểm trên CPU; trên GPU hai lượt forward vẫn giống hệt nhưng chưa có test riêng.
12. **Dữ liệu giả và ResNet width 8 chỉ để kiểm thử logic.** Kết quả của profile `fake` (ví dụ bảng regret, cổng w_F) không có ý nghĩa khoa học và không được đưa vào báo cáo.

## Điều tôi chưa kiểm chứng
- Chưa chạy trên CIFAR-10 thật, nên chưa biết: θ_pre có đạt 90% không, cổng "F đã được học" có đạt ở w_F nào, Cổng 1 (e_pre/nu) ra bao nhiêu, quét lr chọn lr nào, và thời gian chạy thật.
- Ước tính compute trong kịch bản (10-15 giờ GPU cho Phase 1) chưa đo; nên chạy từng seed trong session riêng.
- Tiền đề "loss trước khi học F ~ loss của oracle" là giả thuyết cần đo bằng Cổng 1, không phải điều đã biết.
