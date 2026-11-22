# PRE_STOP.md - ghi chú cài đặt kịch bản PRE-stop

Package `pre_stop/` cài kịch bản mới (hai giai đoạn, đích = loss trước khi học F). Package `ralm/` và kết quả 3 seed cũ giữ nguyên, không bị sửa.
Mục dưới đây là những chỗ **tôi tự quyết hoặc chưa làm**; đọc trước khi chạy.

## Cách chạy
```
python -m pre_stop.run pre   --profile kaggle --out ./pre_out --seeds 0 1 2       # giai đoạn 1
python -m pre_stop.run cfg   ... --ftypes random-450 random-1350 class-0 hard-450  # giai đoạn 2, P, oracle
python -m pre_stop.run gate1 ...                                                  # Cổng 1 (chỉ cảnh báo)
python -m pre_stop.run traj  ... --methods ga scrub salun tarun ft          # quỹ đạo unlearning
python -m pre_stop.run analyze ...
```
Notebook: `pre_stop_kaggle.ipynb`. Kiểm thử: `pytest tests/test_pre_stop.py` (17 test, CPU, dữ liệu giả).
Không còn bước `lrscan`: lr, optimizer, batch lấy đúng giá trị của code gốc nên không có siêu tham số để quét.

## Unlearning: đếm theo EPOCH, đúng code gốc (chỉ thị mới nhất)
Mỗi phương pháp được cài lại từ code gốc (clone vào `_orig/`, không commit). "Epoch" là một vòng `for epoch` của chính code gốc đó;
snapshot lấy ở cuối mỗi epoch (và epoch 0 = θ0). Số bước tối ưu mỗi epoch khác nhau giữa các phương pháp; cột `n_updates` ghi số bước thật.

| Phương pháp | Nguồn | Epoch | Mỗi epoch duyệt | Optimizer | BN |
|---|---|---|---|---|---|
| `ga` | OPTML-Group/Unlearn-Saliency `GA.py` | 10 | chỉ F, loss = -CE | SGD lr 0.01, mom 0.9, wd 5e-4, batch 256 | train |
| `ft` | Unlearn-Saliency `FT.py` | 10 | chỉ R | SGD lr 0.01, mom 0.9, wd 5e-4, batch 256 | train |
| `salun` | Unlearn-Saliency `generate_mask.py` + `RL.py` | 10 | F nhãn ngẫu nhiên rồi R, grad nhân mask (top 50% \|grad(−CE_F)\|), tham số ngoài mask được trả về giá trị đầu epoch | SGD lr 0.013, mom 0.9, wd 5e-4, batch 256 | train |
| `scrub` | meghdadk/SCRUB notebook + `loops.py` | 10 | epoch ≤ 3: maximize (−KL) trên F (batch 64) rồi minimize (CE + 0.5·KL, T=2) trên R (batch 32); epoch > 3 chỉ minimize | Adam lr 5e-4, wd 0.1; lr ×0.1 sau epoch 5, 8, 9 | train (student), eval (teacher) |
| `tarun` | vikram2000b/Fast-Machine-Unlearning notebook (UNSIR) | **2** | epoch 1 = impair (noise đã học + 1000 mẫu R mỗi lớp giữ lại, Adam 0.02); epoch 2 = repair (chỉ R, Adam 0.01); batch 256 | Adam | train |

**Điều bạn cần biết về chuyện đếm epoch:**
1. **GA gốc chỉ duyệt F.** Với |F|=450 và batch 256 thì một epoch chỉ có 2 bước cập nhật, nên quỹ đạo GA rất thô (10 điểm, 20 bước tổng). FT có ~174 bước mỗi epoch (|R|≈44.550, batch 256). Vì vậy "epoch" của các phương pháp **không cùng thang**; không so sánh k giữa phương pháp theo epoch mà phải đọc cùng `n_updates`.
2. **Quỹ đạo ngắn: 11 điểm (epoch 0..10) thay cho 201.** Luật có kiên nhẫn p=3 (first-hit) và q=10 (argmin) vốn tính theo snapshot. Với 11 điểm, q=10 gần như không bao giờ kết thúc sớm, và `R2-fixed-K` dùng K ∈ {1, 2, 3, 5, 10} epoch. Tarun chỉ có 3 điểm (epoch 0, 1, 2) nên các luật có patience về cơ bản không áp dụng được với nó.
3. **Tarun/UNSIR gốc chỉ định nghĩa cho quên theo lớp**, nên chỉ chạy với `class-0`. Các loại F ngẫu nhiên và hard bị bỏ qua và có thông báo. `ft` vẫn chỉ chạy trên `random-450` (đối chứng).
4. **`ngplus` (NegGrad+) đã bị bỏ** theo yêu cầu: lần chạy đầu trên CIFAR-10 thật phân kỳ ở cả 4 loại F (BN eval, lr 0.01), không có quỹ đạo dùng được. `ga` là NegGrad kinh điển (gradient ascent trên F, BN train). Kết quả của lần chạy đó vẫn nằm trong `stopping-mu (1).ipynb` nhưng không đưa vào báo cáo.
5. **SCRUB:** dùng notebook gốc của `meghdadk` (không có SWA/`param_dist`; bản `deepunlearn` có thêm hai thứ này nên **không** dùng). `beta=0` theo notebook. Dữ liệu SCRUB không augmentation (như notebook gốc). Các phương pháp khác dùng augmentation (RandomCrop+Flip) như loader train gốc.
6. **SalUn:** nhánh `cifar10` của `RL.py` (F nhãn ngẫu nhiên, rồi R). Lr 0.013 lấy từ ví dụ trong README của repo, không phải mặc định `--unlearn_lr 0.01` của `arg_parser`; vì chỉ thị là "theo code gốc", nếu bạn muốn dùng 0.01 thì sửa `SalUn.__init__`.
7. **Tarun:** nhãn noise là lớp bị quên (như trong notebook: `class_num = 0` cố định, tôi dùng đúng lớp `c`; **đây là khác biệt so với notebook** vì notebook gán nhãn 0 cho mọi noise, hợp lý chỉ khi quên 1 lớp = lớp 0). Với `class-0` hai cách trùng nhau. Mẫu retain: tối đa 1000 mẫu mỗi lớp giữ lại, đúng notebook.
8. **AMP bật trên GPU** cho mọi phương pháp (code gốc dùng fp32). Đây là khác biệt về số học, không về thuật toán.
9. **Chưa kiểm chứng:** các hằng số trên tôi đọc từ code, **chưa đối chiếu với bài báo**. SCRUB trong kịch bản có `msteps`/`lr` có thể khác bài báo gốc.

## Đã cài (theo kịch bản)
- R0, R1 (rollback, tách riêng), R2-K, R3, R4, R5, R6-R9 (x 2 đích `none`, `V`), R10, R11, R12, R14, R15.
- **R7g, R8g (thêm theo yêu cầu, không có trong kịch bản):** `argmin_k J(k)` thuần trên toàn quỹ đạo, offline, không rào utility, không patience. R7/R8 vẫn là bản online theo §8.
- **Cổng 1 chỉ cảnh báo, không chặn pipeline** (kịch bản §10 yêu cầu dừng khi median e_pre/nu > 10; đã bỏ theo yêu cầu "cứ chạy rồi báo cáo"). Cổng theta_pre >= 90% vẫn chặn.
- Mỗi snapshot đo R_sub, F, V và cả test (file niêm phong).
- Rào utility G_R, G_V; dự phòng "chấp nhận được cuối cùng" rồi k=0 kèm cờ `no_admissible`.
- P đóng băng (hash + dấu thời gian) trước giai đoạn 2; test kiểm tra thứ tự thời gian và hash.
- Metric mục 9: d_hat, regret, Delta0, g_T, e_pre, e_pre,agg, e_R, G_KL, AUC proxy; median + IQR; kiểm định dấu.
- Các biến thể R6-R9 còn chạy thêm đích `log` (P-log, ablation A1) để phân tích lại.

## Chưa cài / khác kịch bản (cần bạn biết)
1. **R13 (MU-Mis) chưa cài.** Cần độ nhạy gradient theo mẫu tại mỗi snapshot; kịch bản tự ghi đây là chuyển giao ngoài thiết kế. Làm sau nếu cần.
2. **Chưa có** Phase 3 (oracle từ θ_Rscr), ablation A2-A14, CIFAR-100. `scrub` đã có ở mức phương pháp unlearning (Phase 2), nhưng chưa có luật R11 dùng `θ_Rscr` thật.
3. **Mục 8-11 và 18-19 của kịch bản mới là bản bạn đã sửa; mục 0-7 chỉ còn trong lần đọc đầu của tôi.** File kịch bản hiện tại bắt đầu từ mục 8. Tôi cài mục 0-7 theo bản đã đọc; nếu bạn đã đổi chúng, báo tôi. **Mục 6 (quét lr, 2000 bước, snapshot mỗi 10 bước) đã bị thay hoàn toàn bởi chỉ thị epoch/code gốc.**
4. **R11 (SCRUB+R):** Err_ref lấy ở epoch cuối quỹ đạo, histogram lớp của F áp lên V. Đây là cách đọc của tôi về mục 18.2, không phải code gốc SCRUB+R.
5. **R10 (Pareto-knee):** chuẩn hoá theo giá trị lớn nhất trên cả quỹ đạo, nên là luật offline (đúng như bảng 18.2 ghi).
6. **R2-K khi K lớn hơn quỹ đạo** thì cắt về epoch cuối và gắn cờ `clipped` (ví dụ `tarun` chỉ có 2 epoch).
7. **R15 cần >= 2 seed** (leave-one-seed-out); với 1 seed nó không xuất hiện trong bảng.
8. **Seed unlearn:** 1 mỗi cấu hình (`--us 0`); tăng bằng `--us 0 1 2` nếu còn compute.
9. **Dữ liệu giả và ResNet width 8 chỉ để kiểm thử logic.** Kết quả của profile `fake` không có ý nghĩa khoa học và không được đưa vào báo cáo.

## Điều tôi chưa kiểm chứng
- Giai đoạn 1 giờ chạy tối đa 300 epoch, dừng sớm khi loss train trên R1 không cải thiện 5 epoch liên tiếp (không tính 5 epoch warmup), rồi nạp lại trọng số của epoch tốt nhất; cosine lr vẫn tính theo 300 epoch. Số epoch thực chạy, epoch tốt nhất và lr lúc dừng ghi trong `stage1_run` của `pre_stats`. Cache θ_pre sinh bằng cách huấn luyện khác bị từ chối (`stage1_signature`).
- Báo cáo có mục 2b và `results/compare_retrain.csv`: L/A trên F, R, V, test của từng luật dừng đặt cạnh dòng RETRAIN (trung bình M=3 oracle), kèm IQR và chênh lệch có dấu.
- Chưa chạy trên CIFAR-10 thật với cấu hình mới, nên chưa biết: θ_pre có đạt 90% không, cổng "F đã được học" đạt ở w_F nào, Cổng 1 (e_pre/nu) ra bao nhiêu, và thời gian chạy thật.
- Ước tính compute trong kịch bản (10-15 giờ GPU cho Phase 1) chưa đo; nên chạy từng seed trong session riêng.
- Tiền đề "loss trước khi học F ~ loss của oracle" là giả thuyết cần đo bằng Cổng 1, không phải điều đã biết.
