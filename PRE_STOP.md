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
| `scrub` | meghdadk/SCRUB notebook + `loops.py` | 10 | epoch ≤ 3: maximize (−KL) trên F (batch 64) rồi minimize (CE + 0.5·KL, T=2) trên R (batch 32); epoch > 3 chỉ minimize | **SGD mom 0.9 lr 1e-2, wd 5e-4 (đổi theo yêu cầu, không phải mặc định của notebook gốc: Adam lr 5e-4 wd 0.1)**; lr ×0.1 sau epoch 5, 8, 9 | train (student), eval (teacher) |
| `tarun` | vikram2000b/Fast-Machine-Unlearning notebook (UNSIR) | **2** | epoch 1 = impair (noise đã học + 1000 mẫu R mỗi lớp giữ lại, Adam 0.02); epoch 2 = repair (chỉ R, Adam 0.01); batch 256 | Adam | train |

**Điều bạn cần biết về chuyện đếm epoch:**
1. **GA gốc chỉ duyệt F.** Với |F|=450 và batch 256 thì một epoch chỉ có 2 bước cập nhật, nên quỹ đạo GA rất thô (10 điểm, 20 bước tổng). FT có ~174 bước mỗi epoch (|R|≈44.550, batch 256). Vì vậy "epoch" của các phương pháp **không cùng thang**; không so sánh k giữa phương pháp theo epoch mà phải đọc cùng `n_updates`.
2. **Quỹ đạo chạy đến khi luật PRE quyết định, không còn cắt cứng ở 10 epoch.** Mỗi epoch, các luật PRE online (R6, R7, R8, R9 ở các chế độ đích `none`/`V`/`log`) được cho ăn snapshot theo đúng thứ tự như `run_online`; quỹ đạo dừng khi **tất cả** đã chọn xong checkpoint (`stop_reason = all_rules_decided`), hoặc khi phân kỳ (`diverged`). **Mặc định KHÔNG có trần epoch** (theo yêu cầu); `--max_epochs N` là trần tuỳ chọn, chỉ dùng khi người dùng tự đặt (`stop_reason = max_epochs`, ghi số luật chưa quyết định). **Rủi ro:** nếu một luật không bao giờ quyết định (ví dụ R7/R8 khi không có snapshot nào qua rào G_R/G_V, hoặc R6/R9 khi L_F không bao giờ chạm đích) thì quỹ đạo không tự dừng, chỉ dừng khi phân kỳ. Việc dừng chỉ dùng thông tin mà các luật được phép dùng (không oracle, không test). R7/R8 (argmin, q=10) cần ít nhất 10 snapshot sau điểm tối ưu nên quỹ đạo thường dài hơn 10 epoch; **đây là kéo dài vượt số epoch của code gốc**, phải nêu trong báo cáo. R3, R4, R5, R12 không nằm trong điều kiện dừng (thường không kích hoạt); chúng được chấm trên cùng quỹ đạo. `--n_epochs N` ép số epoch cố định và chạy đủ (dùng cho kiểm thử). `R2-fixed-K` dùng K ∈ {1, 2, 3, 5, 10}; K vượt quỹ đạo thì cắt về epoch cuối kèm cờ `clipped`. Tarun gốc đúng 2 epoch nên luôn có 3 điểm (epoch 0, 1, 2) và các luật có patience về cơ bản không áp dụng được với nó. **Quỹ đạo của các phương pháp khác nhau có độ dài khác nhau**, nên số liệu chạy cũ (10 epoch cố định) không so sánh trực tiếp được với số liệu mới.
3. **Tarun/UNSIR gốc chỉ định nghĩa cho quên theo lớp**, nên chỉ chạy với `class-0`. Các loại F ngẫu nhiên và hard bị bỏ qua và có thông báo. `ft` vẫn chỉ chạy trên `random-450` (đối chứng).
4. **`ngplus` (NegGrad+) đã bị bỏ** theo yêu cầu: lần chạy đầu trên CIFAR-10 thật phân kỳ ở cả 4 loại F (BN eval, lr 0.01), không có quỹ đạo dùng được. `ga` là NegGrad kinh điển (gradient ascent trên F, BN train). Kết quả của lần chạy đó vẫn nằm trong `stopping-mu (1).ipynb` nhưng không đưa vào báo cáo.
5. **SCRUB:** dùng notebook gốc của `meghdadk` (không có SWA/`param_dist`; bản `deepunlearn` có thêm hai thứ này nên **không** dùng). `beta=0` theo notebook. Dữ liệu SCRUB không augmentation (như notebook gốc). Các phương pháp khác dùng augmentation (RandomCrop+Flip) như loader train gốc.
6. **SalUn:** nhánh `cifar10` của `RL.py` (F nhãn ngẫu nhiên, rồi R). Lr 0.013 lấy từ ví dụ trong README của repo, không phải mặc định `--unlearn_lr 0.01` của `arg_parser`; vì chỉ thị là "theo code gốc", nếu bạn muốn dùng 0.01 thì sửa `SalUn.__init__`.
7. **Tarun:** nhãn noise là lớp bị quên (như trong notebook: `class_num = 0` cố định, tôi dùng đúng lớp `c`; **đây là khác biệt so với notebook** vì notebook gán nhãn 0 cho mọi noise, hợp lý chỉ khi quên 1 lớp = lớp 0). Với `class-0` hai cách trùng nhau. Mẫu retain: tối đa 1000 mẫu mỗi lớp giữ lại, đúng notebook.
8. **AMP bật trên GPU** cho mọi phương pháp (code gốc dùng fp32). Đây là khác biệt về số học, không về thuật toán.
9. **Chưa kiểm chứng:** các hằng số trên tôi đọc từ code, **chưa đối chiếu với bài báo**. SCRUB trong kịch bản có `msteps`/`lr` có thể khác bài báo gốc.
## Lưới lr cho `ga` và `scrub` (EXPLORATORY, không phải kết quả chính)
- Lr mặc định (`ga` SGD 0,01 theo code gốc; `scrub` SGD 1e-2 sau khi đã thay Adam, lịch ×0,1 sau epoch 5, 8, 9) vẫn là **mặc định chính**. Lưới chỉ là bước chẩn đoán: lr được chọn cho mô hình/dữ liệu của bài báo, chưa chắc hợp θ_pre của ta.
- Cách chạy: `python -m pre_stop.run traj --lr_scales 0.1 0.01 --ftypes random-450 [--plateau_w 10] --tag lrgrid`, rồi `analyze` với cùng tham số. Mỗi hệ số `s` tạo phương pháp `ga@s` và `scrub@s` (lr = s × lr gốc; **với `scrub@s`, s ≠ 1, lr giữ KHÔNG ĐỔI, không giảm ×0,1 ở epoch 5, 8, 9** vì lịch gốc đưa lr về ~1e-3 lần sau 9 epoch và quỹ đạo kẹt dao động; `scrub` s = 1 giữ nguyên lịch gốc). `ga` và `scrub` (s = 1) được dùng lại để đặt cạnh. Chỉ `ga` và `scrub` có lưới; `salun`, `tarun`, `ft` giữ nguyên.
- **`scrub` đã đổi từ Adam sang SGD (theo yêu cầu, EXPLORATORY so với code gốc):** dùng nhánh `args.optim == "sgd"` có sẵn trong notebook SCRUB gốc (SGD momentum 0.9), lr 1e-2 (lr 1e-3 thử trước đó quá nhỏ, L_F gần như không giảm), wd 5e-4 (giá trị trong chú thích `0.1#5e-4` của notebook), lịch lr ×0.1 sau epoch 5, 8, 9; các phần khác giữ nguyên. Notebook gốc đặt `optim = "adam"`, lr 5e-4, wd 0.1, nên **khác cấu hình mặc định của code gốc**. Lý do: Adam lr 5e-4 làm L_R vọt 0,007 -> 1,30 ngay epoch 1 và không phục hồi. Không còn nhánh Adam trong code. Meta ghi `optimizer`; quỹ đạo `scrub` Adam cũ trong `OUT` bị tự chạy lại (khác optimizer). Lưới lr `scrub@s` dùng cùng SGD, lr = s × 1e-2 không đổi.

- **Notebook bật `PLATEAU_W = 10` cho cả lần chạy chính (theo yêu cầu "chạy đến vùng loss nhỏ, không cần tối ưu quá").** Lý do: `scrub` (và `ga` sau khi hỏng) có thể phẳng mà không luật PRE nào quyết định, khi đó quỹ đạo chạy mãi. Với `PLATEAU_W`, quỹ đạo dừng khi L_F, L_R, L_V đã ổn định (`stop_reason = plateau`, `n_rules_undecided > 0` cho biết luật nào chưa quyết định). Cửa sổ 10 snapshot (gồm cả epoch 0 lúc bắt đầu). **Hệ quả cần đọc:** quỹ đạo dừng ở `plateau` là quỹ đạo chưa tới điểm tối ưu, nên các luật chưa quyết định chọn theo cơ chế dự phòng (k=0 hoặc snapshot chấp nhận được cuối cùng), không phải do luật chủ động dừng. Tiêu chí này do người triển khai đề xuất, không dùng oracle/test, và không phải trần epoch. Đặt `PLATEAU_W = None` để tắt.

- Báo cáo ghi vào `results_<tag>/` (mặc định `results_lrgrid/`), không ghi đè `results/`. Quỹ đạo lưu dưới tên `ga@0.1_random-450_seed0_u0`.
- Quỹ đạo vẫn **không có trần epoch**. `--plateau_w W` (mặc định TẮT) là tiêu chí **do tôi đề xuất**, không phải trần: dừng khi tới vùng loss ổn định, không tối ưu thêm: trung bình nửa sau so với nửa trước của W epoch gần nhất đổi < `--plateau_tol` (mặc định 5%) tương đối ở cả L_F, L_R, L_V (`stop_reason = plateau`). Dùng trung bình nửa cửa sổ vì L_F dao động ±3% giữa các epoch; tiêu chí max-min 1% của bản đầu không bao giờ thoả (log Kaggle chạy hơn 800 epoch). Lý do: với lr rất nhỏ hoặc quỹ đạo phẳng, luật PRE có thể không bao giờ quyết định và quỹ đạo sẽ chạy vô hạn. Không dùng oracle/test. Nếu không bật, việc dừng chỉ do luật PRE hoặc phân kỳ.
- **Lưu ý khi đọc:** chọn lr phải dựa trên tiêu chí không oracle (L_V không vượt rào G_V, L_F tiến về P). Chọn lr theo khoảng cách tới retrain rồi báo như kết quả chính là vi phạm quy tắc. Báo median và IQR, ghi nhãn exploratory. Chỉ chạy 1 seed nên chưa có IQR qua seed.



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
- Giai đoạn 1 chạy tối đa 300 epoch với cosine lr theo 300 epoch. Dừng sớm (patience 5, theo loss train trên R1) **chỉ được tính từ epoch mà lr đã xuống dưới 10% lr gốc** (khoảng epoch 240, hàm `early_stop_start`); sau đó nạp lại trọng số của epoch tốt nhất. Lý do: lần chạy đầu dừng ở epoch 43 khi lr còn 0,096, θ_pre chỉ đạt test acc 71,6% và bị cổng 90% chặn. Số epoch thực chạy, epoch tốt nhất và lr lúc dừng ghi trong `stage1_run` của `pre_stats`. Cache θ_pre sinh bằng cách huấn luyện khác bị từ chối (`stage1_signature`).
- Báo cáo có mục 2b và `results/compare_retrain.csv`: L/A trên F, R, V, test của từng luật dừng đặt cạnh dòng RETRAIN (trung bình M=3 oracle), kèm IQR và chênh lệch có dấu. Thêm cột `g_T`, `G_KL`, `auc_gap` (MIA tổng thể, proxy yếu) và `gapavg` (GapAvg §19.2: trung bình |lệch so với retrain| của FA, RA, TA, AUC, đơn vị điểm %; `gapavg_ta_mia` chỉ lấy TA và MIA). R1-rollback không có gapavg vì θ_pre không đo RA và MIA. Các chỉ số này chỉ để báo cáo, luật dừng không dùng.
- Chưa chạy trên CIFAR-10 thật với cấu hình mới, nên chưa biết: θ_pre có đạt 90% không, cổng "F đã được học" đạt ở w_F nào, Cổng 1 (e_pre/nu) ra bao nhiêu, và thời gian chạy thật.
- Ước tính compute trong kịch bản (10-15 giờ GPU cho Phase 1) chưa đo; nên chạy từng seed trong session riêng.
- Tiền đề "loss trước khi học F ~ loss của oracle" là giả thuyết cần đo bằng Cổng 1, không phải điều đã biết.
