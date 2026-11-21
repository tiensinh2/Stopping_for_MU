# Chạy thí nghiệm RALM trên Kaggle (GPU T4 ×2)

Code nằm trong package [`ralm/`](ralm/), test trong [`tests/test_ralm.py`](tests/test_ralm.py). Các điểm khác kịch bản ghi ở [`CHANGES.md`](CHANGES.md).

**Cách nhanh nhất:** mở [`ralm_kaggle.ipynb`](ralm_kaggle.ipynb) trên Kaggle (File → Import Notebook → GitHub URL hoặc upload), chạy từng cell theo thứ tự.
Notebook làm đúng các bước dưới đây, gồm cả cell tự tính go/no-go sau smoke test. Phần còn lại của file này là giải thích chi tiết từng bước.

**Trạng thái kiểm chứng:** logic đã được kiểm bằng 17 unit test + chạy end-to-end trên **CPU với dữ liệu giả**
(base → 7 phương pháp → retrain → aggregate → hình). **Chưa chạy trên CIFAR-10 thật hay GPU thật** (máy phát triển chỉ có PyTorch CPU),
nên mọi con số tốc độ dưới đây là ước lượng, không phải đo đạc.

## 0. Chuẩn bị notebook

1. Settings → Accelerator: **GPU T4 x2**; Internet: **On** (cần để `git clone` và tải CIFAR-10).
2. Clone repo vào `/kaggle/working` (repo **public** thì không cần token; repo **private** xem bên dưới).
3. Mỗi cell dưới đây là một lệnh `!python ...`. Mọi lệnh **idempotent**: job đã xong tự bỏ qua, nên session chết thì chạy lại cell là tiếp tục.
   Kết quả ghi vào `/kaggle/working/ralm_out` (nằm ngoài thư mục repo nên `git pull` không đụng tới). Nhớ Save Version, hoặc copy sang Dataset giữa các session vì `/kaggle/working` bị xoá khi session mới bắt đầu.

```python
%cd /kaggle/working
!git clone https://github.com/<user>/<repo>.git ralm_repo     # lần sau: !cd ralm_repo && git pull
%cd /kaggle/working/ralm_repo
!pip install -q -r requirements.txt                             # torch/torchvision đã có sẵn trên Kaggle; dòng này chỉ bổ sung phần thiếu
!python -m ralm.run info          # phải thấy 2 GPU Tesla T4
!python -m pytest tests -x -q     # gate 0: phải xanh (≈1 phút)
```

Repo **private**: tạo Personal Access Token (GitHub → Settings → Developer settings → Fine-grained token, quyền *Contents: read-only*),
thêm vào Kaggle *Add-ons → Secrets* với tên `GH_TOKEN`, rồi:

```python
from kaggle_secrets import UserSecretsClient
tok = UserSecretsClient().get_secret("GH_TOKEN")
!git clone https://{tok}@github.com/<user>/<repo>.git ralm_repo
```
Không dán token trực tiếp vào notebook (notebook có thể bị public khi chia sẻ).

Lưu ý: chạy từ thư mục `ralm_repo` (nơi chứa package `ralm/`) thì `python -m ralm.run ...` mới import được.

## 1. Smoke test có go/no-go (bắt buộc, §12 Gate 0.5)

Chạy 1 seed, random ρ=0.01, GA, 1000 bước (hoặc dùng profile mặc định `kaggle`) trước khi tốn GPU cho phần lớn:

```python
OUT="/kaggle/working/ralm_out"
!python -m ralm.run base    --profile kaggle --out $OUT --seeds 0                 # ~ vài chục phút; GATE: test acc ≥ 93%
!python -m ralm.run traj    --profile kaggle --out $OUT --seeds 0 --fkeys random_rho0.01 --methods ga ngp scrub
!python -m ralm.run retrain --profile kaggle --out $OUT --seeds 0 --fkeys random_rho0.01 --retrain_seeds 1
!python -m ralm.run aggregate --profile kaggle --out $OUT --seeds 0 --fkeys random_rho0.01 --methods ga ngp scrub --tag smoke
```

Đọc `results/rules_table_smoke.md` và `results/figs/regret_curve_*.png`. **NO-GO** (§12) nếu `ralm-dv` luôn chọn `k̂ = k_min`
hoặc `corr_J_G < 0` (trong `results/runs/*.json` → `go_nogo`). Khi đó dừng và báo cáo, đừng chạy tiếp.

> `base` dừng bằng `SystemExit` nếu test acc < 93% (§3.1). Không tự đổi recipe — báo lại.

## 2. lr-scan cho các phương pháp mới (làm TRƯỚC khi chạy chính)

Siêu tham số của **SalUn, Tarun, RL** và **SCRUB** trong [`ralm/config.py`](ralm/config.py) lấy theo giá trị thường dùng trong paper gốc,
**chưa được hiệu chỉnh trên CIFAR-10 thật**. Trong lần chạy thử CPU/dữ liệu giả, salun/rl/tarun làm mất utility rất nhanh
(L_R tăng từ ~0.3 lên ~1.5 sau 40 bước) — có thể là artefact của mô hình tí hon, nhưng cần kiểm chứng.
Theo §5.1, quét lr trên **1 seed**, chọn bằng tiêu chí **V-loss không tăng quá 20% tại step 200** (chọn bằng V, không dùng test/retrain):

```python
# run_id không chứa lr, nên mỗi (method, lr) dùng một --out riêng, copy θ0 + d_logits0 vào để khỏi train lại.
for m in ["salun", "tarun", "scrub"]:
    for lr in [1e-3, 5e-3, 2e-2]:
        d = f"/kaggle/working/lrscan/{m}_{lr}"
        !mkdir -p {d} && cp -r $OUT/ckpt $OUT/data {d}/
        !python -m ralm.run traj --profile kaggle --out {d} --seeds 0 --fkeys random_rho0.1 --methods {m} --lr {lr} --n_steps 200
# Đọc L_V ở step 200 trong {d}/logs/traj_*.jsonl và so với step 0:
# chọn lr lớn nhất có L_V(200) <= 1.2 * L_V(0).
```
Sau đó sửa **một lần** `METHOD_HP` trong `config.py`, ghi vào `CHANGES.md`, và giữ nguyên cho mọi seed (§18.5).

## 3. Chạy chính (Track A, §0B.3) — cả 3 seed trong một session

Seed 0 đo thật khoảng 70 phút (base ≈ 22 phút, retrain ≈ 21 phút, mỗi trajectory ≈ 4–4,5 phút), nên 3 seed chia 2 GPU ước tính ≈ 3 giờ
(chưa đo cho 3 seed). Nếu session ngắn, có thể chạy từng seed: `SEEDS="0"`, rồi `"1"`, `"2"` ở session sau, đặt
`RESTORE_FROM="/kaggle/input/<tên-notebook>/ralm_out"` (kiểm tra bằng `!ls /kaggle/input`). `aggregate` tự nhận các seed có trong `logs/traj_*_meta.json`.

`launch` mở 2 tiến trình, mỗi tiến trình một T4 (`CUDA_VISIBLE_DEVICES`), job chia theo `i % 2`.

```python
OUT="/kaggle/working/ralm_out"
SEEDS="0 1 2"
FK="random_rho0.1 random_rho0.01"
!python -m ralm.run launch base    --gpus 2 --profile kaggle --out $OUT --seeds $SEEDS
!python -m ralm.run launch retrain --gpus 2 --profile kaggle --out $OUT --seeds $SEEDS --fkeys $FK --retrain_seeds 1
!python -m ralm.run launch traj    --gpus 2 --profile kaggle --out $OUT --seeds $SEEDS --fkeys $FK --methods ga ngp scrub salun tarun
!python -m ralm.run aggregate --profile kaggle --out $OUT --seeds $SEEDS --fkeys $FK --methods ga ngp scrub salun tarun --tag trackA
```

Mở rộng (Track B / stage I): thêm `random_rho0.001 highloss_rho0.01 memorized_rho0.01 classwise_airplane classwise_cat` vào `FK`
(highloss/memorized phụ thuộc θ0 từng seed nên cần `base` xong trước). Thêm `rl ft` vào `--methods` nếu muốn đối chứng.
`--profile paper` = snapshot mỗi 10 bước, 2000 bước (đúng stage I, tốn ~2.5× eval). `--delta_f` bật cổng δ_F, `--ema 0.9` bật EMA.

### Thời gian (đo ở seed 0, T4, AMP)
| Việc | Số job (3 seed) | Thời gian/job đã đo |
|---|---|---|
| base θ0 | 3 | ≈ 22 phút |
| retrain θ_R | 6 | ≈ 20–22 phút |
| trajectory | 30 | ≈ 4–4,5 phút |

Tổng ≈ 3 giờ thực khi chia 2 GPU (ước tính từ số đo trên, chưa đo cả 3 seed); chưa tính tải CIFAR-10 (từng mất ≈ 25 phút). Vì idempotent nên có thể tách nhiều session.
Nếu OOM: giảm `eval_batch` trong profile (§1.6: không đổi lr/batch của recipe).

## 4. Đầu ra (`$OUT/results/`)

| File | Nội dung |
|---|---|
| `rules_table_<tag>.md` | bảng chính theo (method, forget set): median regret, IQR, median k̂, %k̂=k_min, chi phí forward online, \|ΔA_T\| so với θ_R |
| `summary_<tag>.csv` | dòng thô cho mọi (run, rule) kèm regret, E_step, G_test_acc |
| `anchors_<tag>.csv`, `tmatrix_<tag>.csv` | T0–T10 (re-analysis, 0 GPU) + tiêu chí "thắng T0" §0A.6.2, corr(A, S_R), corr(A, G_logit_F) |
| `band_<tag>.csv` | H4: coverage, path-width, phương sai G_logit_F trong/ngoài band |
| `mia_<tag>.csv` | AUC logit-gap F vs. test tại k̂ (ralm-dv / retain-only / fixed-last / retrain / θ0); 0.5 = không rò rỉ |
| `figs/` | `regret_curve_*`, `scatter_khat_vs_kstar_*`, `box_regret_by_rule_*` |
| `runs/<run_id>.json` | toàn bộ đường cong để vẽ lại; `logs/traj_*.jsonl` là nguồn số liệu gốc (§1.7) |

Gate Track A (§0B.5): `ralm-dv` thua `retain-only` ở cả hai ρ → NO-GO, viết báo cáo smoke và dừng.
Nhớ báo cáo **median + IQR**, không chỉ mean (§18.4); và framing luôn là "luật chọn k không cần oracle", không phải "ước lượng retrain" (§0.4).
