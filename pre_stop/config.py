"""PRE-stop: mọi hằng số [FIXED] của kịch bản nằm ở đây (kịch bản §2-§8)."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from ralm.config import Profile as _BaseProfile

EPS = 1e-12
SPLIT_SEED = 12345            # §4: hoán vị cố định chia V và D
R1_FRAC = 0.7                 # §2.2: R1 = 70% của D, P = 30%
PHI = 0.1                     # §2.3: replay R1 mỗi epoch của giai đoạn 2
GATE_F = 0.2                  # §2.3: L_F(θ0) <= 0.2 * L_F^pre
WF_ORDER = (1, 5, 10)         # §2.3: thứ tự w_F thử
M_ORACLE = 3                  # §2.4
ORIG_EPOCHS = 10              # số epoch mặc định của code gốc (GA, FT, SalUn, SCRUB)
LAMBDA_R = 1.0                # §8.2 [FIXED]
P_HIT = 3                     # §8.1: kiên nhẫn cho luật first-hit
Q_ARGMIN = 10                 # §8.1: kiên nhẫn cho luật argmin
GR_FRAC, GR_ABS = 0.5, 0.02   # §8.1 rào G_R
GV_MULT = 1.10                # §8.1 rào G_V
UTIL_GATE_PTS = 1.0           # §9.3: g_T <= 1.0 điểm để chọn k*
GATE1_RATIO = 10.0            # §10: median e_pre/nu > 10 -> dừng
FTYPES = ("random-450", "random-1350", "class-0", "hard-450")
METHODS = ("ga", "scrub", "salun", "tarun", "ft")
MAIN_FTYPE = "random-450"


@dataclass(frozen=True)
class Cfg:
    name: str
    dataset: str = "cifar10"
    data_root: str = "./data_raw"
    num_classes: int = 10
    # giai đoạn 1 (§5)
    e1: int = 100
    warm1: int = 5
    lr1: float = 0.1
    min_pre_acc: float | None = 0.90
    # giai đoạn 2 và oracle (§5)
    e2: int = 20
    lr2: float = 0.01
    wd: float = 5e-4
    momentum: float = 0.9
    batch: int = 128
    width: int = 64
    amp: bool = True
    eval_batch: int = 1024
    # dữ liệu (§4)
    v_size: int = 5000
    fake_train: int = 0
    fake_test: int = 0
    f_small: int = 450
    f_large: int = 1350
    rsub_n: int = 5000
    # unlearning: đơn vị là EPOCH, theo định nghĩa của code gốc từng phương pháp (xem unlearn.py)
    n_epochs: int | None = None      # None = số epoch gốc của từng phương pháp (ORIG_EPOCHS; tarun = 2)
    fixed_k: tuple = (1, 2, 3, 5, 10)   # R2-fixed: dừng ở epoch K

    def as_profile(self) -> _BaseProfile:
        """Profile của ralm chỉ để dùng lại Data / make_model."""
        return _BaseProfile(
            name=self.name, dataset=self.dataset, data_root=self.data_root, num_classes=self.num_classes,
            epochs=self.e1, warmup=self.warm1, batch=self.batch, lr=self.lr1, wd=self.wd,
            momentum=self.momentum, width=self.width, min_test_acc=None, v_size=self.v_size,
            fake_train=self.fake_train, fake_test=self.fake_test, eval_batch=self.eval_batch, amp=self.amp)


PROFILES: dict[str, Cfg] = {
    "kaggle": Cfg(name="kaggle"),
    # chỉ để kiểm thử logic trên CPU (dữ liệu giả, ResNet width 8)
    "fake": Cfg(name="fake", e1=20, warm1=1, lr1=0.05, lr2=0.03, min_pre_acc=None, e2=12, batch=64, width=8,
                amp=False,
                eval_batch=512, v_size=150, fake_train=1500, fake_test=300, f_small=20, f_large=60,
                rsub_n=200, n_epochs=4, fixed_k=(1, 2, 4)),
}


def get_cfg(name: str, **over) -> Cfg:
    c = PROFILES[name]
    return replace(c, **{k: v for k, v in over.items() if v is not None}) if over else c


class Paths:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _p(self, *a): return self.root.joinpath(*a)
    def pre_ckpt(self, s): return self._p("ckpt", "pre", f"seed{s}.pth")
    def pre_stats(self, s): return self._p("data", f"pre_stats_seed{s}.json")
    def split(self, s): return self._p("data", f"split_seed{s}.json")
    def forget(self, s, ft): return self._p("data", f"forget_{ft}_seed{s}.json")
    def P_json(self, s, ft): return self._p("data", f"P_{ft}_seed{s}.json")
    def P_loss(self, s, ft): return self._p("data", f"P_{ft}_seed{s}_loss.npy")
    def P_log(self, s, ft): return self._p("data", f"Plog_{ft}_seed{s}.npy")
    def theta0(self, s, ft): return self._p("ckpt", "theta0", f"{ft}_seed{s}.pth")
    def stage2_stats(self, s, ft): return self._p("data", f"stage2_{ft}_seed{s}.json")
    def oracle(self, s, ft, m): return self._p("oracle", f"{ft}_seed{s}_m{m}.npz")
    def run_id(self, method, ft, s, u=0): return f"{method}_{ft}_seed{s}_u{u}"
    def traj_jsonl(self, rid): return self._p("logs", f"traj_{rid}.jsonl")
    def traj_arrays(self, rid): return self._p("logs", f"traj_{rid}_arrays.npz")
    def traj_test(self, rid): return self._p("logs", f"traj_{rid}_test.jsonl")   # niêm phong (§0.4.3)
    def meta(self, rid): return self._p("logs", f"traj_{rid}_meta.json")
    @property
    def results(self): return self._p("results")
