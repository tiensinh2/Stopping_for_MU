"""Má»i háº±ng sá»‘ [FIXED] vÃ  profile cháº¡y náº±m á»Ÿ Ä‘Ã¢y (ká»‹ch báº£n Â§2.1: khÃ´ng hard-code ráº£i rÃ¡c)."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

# ----------------------------------------------------------------------------- [FIXED]
EPS = 1e-12                 # Â§0.2
LAMBDA_D = 1.0              # Â§0.2
LAMBDA_V = 1.0              # Â§0.2
DELTA_F_FRAC = 0.05         # Â§8 warm-up gate: L_F(k) >= L_F0 + 0.05 * L_F0
SPLIT_SEED = 12345          # Â§3.1 chia V + random manifest
DSUB_SEED = 777             # subset-D 10k cá»‘ Ä‘á»‹nh cho T5 (Â§0A.6.3)
BAND_EPS_D = 0.05           # Â§0A.7
BAND_EPS_V = 0.05
IDENTITY_TOL = 1e-5         # Â§6: L_D = (1-rho) L_R + rho L_F
FORGET_TOL = 1e-3           # L_F táº¡i step 0 pháº£i khá»›p manifest (ná»›i tá»« 1e-4 vÃ¬ AMP-eval khÃ¡c láº§n cháº¡y)
RANDOM_K_SEEDS = (0, 1, 2, 3, 4)

# ----------------------------------------------------------------------------- hyper-params phÆ°Æ¡ng phÃ¡p
# KhÃ´ng tune theo káº¿t quáº£ (ká»‹ch báº£n Â§18.1). Ghi Ä‘Ã¨ báº±ng CLI --lr chá»‰ cho smoke/lr-scan.
METHOD_HP: dict[str, dict] = {
    # Â§5.1 GA: SGD khÃ´ng momentum, lr 1e-3
    "ga": dict(lr=1e-3, momentum=0.0, wd=0.0),
    # Â§5.2 NG+: -CE(F) + alpha*CE(R)
    "ngp": dict(lr=1e-3, momentum=0.0, wd=0.0, alpha=1.0),
    # SCRUB (Kurmanji et al. 2023): max-step trÃªn F (-KL), min-step trÃªn R (alpha*KL + gamma*CE), T=4
    "scrub": dict(lr=5e-4, momentum=0.9, wd=5e-4, alpha=0.5, gamma=1.0, temp=4.0, max_iters=300),
    # SalUn (Fan et al. 2024): random-label trÃªn F + CE(R), gradient bá»‹ mask theo weight-saliency
    "salun": dict(lr=5e-3, momentum=0.9, wd=0.0, mask_ratio=0.5),
    # RL = SalUn khÃ´ng mask (Ä‘á»‘i chá»©ng cá»§a SalUn)
    "rl": dict(lr=5e-3, momentum=0.9, wd=5e-4),
    # Tarun et al. 2023 (UNSIR): há»c noise lÃ m tÄƒng loss -> impair -> repair
    "tarun": dict(lr=2e-2, repair_lr=1e-2, momentum=0.9, wd=5e-4, impair_steps=100,
                  noise_per_class=32, noise_iters=50, noise_lr=0.1, noise_lam=1.0),
    # Fine-tune trÃªn R (Ä‘á»‘i chá»©ng)
    "ft": dict(lr=5e-3, momentum=0.9, wd=5e-4),
}
METHODS = tuple(METHOD_HP)


@dataclass(frozen=True)
class Profile:
    name: str
    dataset: str = "cifar10"
    data_root: str = "./data_raw"
    num_classes: int = 10
    # --- huáº¥n luyá»‡n base / retrain: recipe Â§3.1
    epochs: int = 100
    warmup: int = 5
    batch: int = 128
    lr: float = 0.1
    wd: float = 5e-4
    momentum: float = 0.9
    width: int = 64                     # 64 = ResNet-18 chuáº©n
    min_test_acc: float | None = 0.93   # Â§3.1: khÃ´ng Ä‘áº¡t thÃ¬ dá»«ng
    # --- split
    v_size: int = 5000
    dsub_n: int = 10000
    fake_train: int = 0                 # >0: dá»¯ liá»‡u giáº£ (chá»‰ Ä‘á»ƒ test CPU)
    fake_test: int = 0
    # --- Ä‘o / trajectory
    eval_batch: int = 1024
    eval_interval: int = 25
    n_steps: int = 1000
    k_min: int = 100
    fixed_steps: tuple = (50, 100, 200, 500, 1000, 2000)
    amp: bool = True
    # --- unlearning
    unlearn_batch: int = 128
    unlearn_augment: bool = True
    hp_overrides: dict = field(default_factory=dict)   # {method: {hp: value}}

    def hp(self, method: str, **cli) -> dict:
        hp = dict(METHOD_HP[method])
        hp.update(self.hp_overrides.get(method, {}))
        hp.update({k: v for k, v in cli.items() if v is not None})
        return hp


PROFILES: dict[str, Profile] = {
    # Kaggle 2xT4: Ä‘Ãºng recipe Â§3.1, trajectory Track A (25 bÆ°á»›c, 1000 bÆ°á»›c)
    "kaggle": Profile(name="kaggle"),
    # ÄÃºng ká»‹ch báº£n stage I: snapshot 10 bÆ°á»›c, 2000 bÆ°á»›c (tá»‘n ~2.5x eval)
    "paper": Profile(name="paper", eval_interval=10, n_steps=2000),
    "kaggle_c100": Profile(name="kaggle_c100", dataset="cifar100", num_classes=100, min_test_acc=0.70),
    # Chá»‰ Ä‘á»ƒ kiá»ƒm thá»­ logic trÃªn CPU (dá»¯ liá»‡u giáº£, ResNet width 8)
    "fake": Profile(
        name="fake", epochs=25, warmup=1, batch=64, width=8, lr=0.05, min_test_acc=None,
        v_size=150, dsub_n=300, fake_train=1500, fake_test=300, eval_batch=512,
        eval_interval=5, n_steps=40, k_min=10, fixed_steps=(10, 20, 40), amp=False,
        hp_overrides={"tarun": dict(impair_steps=15, noise_iters=5, noise_per_class=8),
                      "scrub": dict(max_iters=15)},
    ),
}


def get_profile(name: str, **over) -> Profile:
    p = PROFILES[name]
    return replace(p, **{k: v for k, v in over.items() if v is not None}) if over else p


class Paths:
    """Cáº¥u trÃºc thÆ° má»¥c Â§2."""

    def __init__(self, root: str | Path, profile: Profile):
        self.root = Path(root)
        self.ds = profile.dataset

    # -- base
    def base_ckpt(self, s): return self.root / "ckpt" / "base" / f"{self.ds}_seed{s}.pth"
    def base_stats(self, s): return self.root / "data" / f"base_stats_{self.ds}_seed{s}.json"
    def d_logits0(self, s): return self.root / "data" / f"d_logits0_{self.ds}_seed{s}.npy"

    # -- manifest
    def manifest(self, fkey, basis_seed=None):
        suf = "" if basis_seed is None else f"_seed{basis_seed}"
        return self.root / "data" / f"forget_manifest_{self.ds}_{fkey}{suf}.json"

    # -- trajectory
    def run_id(self, method, fkey, seed): return f"{method}_{self.ds}_{fkey}_seed{seed}"
    def traj_jsonl(self, rid): return self.root / "logs" / f"traj_{rid}.jsonl"
    def test_jsonl(self, rid): return self.root / "logs" / f"traj_{rid}_test_metrics.jsonl"
    def meta(self, rid): return self.root / "logs" / f"traj_{rid}_meta.json"
    def snap_dir(self, rid): return self.root / "ckpt" / "traj" / rid

    # -- retrain
    def retrain_stem(self, fkey, bseed, rseed):
        return self.root / "ckpt" / "retrain" / self.ds / f"{fkey}_base{bseed}_rseed{rseed}"

    # -- results
    @property
    def results(self): return self.root / "results"
    @property
    def figs(self): return self.root / "results" / "figs"
