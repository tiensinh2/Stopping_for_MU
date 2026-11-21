"""Chia R1/P theo seed, các loại F, và assert tập (kịch bản §2.2, §4)."""
from __future__ import annotations

import json

import numpy as np
import torch

from .config import R1_FRAC, SPLIT_SEED, Cfg


def stage_split(data, seed: int):
    """R1 = 70% của D, P = D \\ R1; ngẫu nhiên theo seed cơ sở (§2.2)."""
    d = data.d_idx_np
    perm = np.random.RandomState(seed).permutation(len(d))
    n1 = int(round(R1_FRAC * len(d)))
    return np.sort(d[perm[:n1]]), np.sort(d[perm[n1:]])


def forget_size(cfg: Cfg, ftype: str):
    return {"random-450": cfg.f_small, "random-1350": cfg.f_large, "hard-450": cfg.f_small}.get(ftype)


def choose_forget(ftype: str, cfg: Cfg, p_np, y_np, seed: int, loss_pre_p=None) -> np.ndarray:
    """F ⊂ P. hard-450 cần loss_pre_p = ℓ^pre trên toàn P (cùng thứ tự với p_np)."""
    if ftype in ("random-450", "random-1350"):
        rs = np.random.RandomState(SPLIT_SEED + seed)
        return np.sort(rs.choice(p_np, forget_size(cfg, ftype), replace=False))
    if ftype == "class-0":
        return np.sort(p_np[y_np[p_np] == 0])
    if ftype == "hard-450":
        assert loss_pre_p is not None and len(loss_pre_p) == len(p_np)
        order = np.lexsort((p_np, -np.asarray(loss_pre_p)))     # loss giảm dần, hoà thì theo index
        return np.sort(p_np[order[:forget_size(cfg, ftype)]])
    raise ValueError(ftype)


class Sets:
    """Chỉ mục R1, P, F, R2, R, R_sub, V dạng numpy và tensor. Có đủ assert §4."""

    def __init__(self, data, cfg: Cfg, r1_np, p_np, f_np, ftype: str | None = None):
        r1, p, f = (np.sort(np.asarray(a, dtype=np.int64)) for a in (r1_np, p_np, f_np))
        v = data.v_idx_np
        assert len(np.intersect1d(f, v)) == 0, "F ∩ V ≠ ∅"
        assert np.isin(f, p).all(), "F ⊄ P"
        assert len(np.intersect1d(r1, p)) == 0, "R1 ∩ P ≠ ∅"
        assert np.array_equal(np.union1d(r1, p), data.d_idx_np), "R1 ∪ P ≠ D"
        r2 = np.setdiff1d(p, f)
        assert len(np.intersect1d(f, r2)) == 0, "F ∩ R2 ≠ ∅"
        n_exp = forget_size(cfg, ftype) if ftype else None
        if n_exp is not None:
            assert len(f) == n_exp, f"|F|={len(f)} ≠ {n_exp}"
        r = np.union1d(r1, r2)
        assert len(r) + len(f) == len(data.d_idx_np)
        rs = np.random.RandomState(SPLIT_SEED + 99)
        n = min(cfg.rsub_n, len(r))
        n1 = min(int(round(n * len(r1) / len(r))), len(r1))
        n2 = min(n - n1, len(r2))
        rsub = np.sort(np.concatenate([rs.choice(r1, n1, replace=False), rs.choice(r2, n2, replace=False)]))
        self.r1_np, self.p_np, self.f_np, self.r2_np, self.r_np = r1, p, f, r2, r
        self.rsub_np, self.v_np = rsub, v
        dev = data.device
        t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.int64)).to(dev)  # noqa: E731
        self.r1, self.p, self.f, self.r2, self.r = t(r1), t(p), t(f), t(r2), t(r)
        self.rsub, self.v = t(rsub), t(v)


def save_split(paths, seed: int, r1_np, p_np):
    f = paths.split(seed)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"seed": seed, "r1": [int(i) for i in r1_np], "p": [int(i) for i in p_np]}),
                 encoding="utf-8")


def load_split(paths, seed: int):
    j = json.loads(paths.split(seed).read_text(encoding="utf-8"))
    return np.asarray(j["r1"], dtype=np.int64), np.asarray(j["p"], dtype=np.int64)


def save_forget(paths, seed: int, ftype: str, f_np):
    f = paths.forget(seed, ftype)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"seed": seed, "ftype": ftype, "f": [int(i) for i in f_np]}), encoding="utf-8")


def load_sets(paths, data, cfg: Cfg, seed: int, ftype: str) -> Sets:
    r1, p = load_split(paths, seed)
    f = np.asarray(json.loads(paths.forget(seed, ftype).read_text(encoding="utf-8"))["f"], dtype=np.int64)
    return Sets(data, cfg, r1, p, f, ftype)
