"""DUY NHẤT file được đọc oracle (kịch bản §9, §14). rules.py và unlearn.py không được import file này."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .config import M_ORACLE, UTIL_GATE_PTS
from .measure import measure, mean_loss_acc, test_metrics


def save_oracle(path, model, data, cfg, sets):
    """Lưu loss_F, logits_F/V, L_R/A_R (R_sub), L_V, L_T, A_T của một oracle θ_R2."""
    lgF, lsF = measure(model, data, cfg, sets.f)
    lgV, _ = measure(model, data, cfg, sets.v)
    L_R, A_R = mean_loss_acc(model, data, cfg, sets.rsub)
    L_V, A_V = mean_loss_acc(model, data, cfg, sets.v)
    L_T, A_T = test_metrics(model, data, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, loss_F=lsF.cpu().numpy().astype(np.float32),
                        logits_F=lgF.cpu().numpy().astype(np.float32),
                        logits_V=lgV.cpu().numpy().astype(np.float16),
                        L_F=float(lsF.double().mean()),
                        A_F=float((lgF.argmax(1) == data.ytr[sets.f]).double().mean()),
                        L_R=L_R, A_R=A_R, L_V=L_V, A_V=A_V, L_T=L_T, A_T=A_T)


class Oracle:
    """M oracle path-matched của một (seed, ftype). Chỉ dùng để đo."""

    def __init__(self, paths, seed: int, ftype: str):
        zs = [np.load(paths.oracle(seed, ftype, m)) for m in range(M_ORACLE)]
        self.M = len(zs)
        self.loss_F = np.stack([z["loss_F"] for z in zs]).astype(np.float64)       # (M, nF)
        self.logits_F = np.stack([z["logits_F"] for z in zs]).astype(np.float64)
        self.logits_V = np.stack([z["logits_V"] for z in zs]).astype(np.float32)
        sc = lambda k: np.array([float(z[k]) for z in zs])                          # noqa: E731
        self.L_F, self.A_F, self.L_R, self.A_R = sc("L_F"), sc("A_F"), sc("L_R"), sc("A_R")
        self.L_V, self.A_V, self.L_T, self.A_T = sc("L_V"), sc("A_V"), sc("L_T"), sc("A_T")
        self.mu = self.loss_F.mean(0)
        self.nu = noise_floor(self.loss_F)
        self.pbar = _softmax(self.logits_F).mean(0)                                  # (nF, C)

    # -- khoảng cách tới oracle
    def d_F(self, loss_F_k) -> float:
        return float(np.mean(np.abs(np.asarray(loss_F_k, dtype=np.float64) - self.mu)))

    def d_hat(self, loss_F_k) -> float:
        return self.d_F(loss_F_k) / self.nu

    def g_T(self, A_T_k: float) -> float:
        return abs(A_T_k - float(self.A_T.mean())) * 100.0

    def g_V(self, L_V_k: float) -> float:
        sd = float(np.std(self.L_V, ddof=1))
        return abs(L_V_k - float(self.L_V.mean())) / max(sd, 1e-12)

    def g_kl(self, logits_F_k) -> float:
        lp = torch.log_softmax(torch.as_tensor(np.asarray(logits_F_k, dtype=np.float32)), 1).numpy()
        pb = self.pbar
        return float(np.mean(np.sum(pb * (np.log(pb + 1e-12) - lp), 1)))

    # -- chẩn đoán tiền đề (§9.3)
    def e_pre(self, t_i) -> float:
        return float(np.mean(np.abs(np.asarray(t_i) - self.mu)) / self.nu)

    def e_pre_agg(self, t_i) -> float:
        sd = float(np.std(self.L_F, ddof=1))
        return abs(float(np.mean(t_i)) - float(self.mu.mean())) / max(sd, 1e-12)

    def e_R(self, L_R0: float) -> float:
        sd = float(np.std(self.L_R, ddof=1))
        return abs(L_R0 - float(self.L_R.mean())) / max(sd, 1e-12)


def _softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def noise_floor(loss_F: np.ndarray) -> float:
    """ν = mean_m mean_i |ℓ_i(m) − mean_{m'≠m} ℓ_i(m')| (§9.1). Từ chối khi M < 3."""
    M = loss_F.shape[0]
    if M < 3:
        raise ValueError(f"nhiễu sàn ν cần M >= 3 oracle, có M={M}")
    tot = 0.0
    for m in range(M):
        others = np.delete(loss_F, m, axis=0).mean(0)
        tot += float(np.mean(np.abs(loss_F[m] - others)))
    return tot / M


def oracle_checkpoint(d_curve, gT_curve):
    """k* = argmin d_F trong các k có g_T <= 1.0 điểm; không có k nào thoả thì argmin không ràng buộc + cờ."""
    d, g = np.asarray(d_curve), np.asarray(gT_curve)
    ok = g <= UTIL_GATE_PTS
    if ok.any():
        idx = np.where(ok)[0]
        return int(idx[np.argmin(d[idx])]), False
    return int(np.argmin(d)), True


def auc_conf(conf_pos: np.ndarray, conf_neg: np.ndarray) -> float:
    """AUC (Mann-Whitney) phân biệt F (pos) với V (neg) theo max-softmax."""
    a = np.concatenate([conf_pos, conf_neg])
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a))
    ranks[order] = np.arange(1, len(a) + 1)
    # xử lý hoà bằng rank trung bình
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    n1, n0 = len(conf_pos), len(conf_neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def conf_of(logits) -> np.ndarray:
    return _softmax(np.asarray(logits, dtype=np.float64)).max(-1)
