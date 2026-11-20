"""Đo lường lõi (§6, §10). Một lần forward trên toàn bộ tập train (= D ∪ V) rồi chỉ mục ra R/D/F/V.

Điều kiện 4 của identity (§0A.1) được thoả mặc nhiên: cùng một θ_k, cùng một lần forward cho cả 3 tập.
Đo ở model.eval(), không augmentation, no_grad. Loss = cross-entropy trung bình, tính float64 trên per-sample float32.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .config import IDENTITY_TOL


def seed_all(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def forward_all(model, data, x_u8: torch.Tensor, bs: int, amp: bool = True, want_feat: bool = False):
    """Trả về logits float32 (N,C) [và feature (N,d)] của toàn bộ x_u8."""
    was_training = model.training
    model.eval()
    dev = x_u8.device
    use_amp = amp and dev.type == "cuda"
    los, fts = [], []
    for s in range(0, x_u8.shape[0], bs):
        xb = data.prep(x_u8[s:s + bs])
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            lo, ft = model(xb, True)
        los.append(lo.float())
        if want_feat:
            fts.append(ft.float())
    model.train(was_training)
    logits = torch.cat(los)
    return (logits, torch.cat(fts)) if want_feat else logits


def kl_between(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """mean_x KL(p ‖ q), p=softmax(logits_p), q=softmax(logits_q); float32, không temperature (§10.1)."""
    return float(kl_per_sample(logits_p, logits_q).mean())


def kl_per_sample(logits_p, logits_q) -> torch.Tensor:
    lp = F.log_softmax(logits_p.float(), 1)
    lq = F.log_softmax(logits_q.float(), 1)
    return (lp.exp() * (lp - lq)).sum(1)


def eval_full(model, data, split, amp: bool = True, bs: int = 1024, want_feat: bool = False, check: bool = True):
    """Metrics §6 tại θ hiện tại. Trả về (metrics, extras).

    extras: logits (N,C) float32 trên cả train, per-sample loss (N,), feat_V (nếu want_feat).
    """
    out = forward_all(model, data, data.xtr, bs, amp, want_feat)
    logits, feat = out if want_feat else (out, None)
    y = data.ytr
    loss = F.cross_entropy(logits, y, reduction="none")
    if not torch.isfinite(loss).all():
        raise FloatingPointError("per-sample loss không hữu hạn (model phân kỳ)")
    correct = (logits.argmax(1) == y).double()
    m = lambda t, idx: float(t[idx].double().mean())  # noqa: E731
    d = {}
    for name, idx in (("R", split.r), ("D", split.d), ("F", split.f), ("V", split.v)):
        d[f"L_{name}"] = m(loss, idx)
        d[f"A_{name}"] = m(correct, idx)
    d["mean_conf_F"] = float(F.softmax(logits[split.f], 1).max(1).values.double().mean())
    if check:   # §6 / §16: bug DataSplit nếu lệch
        ident = (1 - split.rho) * d["L_R"] + split.rho * d["L_F"]
        if abs(d["L_D"] - ident) > IDENTITY_TOL:
            raise AssertionError(f"identity vi phạm: L_D={d['L_D']:.8f} vs {ident:.8f}")
    ex = {"logits": logits, "loss": loss, "feat_V": None if feat is None else feat[split.v]}
    return d, ex


def eval_test(model, data, amp: bool = True, bs: int = 1024):
    """Chỉ gọi ở cuối phase (§1.2). Trả về (L_T, A_T)."""
    logits = forward_all(model, data, data.xte, bs, amp)
    loss = F.cross_entropy(logits, data.yte)
    acc = (logits.argmax(1) == data.yte).float().mean()
    return float(loss), float(acc)


def flat_params(model) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).float().clone() for p in model.parameters()])


def weight_dist(model, ref_flat: torch.Tensor) -> float:
    """T8: ‖θ_k − θ_0‖₂ / sqrt(n_params)."""
    flat = torch.cat([p.detach().reshape(-1).float() for p in model.parameters()])
    return float((flat - ref_flat).norm() / np.sqrt(flat.numel()))


def emb_drift(feat_k: torch.Tensor, feat_0: torch.Tensor) -> float:
    """T9: mean ‖f_k − f_0‖² / mean ‖f_0‖² trên V (chuẩn hoá để so được giữa các model)."""
    return float(((feat_k - feat_0) ** 2).sum(1).mean() / ((feat_0 ** 2).sum(1).mean() + 1e-12))
