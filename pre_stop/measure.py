"""Đo loss/accuracy ở chế độ eval, không augmentation, full forward (kịch bản §5, §7)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ralm.eval_core import forward_all


def measure(model, data, cfg, idx: torch.Tensor):
    """Trả về (logits float32, per-sample loss float32) trên các chỉ mục idx của tập train."""
    lg = forward_all(model, data, data.xtr[idx], cfg.eval_batch, cfg.amp)
    return lg, F.cross_entropy(lg, data.ytr[idx], reduction="none")


def mean_loss_acc(model, data, cfg, idx: torch.Tensor):
    lg, ls = measure(model, data, cfg, idx)
    if not torch.isfinite(ls).all():
        raise FloatingPointError("per-sample loss không hữu hạn (model phân kỳ)")
    acc = (lg.argmax(1) == data.ytr[idx]).double().mean()
    return float(ls.double().mean()), float(acc)


def test_metrics(model, data, cfg):
    """Chỉ gọi cho file test niêm phong hoặc phần oracle (§0.4.3)."""
    lg = forward_all(model, data, data.xte, cfg.eval_batch, cfg.amp)
    ls = F.cross_entropy(lg, data.yte)
    return float(ls), float((lg.argmax(1) == data.yte).float().mean())
