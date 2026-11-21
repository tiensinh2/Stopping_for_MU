"""Tiện ích nạp θ0 (không phụ thuộc eval_oracle)."""
from __future__ import annotations

import torch

from ralm.model import make_model


def load_theta0(cfg, paths, seed, ftype, device):
    m = make_model(cfg.as_profile(), device)
    m.load_state_dict(torch.load(paths.theta0(seed, ftype), map_location=device))
    m.eval()
    return m
