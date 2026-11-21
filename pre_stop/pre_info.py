"""Thông tin P (kịch bản §3): đóng băng trước giai đoạn 2, kèm dấu thời gian và hash."""
from __future__ import annotations

import copy
import hashlib
import json
import time

import numpy as np
import torch
import torch.nn.functional as F


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def train_mode_loss(model, data, cfg, idx: torch.Tensor, seed: int = 0) -> float:
    """Loss ở chế độ train (BN theo batch, có augmentation) trên idx; không đổi trọng số / BN buffer gốc."""
    m = copy.deepcopy(model)
    m.train()
    g = torch.Generator(device=idx.device)
    g.manual_seed(seed)
    perm = idx[torch.randperm(len(idx), device=idx.device, generator=g)]
    tot = 0.0
    with torch.no_grad():
        for s in range(0, len(perm), cfg.batch):
            b = perm[s:s + cfg.batch]
            x, y = data.batch(b, augment=True)
            tot += float(F.cross_entropy(m(x).float(), y, reduction="sum"))
    return tot / len(perm)


def freeze_P(paths, seed: int, ftype: str, loss_F_pre: np.ndarray, L_V_pre: float, A_F_pre: float,
             c_log: float) -> dict:
    """Ghi P.json + P_loss.npy TRƯỚC khi giai đoạn 2 bắt đầu (test 5 §14 kiểm tra thứ tự thời gian)."""
    lp = paths.P_loss(seed, ftype)
    lp.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(loss_F_pre, dtype=np.float32)
    np.save(lp, arr)
    P = {"seed": seed, "ftype": ftype, "n_F": int(len(arr)), "L_F_pre": float(arr.astype(np.float64).mean()),
         "L_V_pre": float(L_V_pre), "A_F_pre": float(A_F_pre), "c_log": float(c_log),
         "loss_file": lp.name, "sha256": _sha(arr), "t_frozen": time.time(),
         "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    paths.P_json(seed, ftype).write_text(json.dumps(P, indent=2), encoding="utf-8")
    return P


def load_P(paths, seed: int, ftype: str) -> dict:
    """Đọc P, kiểm hash. Kèm loss_F_pre (np) và loss_F_log (np hoặc None)."""
    P = json.loads(paths.P_json(seed, ftype).read_text(encoding="utf-8"))
    arr = np.load(paths.P_loss(seed, ftype))
    assert _sha(arr) == P["sha256"], "P bị sửa sau khi đóng băng (hash không khớp)"
    P["loss_F_pre"] = arr.astype(np.float64)
    pl = paths.P_log(seed, ftype)
    P["loss_F_log"] = np.load(pl).astype(np.float64) if pl.exists() else None
    return P
