"""Oracle evaluation (§10) — NƠI DUY NHẤT được đọc θ_R. Toàn bộ chạy offline sau khi đã chọn k̂.

G_logit_F(k) = mean_{x∈F} KL(p_θR ‖ p_θk). Với nhiều retrain seed: trung vị G qua seed (§10.1).
k* = argmin_k G_logit_F;  Regret(k̂) = G(k̂) − G(k*).
"""
from __future__ import annotations

import glob
import json

import numpy as np

from .config import Paths
from .eval_core import kl_per_sample
import torch


def load_retrains(paths: Paths, fkey: str, base_seed: int):
    stem = paths.retrain_stem(fkey, base_seed, 0)
    files = sorted(glob.glob(str(stem.parent / f"{fkey}_base{base_seed}_rseed*.npz")))
    out = []
    for f in files:
        z = np.load(f)
        out.append({"train_logits": z["train_logits"].astype(np.float32),
                    "test_logits": z["test_logits"].astype(np.float32),
                    "meta": json.loads(open(f.replace(".npz", ".json"), encoding="utf-8").read())})
    return out


def _kl_traj(p_logits: np.ndarray, seq: np.ndarray) -> np.ndarray:
    """KL(p_R ‖ p_k) trung bình theo mẫu cho từng snapshot k. p_logits (n,C), seq (K,n,C)."""
    p = torch.from_numpy(p_logits.astype(np.float32))
    return np.array([float(kl_per_sample(p, torch.from_numpy(s.astype(np.float32))).mean()) for s in seq])


def oracle_curves(arr: dict, tlogits: np.ndarray, traj: list[dict], retrains: list[dict]) -> dict:
    """Trả về các đường cong theo snapshot (trung vị qua retrain seed)."""
    f_idx, v_idx = arr["f_idx"], arr["v_idx"]
    K = len(traj)
    G_F, G_V, G_FL, G_RL, G_TL, G_TA = [], [], [], [], [], []
    yT = arr["t_y"]
    for rt in retrains:
        zl = rt["train_logits"]
        G_F.append(_kl_traj(zl[f_idx], arr["flogits"]))
        G_V.append(_kl_traj(zl[v_idx], arr["vlogits"]))
        meta = rt["meta"]
        G_FL.append(np.array([abs(m["L_F"] - meta["L_F"]) for m in traj]))
        G_RL.append(np.array([abs(m["L_R"] - meta["L_R"]) for m in traj]))
        # test (post-selection, chỉ ở đây)
        tl_k = arr["tlogits"].astype(np.float32)
        acc_k = (tl_k.argmax(-1) == yT[None]).mean(1)
        G_TA.append(np.abs(acc_k - meta["A_T"]))
        t = torch.from_numpy(tl_k.reshape(-1, tl_k.shape[-1]))
        ce = torch.nn.functional.cross_entropy(t, torch.from_numpy(np.tile(yT, K)), reduction="none").view(K, -1).mean(1).numpy()
        G_TL.append(np.abs(ce - meta["L_T"]))
    med = lambda L: np.median(np.stack(L), 0)  # noqa: E731
    return {"G_logit_F": med(G_F), "G_logit_V": med(G_V), "G_F": med(G_FL), "G_R": med(G_RL),
            "G_test_loss": med(G_TL), "G_test_acc": med(G_TA),
            "A_T_k": (arr["tlogits"].astype(np.float32).argmax(-1) == yT[None]).mean(1),
            "A_T_retrain": float(np.median([r["meta"]["A_T"] for r in retrains])),
            "L_D_gap": float(np.median([r["meta"]["L_D_minus_L_D0"] for r in retrains]))}


def regret(curve: np.ndarray, steps: np.ndarray, k_hat: int) -> dict:
    i_star = int(np.argmin(curve))
    i_hat = int(np.argmin(np.abs(steps - k_hat)))
    return {"k_star": int(steps[i_star]), "G_star": float(curve[i_star]), "G_hat": float(curve[i_hat]),
            "regret": float(curve[i_hat] - curve[i_star]),
            "regret_rel": float((curve[i_hat] - curve[i_star]) / (curve.max() - curve.min() + 1e-12)),
            "E_step": int(abs(steps[i_hat] - steps[i_star])),
            "E_rank": abs(i_hat - i_star) / len(steps)}
