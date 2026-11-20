"""Stopping rules (§8): mỗi luật chọn k̂ chỉ từ trajectory metrics (không oracle, không test).

Chi phí online = số ảnh phải forward mỗi snapshot để luật tự chạy được (§8: ralm đắt ~3× retain-only).
"""
from __future__ import annotations

import numpy as np

from . import score_ralm as S
from .config import DELTA_F_FRAC, RANDOM_K_SEEDS
from .guard import assert_no_oracle_imports

assert_no_oracle_imports(__file__)


def _nearest(traj, s):
    steps = np.array([m["step"] for m in traj])
    return int(steps[np.argmin(np.abs(steps - s))])


def kneedle_pareto(traj, k_min, delta=DELTA_F_FRAC):
    """pareto-knee (baseline D): u_k = norm(L_V), v_k = norm(max(0, L_F − L_F0 + δ)). Knee = điểm xa đường
    chéo (u + v = 1 trên đường cong quy chuẩn hoá) nhất; thất bại -> max(u+v)."""
    c = [m for m in traj if m["step"] >= k_min] or [traj[-1]]
    lv = np.array([m["L_V"] for m in c]); lf = np.array([m["L_F"] for m in c])
    LF0 = traj[0]["L_F"]
    v_raw = np.maximum(0.0, lf - LF0 + delta * LF0)
    norm = lambda a: (a - a.min()) / (a.max() - a.min()) if a.max() > a.min() else np.zeros_like(a)  # noqa: E731
    u, v = norm(lv), norm(v_raw)
    # đường (v theo u sắp theo step); khoảng cách tới đường chéo nối 2 đầu mút
    p0, p1 = np.array([u[0], v[0]]), np.array([u[-1], v[-1]])
    d = p1 - p0
    if np.linalg.norm(d) < 1e-12:
        return c[int(np.argmax(u + v))]["step"]
    q = np.stack([u, v], 1) - p0
    dist = np.abs(d[0] * q[:, 1] - d[1] * q[:, 0]) / np.linalg.norm(d)
    if dist.max() < 1e-9:
        return c[int(np.argmax(u + v))]["step"]
    return c[int(np.argmax(dist))]["step"]


def random_k(traj, k_min, seed) -> int:
    steps = [m["step"] for m in traj if m["step"] >= k_min] or [traj[-1]["step"]]
    return int(np.random.RandomState(seed).choice(steps))


def run_rules(traj, profile_fixed, k_min, n_D, n_V, n_R, delta_F=None, ema=None) -> dict:
    """Trả về {rule: {"k": step, "fwd": ảnh/snapshot}}."""
    last = traj[-1]["step"]
    out = {}
    for s in profile_fixed:
        out[f"fixed-{s}"] = {"k": _nearest(traj, min(s, last)), "fwd": 0}
    out["fixed-last"] = {"k": last, "fwd": 0}
    out["retain-only"] = {"k": S.select_min(traj, "L_R", k_min, delta_F), "fwd": n_R}
    out["val-only"] = {"k": S.select_min(traj, "L_V", k_min, delta_F), "fwd": n_V}
    out["pareto-knee"] = {"k": kneedle_pareto(traj, k_min), "fwd": n_V + 0}   # cần thêm F (nhỏ) — ghi gần đúng
    for s in RANDOM_K_SEEDS:
        out[f"random-k{s}"] = {"k": random_k(traj, k_min, s), "fwd": 0}
    sel = S.select_ralm(traj, k_min=k_min, ema_beta=ema, delta_F=delta_F)
    out["ralm-dv"] = {"k": sel["J_DV"], "fwd": n_D + n_V}
    out["ralm-d"] = {"k": sel["J_D"], "fwd": n_D}
    out["ralm-v"] = {"k": sel["J_V"], "fwd": n_R + n_V}
    sc = S.compute_scores(traj, ema_beta=ema)
    kb, info = S.select_band(sc, k_min)
    out["ralm-band"] = {"k": kb, "fwd": n_D + n_V, "band": info}
    out["ralm-dv-acc"] = {"k": S.select_ralm_acc(traj, k_min), "fwd": n_D + n_V}
    return out
