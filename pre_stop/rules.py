"""Luật dừng R0-R12 (kịch bản §8, §18.2), cài online: tại snapshot k chỉ đọc snapshot 0..k và P.

CÁCH LY (§0.4.1, §0.4.3): file này KHÔNG import eval_oracle, không đọc oracle, không đọc file test niêm phong,
không đọc checkpoint θ_pre. R14 / R15 (cần oracle) nằm trong analyze.py; R1 (rollback) nằm riêng ở analyze.py.
R13 (MU-Mis, cần gradient) chưa cài: xem CHANGES.md.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .config import GR_ABS, GR_FRAC, GV_MULT, LAMBDA_R, P_HIT, Q_ARGMIN


@dataclass
class Ctx:
    """Mọi thứ luật được phép biết ngoài snapshot: đích t_i (từ P), thang s, và giá trị tại bước 0."""
    t: np.ndarray            # đích theo từng mẫu t_i
    L_R0: float
    L_V0: float
    lam_R: float = LAMBDA_R
    p: int = P_HIT
    q: int = Q_ARGMIN

    @property
    def s(self) -> float:
        return float(np.mean(self.t))

    def admissible(self, sn: dict) -> bool:
        g_r = sn["L_R"] <= self.L_R0 + max(GR_FRAC * self.L_R0, GR_ABS)
        g_v = sn["L_V"] <= GV_MULT * self.L_V0
        return bool(g_r and g_v)


def make_target(P: dict, mode: str, L_V0: float) -> np.ndarray:
    """none: t = ℓ^pre; V: t = c_V·ℓ^pre với c_V = L_V(θ0)/L_V^pre; log: P-log (đã nhân c_log)."""
    if mode == "none":
        return np.asarray(P["loss_F_pre"], dtype=np.float64)
    if mode == "V":
        return (L_V0 / P["L_V_pre"]) * np.asarray(P["loss_F_pre"], dtype=np.float64)
    if mode == "log":
        assert P.get("loss_F_log") is not None, "thiếu P-log"
        return np.asarray(P["loss_F_log"], dtype=np.float64)
    raise ValueError(mode)


# ---------------------------------------------------------------- điểm của luật (mục 8.2)
def delta_F(sn, ctx):  return abs(sn["L_F"] - ctx.s) / ctx.s
def delta_R(sn, ctx):  return max(0.0, sn["L_R"] - ctx.L_R0) / ctx.s
def d_pre(sn, ctx):    return float(np.mean(np.abs(np.asarray(sn["loss_F"], dtype=np.float64) - ctx.t))) / ctx.s
def J7(sn, ctx):       return delta_F(sn, ctx) + ctx.lam_R * delta_R(sn, ctx)
def J8(sn, ctx):       return d_pre(sn, ctx) + ctx.lam_R * delta_R(sn, ctx)


# ---------------------------------------------------------------- khung online
class Rule:
    name = "rule"

    def __init__(self, ctx: Ctx):
        self.ctx = ctx
        self.last_adm = None
        self.n_adm = 0

    def feed(self, k: int, sn: dict):
        """Trả về k được chọn nếu đã quyết định tại snapshot này, ngược lại None."""
        adm = self.ctx.admissible(sn)
        if adm:
            self.last_adm = k
            self.n_adm += 1
        return self._on(k, sn, adm)

    def _on(self, k, sn, adm):
        raise NotImplementedError

    def finish(self):
        """Dự phòng (§8.1): snapshot chấp nhận được cuối cùng; không có thì k=0 kèm cờ no_admissible."""
        if self.last_adm is None:
            return 0, True
        return self.last_adm, False


class FirstHit(Rule):
    """k đầu tiên chấp nhận được thoả cond trong p snapshot liên tiếp; chọn k đầu của chuỗi (Phụ lục A)."""

    def __init__(self, ctx, cond, name):
        super().__init__(ctx)
        self.cond, self.name, self.run, self.start = cond, name, 0, None

    def _on(self, k, sn, adm):
        if adm and self.cond(sn, self.ctx):
            if self.run == 0:
                self.start = k
            self.run += 1
        else:
            self.run, self.start = 0, None
        return self.start if self.run == self.ctx.p else None


class ArgMin(Rule):
    """Giữ snapshot có điểm nhỏ nhất (trong các k chấp nhận được); dừng khi q snapshot không cải thiện."""

    def __init__(self, ctx, score, name):
        super().__init__(ctx)
        self.score, self.name = score, name
        self.best_k, self.best, self.since = None, np.inf, 0

    def _on(self, k, sn, adm):
        if not adm:
            return None
        j = self.score(sn, self.ctx)
        if j < self.best:
            self.best_k, self.best, self.since = k, j, 0
        else:
            self.since += 1
        return self.best_k if self.since >= self.ctx.q else None

    def finish(self):
        if self.best_k is not None:
            return self.best_k, False
        return super().finish()


def online_rules(ctx: Ctx) -> dict[str, Rule]:
    """R3, R4, R5 (không dùng P) và R6-R9, R12 (R6-R9 dùng đích trong ctx)."""
    return {
        "R3-retain-only": ArgMin(ctx, lambda sn, c: sn["L_R"], "R3-retain-only"),
        "R4-val-only": ArgMin(ctx, lambda sn, c: sn["L_V"], "R4-val-only"),
        "R5-heldout-hit": FirstHit(ctx, lambda sn, c: sn["L_F"] >= sn["L_V"], "R5-heldout-hit"),
        "R6-PRE-hit": FirstHit(ctx, lambda sn, c: sn["L_F"] >= c.s, "R6-PRE-hit"),
        "R7-PRE-J": ArgMin(ctx, J7, "R7-PRE-J"),
        "R8-PRE-sample": ArgMin(ctx, J8, "R8-PRE-sample"),
        "R9-PRE-median-hit": FirstHit(
            ctx, lambda sn, c: float(np.median(np.asarray(sn["loss_F"], dtype=np.float64) - c.t)) >= 0.0,
            "R9-PRE-median-hit"),
        "R12-acc-hit": FirstHit(ctx, lambda sn, c: sn["A_F"] <= sn["A_V"], "R12-acc-hit"),
    }


def run_online(rule: Rule, traj: list[dict]) -> dict:
    """Phát lại quỹ đạo từng snapshot; dừng ngay khi luật quyết định. Offline = cùng hàm này (test 1 §14)."""
    for k, sn in enumerate(traj):
        out = rule.feed(k, sn)
        if out is not None:
            return {"k": int(out), "decided_at": k, "triggered": True, "no_admissible": False}
    k, no_adm = rule.finish()
    return {"k": int(k), "decided_at": len(traj) - 1, "triggered": False, "no_admissible": bool(no_adm)}


# ---------------------------------------------------------------- luật offline (cần toàn quỹ đạo)
def r0_do_nothing(traj) -> dict:
    return {"k": 0, "decided_at": 0, "triggered": True, "no_admissible": False}


def r2_fixed(traj, K: int) -> dict:
    steps = [sn["step"] for sn in traj]
    k = steps.index(K) if K in steps else len(traj) - 1      # K vượt quỹ đạo -> snapshot cuối (ghi trong report)
    return {"k": int(k), "decided_at": int(k), "triggered": True, "no_admissible": False,
            "clipped": K not in steps}


def r10_pareto_knee(traj, ctx: Ctx) -> dict:
    LV0 = traj[0]["L_V"]
    LF0 = traj[0]["L_F"]
    x = np.array([max(0.0, sn["L_V"] - LV0) / LV0 for sn in traj])
    y = np.array([sn["L_F"] - LF0 for sn in traj])
    x = x / x.max() if x.max() > 0 else x
    y = y / y.max() if y.max() > 0 else y
    return _argbest(traj, ctx, y - x, "R10-pareto-knee")


def r11_rewind(traj, ctx: Ctx) -> dict:
    ref = traj[-1]["err_V_matched"]
    score = -np.array([abs((1.0 - sn["A_F"]) - ref) for sn in traj])
    return _argbest(traj, ctx, score, "R11-scrubR-rewind")


def _argbest(traj, ctx, score, name) -> dict:
    """argmax score trên các k chấp nhận được; không có -> k=0 + cờ no_admissible."""
    adm = np.array([ctx.admissible(sn) for sn in traj])
    if not adm.any():
        return {"k": 0, "decided_at": len(traj) - 1, "triggered": False, "no_admissible": True}
    idx = np.where(adm)[0]
    return {"k": int(idx[np.argmax(score[idx])]), "decided_at": len(traj) - 1, "triggered": True,
            "no_admissible": False}


# ---------------------------------------------------------------- nạp quỹ đạo (không đụng oracle / test)
def load_traj(paths, rid: str) -> list[dict]:
    rows = [json.loads(l) for l in paths.traj_jsonl(rid).read_text(encoding="utf-8").splitlines() if l.strip()]
    z = np.load(paths.traj_arrays(rid))
    n = len(rows)
    assert len(z["steps"]) == n, f"{rid}: số snapshot jsonl {n} ≠ mảng {len(z['steps'])}"
    f_y, v_y = z["f_y"], z["v_y"]
    hist = np.bincount(f_y, minlength=int(max(f_y.max(), v_y.max())) + 1).astype(np.float64)
    w = hist / hist.sum()
    for i, r in enumerate(rows):
        assert r["step"] == int(z["steps"][i])
        r["loss_F"] = z["loss_F"][i].astype(np.float64)
        pred = z["logits_V"][i].astype(np.float32).argmax(1)
        err_c = np.array([np.mean(pred[v_y == c] != c) if (v_y == c).any() else 0.0 for c in range(len(w))])
        r["err_V_matched"] = float((w * err_c).sum())
    return rows
