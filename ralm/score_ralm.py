"""RALM scoring engine (§7) + họ anchor T0–T10 (§0A.4, §0A.8) + band H4 (§0A.7).

Chỉ đọc: traj jsonl (không có cột test) và các mảng per-sample đã lưu (dperloss, logits V/subset-D).
KHÔNG import retrain / eval_oracle (§1.1) và KHÔNG đọc `tlogits`, `L_T`, `A_T` (§1.2).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .config import BAND_EPS_D, BAND_EPS_V, DELTA_F_FRAC, EPS, LAMBDA_D, LAMBDA_V
from .guard import assert_no_oracle_imports

assert_no_oracle_imports(__file__)          # §1.1
log = logging.getLogger("ralm.score")

TEST_COLS = ("L_T", "A_T")
_ARRAY_KEYS_ALLOWED = ("steps", "dperloss", "vlogits", "dsublogits", "flogits", "d_y", "v_y", "dsub_y", "f_y",
                       "d_idx", "v_idx", "f_idx", "dsub_idx")   # không có tlogits / t_y


# ----------------------------------------------------------------------------- I/O (có guard test)
def load_traj(path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("post_selection"):
            raise AssertionError("score_ralm từ chối đọc dòng post_selection=true (§1.2)")
        for c in TEST_COLS:
            if r.get(c) is not None:
                raise AssertionError(f"score_ralm từ chối đọc cột test {c} (§1.2)")
            r.pop(c, None)
        rows.append(r)
    if not rows or rows[0]["step"] != 0:
        raise ValueError("trajectory phải bắt đầu bằng dòng step=0 (§6)")
    return rows


def load_arrays(npz_path) -> dict:
    """Chỉ nạp khoá được phép (không nạp tlogits)."""
    z = np.load(npz_path)
    return {k: z[k] for k in _ARRAY_KEYS_ALLOWED if k in z.files}


def apply_ema(traj: list[dict], key: str, beta: float) -> list[dict]:
    """EMA theo thứ tự step; ema_0 = x_0. Trả về bản sao."""
    out = [dict(m) for m in traj]
    prev = None
    for m in out:
        prev = m[key] if prev is None else beta * prev + (1 - beta) * m[key]
        m[key] = prev
    return out


def base_from_traj(traj: list[dict]) -> dict:
    t0 = traj[0]
    return {f"{k}0": t0[k] for k in ("L_R", "L_D", "L_V", "L_F", "A_R", "A_D", "A_V", "A_F")}


# ----------------------------------------------------------------------------- §7
def compute_scores(traj, base=None, ema_beta=None, lamD=LAMBDA_D, lamV=LAMBDA_V) -> list[dict]:
    """Thêm S_R, S_D, S_V, J_DV, J_D, J_V vào bản sao của trajectory."""
    base = base or base_from_traj(traj)
    if ema_beta:
        for key in ("L_R", "L_D", "L_V"):
            traj = apply_ema(traj, key, ema_beta)
    else:
        traj = [dict(m) for m in traj]
    LR0, LD0, LV0 = base["L_R0"], base["L_D0"], base["L_V0"]
    for m in traj:
        m["S_R"] = m["L_R"] / (LR0 + EPS)
        m["S_D"] = abs(m["L_D"] - LD0) / (LD0 + EPS)
        m["S_V"] = abs(m["L_V"] - LV0) / (LV0 + EPS)
        m["J_DV"] = m["S_R"] + lamD * m["S_D"] + lamV * m["S_V"]
        m["J_D"] = m["S_R"] + lamD * m["S_D"]
        m["J_V"] = m["S_R"] + lamV * m["S_V"]
    return traj


def candidates(traj, k_min, delta_F=None, base=None):
    """k ≥ k_min và (tuỳ chọn) cổng δ_F (§8). Nếu không ai qua cổng: dùng điểm có L_F lớn nhất, ghi cảnh báo."""
    cand = [m for m in traj if m["step"] >= k_min]
    if not cand:
        cand = [traj[-1]]
        log.warning("không có snapshot nào ≥ k_min; dùng snapshot cuối")
    if delta_F:
        LF0 = (base or base_from_traj(traj))["L_F0"]
        gated = [m for m in cand if m["L_F"] >= LF0 + delta_F * LF0]
        if gated:
            return gated, False
        log.warning("δ_F gate: không checkpoint nào vượt cổng -> chọn L_F lớn nhất (§8)")
        return [max(cand, key=lambda m: m["L_F"])], True
    return cand, False


def select_min(traj, key, k_min=100, delta_F=None, base=None) -> int:
    cand, _ = candidates(traj, k_min, delta_F, base)
    return min(cand, key=lambda m: (m[key], m["step"]))["step"]      # hoà -> step nhỏ hơn


def select_ralm(traj, k_min=100, ema_beta=None, lamD=LAMBDA_D, lamV=LAMBDA_V, delta_F=None) -> dict:
    sc = compute_scores(traj, ema_beta=ema_beta, lamD=lamD, lamV=lamV)
    return {j: select_min(sc, j, k_min, delta_F, base_from_traj(traj)) for j in ("J_DV", "J_D", "J_V")}


# ----------------------------------------------------------------------------- band H4 (§0A.7)
def band_mask(sc, eps_D=BAND_EPS_D, eps_V=BAND_EPS_V, k_min=0) -> np.ndarray:
    return np.array([(m["S_D"] <= eps_D) and (m["S_V"] <= eps_V) and (m["step"] >= k_min) for m in sc])


def longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for b in mask:
        cur = cur + 1 if b else 0
        best = max(best, cur)
    return best


def select_band(sc, k_min=100, eps_D=BAND_EPS_D, eps_V=BAND_EPS_V) -> tuple[int, dict]:
    """k̂_band = argmin_{k∈B(ε), k≥k_min} S_R(k). Nếu B rỗng: lấy k ít vi phạm nhất (max(S_D/ε_D, S_V/ε_V))."""
    mask = band_mask(sc, eps_D, eps_V, k_min)
    info = {"band_size": int(mask.sum()), "path_width": longest_run(band_mask(sc, eps_D, eps_V, 0)),
            "band_empty": not mask.any()}
    if mask.any():
        idx = [i for i in range(len(sc)) if mask[i]]
        k = min(idx, key=lambda i: (sc[i]["S_R"], sc[i]["step"]))
    else:
        cand = [i for i in range(len(sc)) if sc[i]["step"] >= k_min] or [len(sc) - 1]
        k = min(cand, key=lambda i: max(sc[i]["S_D"] / eps_D, sc[i]["S_V"] / eps_V))
    return sc[k]["step"], info


# ----------------------------------------------------------------------------- A3: bản accuracy
def select_ralm_acc(traj, k_min=100, lamD=LAMBDA_D, lamV=LAMBDA_V) -> int:
    """A3: thay L bằng A: S_R = 1−A_R (sai số retain), S_D = |A_D−A_D0|, S_V = |A_V−A_V0| (tuyệt đối)."""
    b = traj[0]
    sc = [dict(m, J=(1 - m["A_R"]) + lamD * abs(m["A_D"] - b["A_D"]) + lamV * abs(m["A_V"] - b["A_V"]))
          for m in traj]
    return select_min(sc, "J", k_min)


# ----------------------------------------------------------------------------- họ anchor T0–T10
def _ratio(series: np.ndarray) -> np.ndarray:
    """A(k) = |m(k) − m0| / (m0 + ε)."""
    return np.abs(series - series[0]) / (np.abs(series[0]) + EPS)


def _ce_from_logits(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64)
    zm = z - z.max(-1, keepdims=True)
    lse = np.log(np.exp(zm).sum(-1))
    return lse - zm[np.arange(len(y)), y] if z.ndim == 2 else None


def _kl_series(logits_seq: np.ndarray) -> np.ndarray:
    """KL(p_θk ‖ p_θ0) trung bình theo mẫu, theo snapshot. logits_seq: (K, n, C). KL(θ0,θ0)=0 (tự chuẩn hoá)."""
    z = logits_seq.astype(np.float64)
    lp = z - z.max(-1, keepdims=True)
    lp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
    p = np.exp(lp)
    return (p * (lp - lp[0:1])).sum(-1).mean(-1)


def _conf_series(logits_seq: np.ndarray) -> np.ndarray:
    z = logits_seq.astype(np.float64)
    p = np.exp(z - z.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return p.max(-1).mean(-1)


def anchor_series(name: str, traj: list[dict], arr: dict) -> np.ndarray:
    """A_i(k) cho từng snapshot (cùng thứ tự traj). Tên:
    T0, T1, T2_q (q∈{0.5,1,5}), T3_P (P∈{50,90,99}), T4, T5, T6_D, T6_V, T7_p (p∈{2,4}), T8, T9, T10, T0V.
    """
    L = np.asarray(arr["dperloss"], dtype=np.float64)          # (K, |D|)
    base, _, p = name.partition("_")
    if base == "T0":
        return _ratio(np.array([m["L_D"] for m in traj]))
    if base == "T0V":
        return _ratio(np.array([m["L_V"] for m in traj]))
    if base == "T1":
        return _ratio(L.var(1))
    if base == "T2":
        q = float(p) / 100.0
        n = max(1, int(np.ceil(q * L.shape[1])))
        top = np.partition(L, L.shape[1] - n, axis=1)[:, -n:]
        return _ratio(top.mean(1))
    if base == "T3":
        return _ratio(np.percentile(L, float(p), axis=1))
    if base == "T4":
        return _kl_series(arr["vlogits"])
    if base == "T5":
        return _kl_series(arr["dsublogits"])
    if base == "T6":
        return _ratio(_conf_series(arr["dsublogits" if p == "D" else "vlogits"]))
    if base == "T7":
        pp = float(p)
        return _ratio((L ** pp).mean(1) ** (1.0 / pp))
    if base == "T8":
        return np.array([m["w_dist"] for m in traj])
    if base == "T9":
        return np.array([m["emb_drift_V"] for m in traj])
    if base == "T10":
        y = np.asarray(arr["d_y"])
        classes = np.unique(y)
        mc = np.stack([L[:, y == c].mean(1) for c in classes], 1)      # (K, C)
        rel = (mc - mc[0:1]) / (mc[0:1] + EPS)
        return np.sqrt((rel ** 2).mean(1))
    raise KeyError(name)


ANCHOR_NAMES = ("T0", "T1", "T2_0.5", "T2_1", "T2_5", "T3_50", "T3_90", "T3_99", "T4", "T5",
                "T6_D", "T6_V", "T7_2", "T7_4", "T8", "T9", "T10")


def select_with_anchor(traj, anchor: np.ndarray, mode: str = "replace_D", k_min=100, lam=1.0,
                       lamV=LAMBDA_V, delta_F=None) -> int:
    """§0A.6.1. mode='replace_D': J = S_R + λ·A + λ_V·S_V ; mode='only': J = S_R + λ·A."""
    sc = compute_scores(traj)
    for m, a in zip(sc, anchor):
        m["J_A"] = m["S_R"] + lam * a + (lamV * m["S_V"] if mode == "replace_D" else 0.0)
    return select_min(sc, "J_A", k_min, delta_F, base_from_traj(traj))


def redundancy(anchor: np.ndarray, S_R: np.ndarray) -> float:
    """corr(A_i(k), S_R(k)) (§0A.5.1)."""
    if np.std(anchor) < 1e-15 or np.std(S_R) < 1e-15:
        return float("nan")
    return float(np.corrcoef(anchor, S_R)[0, 1])
