"""Forget-set manifest (§4). Mọi chỉ mục là chỉ mục vào mảng train gốc.

fkey (khoá cấu hình):
  random_rho{r}          basis=global   uniform không hoàn lại từ D, seed 12345
  highloss_rho{r}        basis=per-seed top-ρ CE loss tại θ0
  memorized_rho{r}       basis=per-seed top-ρ margin y_true·z_true − max_{c≠true} z_c tại θ0 (margin cao nhất)
  classwise_{name|id}    basis=global   toàn bộ mẫu D của một class (airplane=0, cat=3 ở CIFAR-10)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .config import DSUB_SEED, FORGET_TOL, SPLIT_SEED, Paths, Profile
from .data import CIFAR10_CLASSES, Data

PER_SEED = ("highloss", "memorized")


def parse_fkey(fkey: str):
    m = re.fullmatch(r"(random|highloss|memorized)_rho([0-9.]+)", fkey)
    if m:
        return m.group(1), float(m.group(2)), None
    m = re.fullmatch(r"classwise_(\w+)", fkey)
    if m:
        name = m.group(1)
        cid = int(name) if name.isdigit() else CIFAR10_CLASSES.index(name)
        return "classwise", None, cid
    raise ValueError(f"fkey không hợp lệ: {fkey}")


def dsub_indices(data: Data, n: int) -> np.ndarray:
    """Subset-D cố định (§0A.6.3, chỉ mục cố định theo manifest): không phụ thuộc seed/forget set."""
    n = min(n, len(data.d_idx_np))
    rs = np.random.RandomState(DSUB_SEED)
    return np.sort(rs.choice(data.d_idx_np, n, replace=False))


def build_manifest(profile: Profile, paths: Paths, data: Data, fkey: str, base_seed: int) -> dict:
    ftype, rho, cid = parse_fkey(fkey)
    d = data.d_idx_np
    basis = "per-seed" if ftype in PER_SEED else "global"
    path = paths.manifest(fkey, base_seed if basis == "per-seed" else None)
    if path.exists():
        return load_manifest(path, data)
    if ftype == "random":
        n = int(round(rho * len(d)))
        idx = np.sort(np.random.RandomState(SPLIT_SEED).choice(d, n, replace=False))
        src = None
    elif ftype == "classwise":
        idx = d[data.ytr_np[d] == cid]
        src = None
    else:
        lg = np.load(paths.d_logits0(base_seed))                 # (N, C) trên toàn bộ train
        y = data.ytr_np
        n = int(round(rho * len(d)))
        zl = lg[d]
        yd = y[d]
        if ftype == "highloss":
            m = zl - zl.max(1, keepdims=True)
            lse = np.log(np.exp(m).sum(1)) + zl.max(1)
            score = lse - zl[np.arange(len(d)), yd]               # CE per-sample
        else:
            zt = zl[np.arange(len(d)), yd]
            other = zl.copy(); other[np.arange(len(d)), yd] = -np.inf
            score = zt - other.max(1)                             # margin
        idx = np.sort(d[np.argsort(-score, kind="stable")[:n]])
        src = paths.d_logits0(base_seed).name
    man = {
        "dataset": profile.dataset, "type": ftype, "fkey": fkey, "rho": len(idx) / len(d),
        "rho_nominal": rho, "class_id": cid, "basis": basis,
        "base_model_seed": base_seed if basis == "per-seed" else None, "source_logits": src,
        "n_forget": int(len(idx)), "d_indices": idx.tolist(), "v_indices_intersect": [],
        "seed": SPLIT_SEED, "created_from_recipe": "§4.1",
    }
    # §4.3 checks
    assert np.isin(idx, d).all() and len(np.intersect1d(idx, data.v_idx_np)) == 0
    if ftype != "classwise":
        assert len(idx) == int(round(rho * len(d))), "|F| ≠ round(ρ|D|)"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(man), encoding="utf-8")
    return man


def load_manifest(path: Path, data: Data) -> dict:
    man = json.loads(Path(path).read_text(encoding="utf-8"))
    idx = np.asarray(man["d_indices"])
    assert np.isin(idx, data.d_idx_np).all(), "F ⊄ D"
    assert len(np.intersect1d(idx, data.v_idx_np)) == 0, "F ∩ V ≠ ∅"
    assert man["v_indices_intersect"] == []
    return man


def get_manifest(profile: Profile, paths: Paths, data: Data, fkey: str, base_seed: int) -> dict:
    return build_manifest(profile, paths, data, fkey, base_seed)


def check_forget_L0(man: dict, d_logits0: np.ndarray, data: Data, L_F_step0: float):
    """§4.3.3: L_F0 tại θ0 (đo trong pipeline) khớp mean CE per-sample tính từ d_logits0."""
    idx = np.asarray(man["d_indices"])
    z = d_logits0[idx].astype(np.float64)
    y = data.ytr_np[idx]
    lse = np.log(np.exp(z - z.max(1, keepdims=True)).sum(1)) + z.max(1)
    ce = float((lse - z[np.arange(len(idx)), y]).mean())
    if abs(ce - L_F_step0) > FORGET_TOL * max(1.0, abs(ce)):
        raise AssertionError(f"L_F0 lệch manifest: {L_F_step0:.6f} vs {ce:.6f}")
    return ce
