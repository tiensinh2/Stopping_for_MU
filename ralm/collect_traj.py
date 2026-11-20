"""Orchestrator: load θ0 + manifest, chạy một phương pháp unlearning, snapshot mỗi `eval_interval` bước.

CÁCH LY (§1.1): file này không được import retrain / eval_oracle và không được đụng tới θ_R.
Test set KHÔNG được dùng để tính metric chọn k: logits test được ghi ra file riêng (post_selection=true)
mà score_ralm.py từ chối đọc; chỉ aggregate/eval_oracle mở sau khi đã chọn k̂.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

from .config import Paths, Profile
from .data import Data, Split
from .eval_core import emb_drift, eval_full, flat_params, forward_all, seed_all, weight_dist
from .manifest import check_forget_L0, dsub_indices, get_manifest
from .model import make_model
from .unlearn import make_unlearner

# ---- assert cách ly (§1.1)
from .guard import assert_no_oracle_imports
assert_no_oracle_imports(__file__)   # §1.1



def _f16(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float16)


def collect(profile: Profile, paths: Paths, method: str, fkey: str, seed: int, device,
            n_steps: int | None = None, hp_cli: dict | None = None, save_ckpt_every: int = 0,
            log=print, force: bool = False) -> dict:
    rid = paths.run_id(method, fkey, seed)
    jl, meta_p = paths.traj_jsonl(rid), paths.meta(rid)
    arr_p = jl.with_name(f"traj_{rid}_arrays.npz")
    if meta_p.exists() and arr_p.exists() and not force:
        log(f"[traj] {rid} đã xong, bỏ qua"); return json.loads(meta_p.read_text(encoding="utf-8"))

    n_steps = n_steps or profile.n_steps
    interval = profile.eval_interval
    seed_all(seed)                                              # seed_unlearn = seed_train_base (§1.4)
    data = Data(profile, device)
    man = get_manifest(profile, paths, data, fkey, seed)
    split = Split(data, np.asarray(man["d_indices"]))
    dsub = dsub_indices(data, profile.dsub_n)
    dsub_t = torch.from_numpy(dsub).to(device)

    model = make_model(profile, device)
    sd0 = torch.load(paths.base_ckpt(seed), map_location=device)
    model.load_state_dict(sd0)
    theta0_state = {k: v.clone() for k, v in sd0.items()}
    hp = profile.hp(method, **(hp_cli or {}))
    um = make_unlearner(method, model, theta0_state, data, split, hp, profile, device)
    d_logits0 = np.load(paths.d_logits0(seed))

    jl.parent.mkdir(parents=True, exist_ok=True)
    jl.write_text("", encoding="utf-8")
    ref_flat = flat_params(model)
    arrays = {"steps": [], "dperloss": [], "vlogits": [], "dsublogits": [], "flogits": [], "tlogits": []}
    feat0_V = None

    def snapshot(step: int, t_wall: float, gnorm, last_loss):
        nonlocal feat0_V
        model.eval()
        m, ex = eval_full(model, data, split, profile.amp, profile.eval_batch, want_feat=True)
        if step == 0:
            feat0_V = ex["feat_V"].clone()
            check_forget_L0(man, d_logits0, data, m["L_F"])           # §4.3.3
        row = {
            "run_id": rid, "step": step, "wall_time_s": round(t_wall, 3),
            **m, "L_T": None, "A_T": None,
            "w_dist": weight_dist(model, ref_flat),                    # T8
            "emb_drift_V": emb_drift(ex["feat_V"], feat0_V),           # T9
            "grad_norm_last_step": gnorm, "train_loss_last_step": last_loss, "lr": um.lr(),
            "post_selection": False,
        }
        with open(jl, "a", encoding="utf-8") as f:                                       # §1.3: ghi + flush mỗi snapshot
            f.write(json.dumps(row) + "\n"); f.flush()
        lg = ex["logits"]
        arrays["steps"].append(step)
        arrays["dperloss"].append(_f16(ex["loss"][split.d]))           # §0A.6.3
        arrays["vlogits"].append(_f16(lg[split.v]))
        arrays["dsublogits"].append(_f16(lg[dsub_t]))
        arrays["flogits"].append(_f16(lg[split.f]))                    # cho G_logit_F offline
        tl = forward_all(model, data, data.xte, profile.eval_batch, profile.amp)
        arrays["tlogits"].append(_f16(tl))                             # post_selection: chỉ aggregate đọc
        if save_ckpt_every and step > 0 and step % save_ckpt_every == 0:
            d = paths.snap_dir(rid); d.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), d / f"step{step:04d}.pth")
        return row

    def grad_norm():
        g = [p.grad for p in model.parameters() if p.grad is not None]
        return float(torch.sqrt(sum((x.float() ** 2).sum() for x in g))) if g else 0.0

    t0 = time.time()
    um.prepare()
    snapshot(0, 0.0, None, None)
    log(f"[traj] {rid} ρ={split.rho:.4f} |F|={len(split.f)} hp={hp}")
    model.train()
    last_loss, gnorm = None, None
    for step in range(1, n_steps + 1):
        last_loss = um.step()
        um.t += 1
        if step % interval == 0 or step == n_steps:
            gnorm = grad_norm()
            row = snapshot(step, time.time() - t0, gnorm, last_loss)
            model.train()
            if step % (interval * 8) == 0 or step == n_steps:
                log(f"  step {step:5d} L_R {row['L_R']:.4f} L_F {row['L_F']:.4f} L_V {row['L_V']:.4f} "
                    f"A_F {row['A_F']:.3f} A_V {row['A_V']:.3f}  [{time.time() - t0:.0f}s]")
    np.savez_compressed(arr_p, steps=np.array(arrays["steps"]),
                        dperloss=np.stack(arrays["dperloss"]), vlogits=np.stack(arrays["vlogits"]),
                        dsublogits=np.stack(arrays["dsublogits"]), flogits=np.stack(arrays["flogits"]),
                        tlogits=np.stack(arrays["tlogits"]),
                        d_idx=split.d_np, v_idx=split.v_np, f_idx=split.f_np, dsub_idx=dsub,
                        d_y=data.ytr_np[split.d_np], v_y=data.ytr_np[split.v_np],
                        f_y=data.ytr_np[split.f_np], dsub_y=data.ytr_np[dsub], t_y=data.yte_np)
    meta = {"run_id": rid, "method": method, "fkey": fkey, "seed": seed, "rho": split.rho, "hp": hp,
            "n_steps": n_steps, "eval_interval": interval, "profile": profile.name,
            "n_snapshots": len(arrays["steps"]), "wall_s": time.time() - t0,
            "n_forward_images_per_snapshot": data.n_train,
            "arrays": arr_p.name, "extra": {"mask_frac": getattr(um, "mask_frac", None)}}
    meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
