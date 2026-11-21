"""Unlearning ga / ngplus / ft và quỹ đạo (kịch bản §6, §7).

CÁCH LY (§0.4.1): file này không import eval_oracle và không đụng tới θ_R2. Metric test ghi vào file niêm phong.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from ralm.eval_core import seed_all
from ralm.model import make_model

from .config import ALPHA_NG, LR_GRID, LR_SCAN_LV, LR_SCAN_STEPS, Cfg
from .data import load_sets
from .measure import measure, test_metrics

assert "eval_oracle" not in globals(), "unlearn.py không được import eval_oracle (§0.4.1)"


class Cycler:
    """Duyệt tuần hoàn xáo trộn trên idx với batch cố định."""

    def __init__(self, idx: torch.Tensor, bs: int, device):
        self.idx, self.bs, self.dev = idx, min(bs, len(idx)), device
        self._new()

    def _new(self):
        self.perm = self.idx[torch.randperm(len(self.idx), device=self.dev)]
        self.p = 0

    def next(self) -> torch.Tensor:
        if self.p + self.bs > len(self.perm):
            self._new()
        b = self.perm[self.p:self.p + self.bs]
        self.p += self.bs
        return b


class Unlearner:
    def __init__(self, cfg: Cfg, data, sets, model, method: str, lr: float, device):
        assert method in ("ga", "ngplus", "ft"), method
        self.cfg, self.data, self.model, self.method, self.dev = cfg, data, model, method, device
        self.opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg.momentum)   # §6: momentum 0.9
        self.use_amp = cfg.amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.cf = Cycler(sets.f, cfg.batch_f, device)      # batch F = min(64, |F|)
        self.cr = Cycler(sets.r, cfg.batch_r, device)      # batch R = 128 từ R

    def _ce(self, idx):
        x, y = self.data.batch(idx, augment=True)
        with torch.autocast(device_type=self.dev.type, dtype=torch.float16, enabled=self.use_amp):
            out = self.model(x)
        return F.cross_entropy(out.float(), y)

    def step(self) -> float:
        m = self.method
        if m == "ga":
            loss = -self._ce(self.cf.next())
        elif m == "ngplus":
            loss = ALPHA_NG * self._ce(self.cr.next()) - (1 - ALPHA_NG) * self._ce(self.cf.next())
        else:
            loss = self._ce(self.cr.next())
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()
        return float(loss.detach())


def _load_theta0(cfg, paths, seed, ftype, device):
    m = make_model(cfg.as_profile(), device)
    m.load_state_dict(torch.load(paths.theta0(seed, ftype), map_location=device))
    return m


def scan_lr(cfg: Cfg, paths, data, method: str, seed: int, ftype: str, device, log=print) -> dict:
    """§6: lr lớn nhất thuộc lưới sao cho L_V(bước 200) <= 1.2·L_V(0). Chỉ dùng V."""
    sets = load_sets(paths, data, cfg, seed, ftype)
    n = min(LR_SCAN_STEPS, cfg.n_steps)
    res, ok_lrs = [], []
    for lr in sorted(LR_GRID):
        seed_all(seed)
        model = _load_theta0(cfg, paths, seed, ftype, device)
        model.eval()
        LV0 = float(measure(model, data, cfg, sets.v)[1].double().mean())
        um = Unlearner(cfg, data, sets, model, method, lr, device)
        model.train()
        try:
            for _ in range(n):
                um.step()
            LV = float(measure(model, data, cfg, sets.v)[1].double().mean())
            ok = bool(np.isfinite(LV) and LV <= LR_SCAN_LV * LV0)
        except FloatingPointError:
            LV, ok = float("nan"), False
        res.append({"lr": lr, "L_V": LV, "L_V0": LV0, "ratio": LV / LV0 if LV0 else None, "ok": ok})
        log(f"[lr-scan] {method} lr={lr:g}: L_V {LV:.4f} / L_V0 {LV0:.4f} -> {'đạt' if ok else 'không đạt'}")
        if ok:
            ok_lrs.append(lr)
    chosen = max(ok_lrs) if ok_lrs else min(LR_GRID)
    out = {"method": method, "seed": seed, "ftype": ftype, "steps": n, "chosen_lr": chosen,
           "none_passed": not ok_lrs, "scan": res}
    p = paths.lr_json(method)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    if not ok_lrs:
        log(f"[lr-scan] CẢNH BÁO {method}: không lr nào đạt tiêu chí; dùng lr nhỏ nhất {chosen:g} và ghi cờ")
    return out


def collect(cfg: Cfg, paths, data, method: str, ftype: str, seed: int, device, u: int = 0,
            lr: float | None = None, log=print, force: bool = False) -> dict:
    rid = paths.run_id(method, ftype, seed, u)
    jl, arr_p, meta_p, test_p = paths.traj_jsonl(rid), paths.traj_arrays(rid), paths.meta(rid), paths.traj_test(rid)
    if meta_p.exists() and arr_p.exists() and not force:
        log(f"[traj] {rid} đã xong, bỏ qua")
        return json.loads(meta_p.read_text(encoding="utf-8"))
    if lr is None:
        lr = json.loads(paths.lr_json(method).read_text(encoding="utf-8"))["chosen_lr"]
    sets = load_sets(paths, data, cfg, seed, ftype)
    seed_all(seed + 1000 * u)
    model = _load_theta0(cfg, paths, seed, ftype, device)
    um = Unlearner(cfg, data, sets, model, method, lr, device)
    jl.parent.mkdir(parents=True, exist_ok=True)
    jl.write_text("", encoding="utf-8")
    test_p.write_text("", encoding="utf-8")
    arrays = {k: [] for k in ("steps", "loss_F", "loss_V", "logits_F", "logits_V")}

    def snapshot(step: int, t_wall: float, last_loss):
        model.eval()
        lgR, lsR = measure(model, data, cfg, sets.rsub)
        lgF, lsF = measure(model, data, cfg, sets.f)
        lgV, lsV = measure(model, data, cfg, sets.v)
        for t in (lsR, lsF, lsV):
            if not torch.isfinite(t).all():
                raise FloatingPointError("per-sample loss không hữu hạn (model phân kỳ)")
        acc = lambda lg, idx: float((lg.argmax(1) == data.ytr[idx]).double().mean())    # noqa: E731
        row = {"run_id": rid, "step": step, "wall_time_s": round(t_wall, 3),
               "L_R": float(lsR.double().mean()), "A_R": acc(lgR, sets.rsub),
               "L_F": float(lsF.double().mean()), "A_F": acc(lgF, sets.f),
               "L_V": float(lsV.double().mean()), "A_V": acc(lgV, sets.v),
               "mean_conf_F": float(F.softmax(lgF, 1).max(1).values.double().mean()),
               "train_loss_last_step": last_loss, "lr": lr}
        L_T, A_T = test_metrics(model, data, cfg)                         # niêm phong: file riêng (§7)
        with open(jl, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n"); f.flush()
        with open(test_p, "a", encoding="utf-8") as f:
            f.write(json.dumps({"run_id": rid, "step": step, "L_T": L_T, "A_T": A_T,
                                "post_selection": True}) + "\n"); f.flush()
        arrays["steps"].append(step)
        arrays["loss_F"].append(lsF.cpu().numpy().astype(np.float32))
        arrays["loss_V"].append(lsV.cpu().numpy().astype(np.float32))
        arrays["logits_F"].append(lgF.cpu().numpy().astype(np.float16))
        arrays["logits_V"].append(lgV.cpu().numpy().astype(np.float16))
        return row

    t0 = time.time()
    snapshot(0, 0.0, None)
    log(f"[traj] {rid} |F|={len(sets.f)} lr={lr:g}")
    model.train()
    last, diverged = None, None
    for step in range(1, cfg.n_steps + 1):
        try:
            last = um.step()
            if step % cfg.interval == 0 or step == cfg.n_steps:
                row = snapshot(step, time.time() - t0, last)
                model.train()
                if step % (cfg.interval * 20) == 0 or step == cfg.n_steps:
                    log(f"  step {step:5d} L_R {row['L_R']:.4f} L_F {row['L_F']:.4f} L_V {row['L_V']:.4f} "
                        f"A_F {row['A_F']:.3f} [{time.time() - t0:.0f}s]")
        except FloatingPointError as e:     # ghi rõ, không bỏ qua im lặng (§17.3)
            diverged = {"step": step, "error": str(e)}
            log(f"[traj] {rid} PHÂN KỲ tại bước {step}: quỹ đạo bị cắt ở đây (ghi cờ diverged)")
            break
    np.savez_compressed(arr_p, steps=np.array(arrays["steps"]), loss_F=np.stack(arrays["loss_F"]),
                        loss_V=np.stack(arrays["loss_V"]), logits_F=np.stack(arrays["logits_F"]),
                        logits_V=np.stack(arrays["logits_V"]), f_idx=sets.f_np, v_idx=sets.v_np,
                        f_y=data.ytr_np[sets.f_np], v_y=data.ytr_np[sets.v_np])
    meta = {"run_id": rid, "method": method, "ftype": ftype, "seed": seed, "u": u, "lr": lr,
            "n_steps": cfg.n_steps, "interval": cfg.interval, "n_snapshots": len(arrays["steps"]),
            "diverged": diverged, "wall_s": time.time() - t0, "arrays": arr_p.name,
            "bn_mode": "train (BN theo batch)"}
    meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
