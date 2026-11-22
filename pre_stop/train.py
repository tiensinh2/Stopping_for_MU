"""Giai đoạn 1 (θ_pre), giai đoạn 2 (θ0 có replay, cổng w_F) và oracle θ_R2 (kịch bản §2, §5)."""
from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from ralm.eval_core import seed_all
from ralm.model import make_model
from ralm.train import lr_at

from . import pre_info
from .config import GATE_F, M_ORACLE, PHI, WF_ORDER, Cfg
from .data import Sets, choose_forget, load_split, save_forget, save_split, stage_split
from .eval_oracle import save_oracle
from .measure import mean_loss_acc, measure, test_metrics


class EarlyStop:
    """Dừng khi loss không cải thiện so với giá trị tốt nhất sau `patience` epoch liên tiếp.

    Chỉ đếm epoch xấu từ epoch `start` trở đi (bỏ qua giai đoạn warmup). update() trả về (cải thiện, dừng).
    """

    def __init__(self, patience: int, start: int = 0):
        self.patience, self.start = patience, start
        self.best, self.best_ep, self.bad = float("inf"), -1, 0

    def update(self, ep: int, loss: float):
        improved = loss < self.best
        if improved:
            self.best, self.best_ep, self.bad = loss, ep, 0
        elif ep >= self.start:
            self.bad += 1
        return improved, self.bad >= self.patience


def _fit(cfg: Cfg, data, model, epoch_idx, epochs: int, base_lr: float, warm_epochs: int, device,
         log=print, rec=None, rec_pos=None, tag="", patience=None, info=None):
    """SGD momentum 0.9 nesterov, wd 5e-4, cosine theo iteration. epoch_idx(ep) -> chỉ mục của epoch đó.

    rec/rec_pos: ghi loss train-mode lần xuất hiện đầu tiên của từng mẫu F trong epoch 0 (P-log, §3).
    patience: nếu có, dừng sớm theo loss train trung bình của epoch; cuối cùng nạp lại trọng số của epoch tốt nhất.
    info: dict được điền epochs_run / best_epoch / best_loss / stopped_early / lr_at_stop.
    """
    opt = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=cfg.momentum, nesterov=True,
                          weight_decay=cfg.wd)
    n0 = len(epoch_idx(0))
    spe = -(-n0 // cfg.batch)
    total, warm = epochs * spe, warm_epochs * spe
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    it, t0 = 0, time.time()
    es = EarlyStop(patience, start=warm_epochs) if patience else None
    best_state, ran, stopped = None, 0, False
    for ep in range(epochs):
        model.train()
        idx_all = epoch_idx(ep)
        n = len(idx_all)
        perm = idx_all[torch.randperm(n, device=device)]
        tl = ta = 0.0
        for s in range(0, n, cfg.batch):
            idx = perm[s:s + cfg.batch]
            x, y = data.batch(idx, augment=True)
            for g in opt.param_groups:
                g["lr"] = lr_at(it, total, warm, base_lr)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
            ls = F.cross_entropy(out.float(), y, reduction="none")
            if rec is not None and ep == 0:
                p = rec_pos[idx]
                m = (p >= 0) & torch.isnan(rec[p.clamp(min=0)])
                rec[p[m]] = ls.detach()[m]
            loss = ls.mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            it += 1
            tl += float(loss.detach()) * len(idx)
            ta += float((out.argmax(1) == y).sum())
        ran = ep + 1
        stop = False
        if es is not None:
            improved, stop = es.update(ep, tl / n)
            if improved:
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == epochs - 1 or stop:
            log(f"  {tag} ep {ep + 1}/{epochs} loss {tl / n:.4f} acc {ta / n:.4f} t {time.time() - t0:.0f}s")
        if stop:
            stopped = True
            log(f"  {tag} DỪNG SỚM sau {ran} epoch: loss train không cải thiện {patience} epoch liên tiếp; "
                f"nạp lại epoch {es.best_ep + 1} (loss {es.best:.4f}), lr hiện tại {opt.param_groups[0]['lr']:.5f}")
            break
    if es is not None and best_state is not None:
        model.load_state_dict(best_state)
    if info is not None:
        info.update({"epochs_run": ran, "best_epoch": (es.best_ep + 1) if es else ran,
                     "best_loss": float(es.best) if es else None, "stopped_early": stopped,
                     "lr_at_stop": float(opt.param_groups[0]["lr"])})
    model.eval()
    return model


def stage1_signature(cfg: Cfg) -> dict:
    """Mọi thứ quyết định θ_pre. Lưu cùng stats để không nạp nhầm cache của cách huấn luyện khác."""
    return {"e1": cfg.e1, "patience1": cfg.patience1, "warm1": cfg.warm1, "lr1": cfg.lr1,
            "monitor": "train_loss_R1", "restore_best": True}


def check_stage1(cfg: Cfg, paths, seeds):
    """θ_pre đổi thì θ0, oracle, quỹ đạo đều đổi: cache sinh bởi cách huấn luyện khác phải bị từ chối."""
    sig = stage1_signature(cfg)
    for s in seeds:
        p = paths.pre_stats(s)
        if p.exists() and json.loads(p.read_text(encoding="utf-8")).get("stage1_signature") != sig:
            raise SystemExit(f"[cache cũ] θ_pre của seed {s} trong {paths.root} được train bằng cấu hình khác {sig}. "
                             "θ0, oracle và quỹ đạo phụ thuộc vào nó nên không dùng lại được: hãy chạy với thư mục OUT "
                             "mới và để RESTORE_FROM rỗng.")


def train_stage1(cfg: Cfg, data, r1: torch.Tensor, seed: int, device, log=print, info=None):
    seed_all(seed)
    model = make_model(cfg.as_profile(), device)
    return _fit(cfg, data, model, lambda ep: r1, cfg.e1, cfg.lr1, cfg.warm1, device, log, tag="[stage1]",
                patience=cfg.patience1, info=info)


def train_stage2(cfg: Cfg, data, sets: Sets, pre_state: dict, w_F: int, seed: int, device, with_f: bool = True,
                 log=print, rec=None):
    """Từ θ_pre: mỗi epoch = R2 + F×w_F (nếu with_f) + replay φ·|R1| mẫu ngẫu nhiên từ R1."""
    seed_all(seed)
    model = make_model(cfg.as_profile(), device)
    model.load_state_dict(pre_state)
    n_rep = int(round(PHI * len(sets.r1)))

    def epoch_idx(ep):
        parts = [sets.r2] + ([sets.f] * w_F if with_f else [])
        parts.append(sets.r1[torch.randperm(len(sets.r1), device=device)[:n_rep]])
        return torch.cat(parts)

    pos = None
    if rec is not None:
        pos = torch.full((data.n_train,), -1, dtype=torch.long, device=device)
        pos[sets.f] = torch.arange(len(sets.f), device=device)
    return _fit(cfg, data, model, epoch_idx, cfg.e2, cfg.lr2, 0, device, log, rec=rec, rec_pos=pos,
                tag="[stage2]" if with_f else "[oracle]")


def prepare_seed(cfg: Cfg, paths, data, seed: int, device, log=print, force: bool = False):
    ck, st = paths.pre_ckpt(seed), paths.pre_stats(seed)
    if ck.exists() and st.exists() and paths.split(seed).exists() and not force:
        log(f"[pre] seed {seed} đã có θ_pre, bỏ qua")
        return
    r1, p = stage_split(data, seed)
    save_split(paths, seed, r1, p)
    log(f"[pre] seed {seed}: train θ_pre trên R1 |R1|={len(r1)} |P|={len(p)}")
    info: dict = {}
    model = train_stage1(cfg, data, torch.from_numpy(r1).to(device), seed, device, log, info)
    ck.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ck)
    L_T, A_T = test_metrics(model, data, cfg)      # chỉ ghi file stats; luật không đọc file này
    stats = {"seed": seed, "L_T_pre": L_T, "A_T_pre": A_T, "gate_min_acc": cfg.min_pre_acc,
             "stage1_signature": stage1_signature(cfg), "stage1_run": info,
             "gate_passed": cfg.min_pre_acc is None or A_T >= cfg.min_pre_acc,
             "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    st.parent.mkdir(parents=True, exist_ok=True)
    st.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log(f"[pre] seed {seed}: A_T(θ_pre)={A_T:.4f}")
    if not stats["gate_passed"]:
        raise SystemExit(f"[GATE] θ_pre test acc {A_T:.4f} < {cfg.min_pre_acc} (§5). DỪNG và báo user; "
                         "không tự đổi công thức.")


def prepare_config(cfg: Cfg, paths, data, seed: int, ftype: str, device, log=print, force: bool = False):
    """F, P (đóng băng), giai đoạn 2 (θ0, cổng w_F) và M oracle θ_R2 cho một (seed, ftype)."""
    done = (paths.theta0(seed, ftype).exists() and paths.stage2_stats(seed, ftype).exists()
            and all(paths.oracle(seed, ftype, m).exists() for m in range(M_ORACLE)))
    if done and not force:
        log(f"[cfg] {ftype} seed {seed} đã xong, bỏ qua")
        return
    r1, p = load_split(paths, seed)
    pre_state = torch.load(paths.pre_ckpt(seed), map_location=device)
    mpre = make_model(cfg.as_profile(), device)
    mpre.load_state_dict(pre_state)
    mpre.eval()
    loss_p = None
    if ftype == "hard-450":
        _, ls = measure(mpre, data, cfg, torch.from_numpy(p).to(device))
        loss_p = ls.cpu().numpy()
    f = choose_forget(ftype, cfg, p, data.ytr_np, seed, loss_p)
    save_forget(paths, seed, ftype, f)
    sets = Sets(data, cfg, r1, p, f, ftype)

    # --- P: đo θ_pre trên F và V (eval, không augmentation), đóng băng TRƯỚC giai đoạn 2
    lg, ls = measure(mpre, data, cfg, sets.f)
    L_V_pre, _ = mean_loss_acc(mpre, data, cfg, sets.v)
    A_F_pre = float((lg.argmax(1) == data.ytr[sets.f]).double().mean())
    c_log = L_V_pre / max(pre_info.train_mode_loss(mpre, data, cfg, sets.v, seed), 1e-12)
    P = pre_info.freeze_P(paths, seed, ftype, ls.cpu().numpy(), L_V_pre, A_F_pre, c_log)
    L_F_pre = P["L_F_pre"]
    log(f"[cfg] {ftype} seed {seed}: |F|={len(f)} L_F^pre={L_F_pre:.4f} L_V^pre={L_V_pre:.4f} c_log={c_log:.3f}")

    # --- giai đoạn 2 với cổng 'F đã được học' (§2.3)
    gate_ok, tried = False, []
    for wF in WF_ORDER:
        rec = torch.full((len(sets.f),), float("nan"), device=device)
        t_start = time.time()
        m0 = train_stage2(cfg, data, sets, pre_state, wF, 100 + seed, device, True, log, rec)
        L_F0, _ = mean_loss_acc(m0, data, cfg, sets.f)
        gate_ok = L_F0 <= GATE_F * L_F_pre
        tried.append({"w_F": wF, "L_F0": L_F0, "ok": bool(gate_ok)})
        log(f"[cfg] w_F={wF}: L_F(θ0)={L_F0:.4f} vs {GATE_F}·L_F^pre={GATE_F * L_F_pre:.4f} -> "
            f"{'đạt' if gate_ok else 'chưa đạt'}")
        if gate_ok:
            break
    assert not torch.isnan(rec).any(), "P-log còn NaN: có mẫu F không xuất hiện trong epoch 1"
    paths.theta0(seed, ftype).parent.mkdir(parents=True, exist_ok=True)
    torch.save(m0.state_dict(), paths.theta0(seed, ftype))
    np.save(paths.P_log(seed, ftype), (rec.cpu().numpy() * c_log).astype(np.float32))
    paths.stage2_stats(seed, ftype).write_text(json.dumps(
        {"seed": seed, "ftype": ftype, "w_F": wF, "gate_passed": bool(gate_ok), "tried": tried,
         "L_F0": L_F0, "L_F_pre": L_F_pre, "t_stage2_start": t_start, "t_P_frozen": P["t_frozen"],
         "e2": cfg.e2}, indent=2), encoding="utf-8")

    # --- oracle path-matched θ_R2 ×M (cùng công thức, không có F)
    for m in range(M_ORACLE):
        mo = train_stage2(cfg, data, sets, pre_state, 1, 200 + 10 * seed + m, device, False, log)
        save_oracle(paths.oracle(seed, ftype, m), mo, data, cfg, sets)
    log(f"[cfg] {ftype} seed {seed}: xong θ0 (w_F={wF}) + {M_ORACLE} oracle")
