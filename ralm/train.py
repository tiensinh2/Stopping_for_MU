"""Huáº¥n luyá»‡n Î¸0 (trÃªn D) vÃ  Î¸_R (trÃªn R) theo recipe Â§3.1. DÃ¹ng chung má»™t hÃ m `train_model`.

SGD momentum 0.9, wd 5e-4, lr 0.1, cosine + warmup 5 epoch (theo iteration), batch 128, crop+flip.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import Paths, Profile
from .data import Data, Split
from .eval_core import eval_full, eval_test, seed_all
from .model import make_model


def lr_at(it: int, total: int, warm_iters: int, base_lr: float) -> float:
    if it < warm_iters:
        return base_lr * (it + 1) / warm_iters
    p = (it - warm_iters) / max(1, total - warm_iters)
    return 0.5 * base_lr * (1 + math.cos(math.pi * p))


def train_model(profile: Profile, data: Data, train_idx: torch.Tensor, seed: int, device, log=print):
    """Huáº¥n luyá»‡n ResNet-18 tá»« Ä‘áº§u trÃªn cÃ¡c chá»‰ má»¥c train_idx. Tráº£ vá» model (eval mode)."""
    seed_all(seed)
    model = make_model(profile, device)
    opt = torch.optim.SGD(model.parameters(), lr=profile.lr, momentum=profile.momentum,
                          weight_decay=profile.wd)
    n = len(train_idx)
    steps_per_epoch = math.ceil(n / profile.batch)
    total, warm = profile.epochs * steps_per_epoch, profile.warmup * steps_per_epoch
    use_amp = profile.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    it, t0 = 0, time.time()
    for ep in range(profile.epochs):
        model.train()
        perm = train_idx[torch.randperm(n, device=device)]
        tot_loss, tot_acc = 0.0, 0.0
        for s in range(0, n, profile.batch):
            idx = perm[s:s + profile.batch]
            x, y = data.batch(idx, augment=True)
            for g in opt.param_groups:
                g["lr"] = lr_at(it, total, warm, profile.lr)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
                loss = F.cross_entropy(out.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            it += 1
            tot_loss += float(loss.detach()) * len(idx)
            tot_acc += float((out.argmax(1) == y).sum())
        if ep % 5 == 0 or ep == profile.epochs - 1:
            log(f"  ep {ep + 1}/{profile.epochs} loss {tot_loss / n:.4f} acc {tot_acc / n:.4f} "
                f"lr {lr_at(it - 1, total, warm, profile.lr):.4f} t {time.time() - t0:.0f}s")
    model.eval()
    return model


def train_base(profile: Profile, paths: Paths, seed: int, device, log=print, force: bool = False):
    """Î¸0 trÃªn D. Ghi ckpt, base_stats_*.json, d_logits0_*.npy (logits trÃªn toÃ n bá»™ train, float32)."""
    ck, st = paths.base_ckpt(seed), paths.base_stats(seed)
    if ck.exists() and st.exists() and not force:
        log(f"[base] seed {seed} Ä‘Ã£ cÃ³, bá» qua"); return
    data = Data(profile, device)
    # Î¸0 Ä‘Æ°á»£c Ä‘o báº±ng Split giáº£ (F = 1 máº«u) vÃ¬ chÆ°a cÃ³ forget set; identity check táº¯t á»Ÿ dÆ°á»›i.
    log(f"[base] train Î¸0 seed={seed} |D|={len(data.d_idx_np)} |V|={len(data.v_idx_np)}")
    d_t = torch.from_numpy(data.d_idx_np).to(device)
    model = train_model(profile, data, d_t, seed, device, log)
    ck.parent.mkdir(parents=True, exist_ok=True); st.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ck)
    # Split rá»—ng khÃ´ng dÃ¹ng Ä‘Æ°á»£c cho identity (rho=0): Ä‘o trá»±c tiáº¿p báº±ng má»™t F giáº£ (1 máº«u) vÃ  táº¯t check.
    dummy = Split(data, data.d_idx_np[:1])
    m, ex = eval_full(model, data, dummy, profile.amp, profile.eval_batch, check=False)
    L_T, A_T = eval_test(model, data, profile.amp, profile.eval_batch)
    np.save(paths.d_logits0(seed), ex["logits"].cpu().numpy().astype(np.float32))
    stats = {
        "seed": seed, "dataset": profile.dataset, "profile": profile.name,
        "L_D0": m["L_D"], "L_V0": m["L_V"], "L_T0": L_T,
        "A_D0": m["A_D"], "A_V0": m["A_V"], "A_T0": A_T,
        "per_sample_logit_file": paths.d_logits0(seed).name,
        "gate_min_test_acc": profile.min_test_acc,
        "gate_passed": profile.min_test_acc is None or A_T >= profile.min_test_acc,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "recipe": "Â§3.1",
    }
    st.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log(f"[base] seed {seed}: A_T={A_T:.4f} L_D0={m['L_D']:.4f} L_V0={m['L_V']:.4f}")
    if not stats["gate_passed"]:
        raise SystemExit(f"[GATE] Î¸0 test acc {A_T:.4f} < {profile.min_test_acc} (Â§3.1). "
                         "Dá»ªNG vÃ  bÃ¡o user; khÃ´ng tá»± Ä‘á»•i recipe.")
