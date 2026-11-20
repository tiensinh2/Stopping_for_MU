"""Retrain oracle (§9): θ_R huấn luyện từ đầu chỉ trên R, cùng recipe §3.1.

Chỉ lưu logits (train toàn bộ + test) và metrics; file này KHÔNG được import bởi collect_traj/score_ralm/rules (guard §1.1).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .config import Paths, Profile
from .data import Data, Split
from .eval_core import eval_full, eval_test, forward_all
from .manifest import get_manifest
from .train import train_model


def retrain(profile: Profile, paths: Paths, fkey: str, base_seed: int, rseed: int, device,
            log=print, force: bool = False) -> dict:
    stem = paths.retrain_stem(fkey, base_seed, rseed)
    out_npz, out_json = Path(str(stem) + ".npz"), Path(str(stem) + ".json")
    if out_npz.exists() and out_json.exists() and not force:
        log(f"[retrain] {stem.name} đã có, bỏ qua"); return json.loads(out_json.read_text(encoding="utf-8"))
    data = Data(profile, device)
    man = get_manifest(profile, paths, data, fkey, base_seed)
    split = Split(data, np.asarray(man["d_indices"]))
    log(f"[retrain] {fkey} base={base_seed} rseed={rseed} |R|={len(split.r)}")
    t0 = time.time()
    model = train_model(profile, data, split.r, 1000 + rseed + 17 * base_seed, device, log)
    stem.parent.mkdir(parents=True, exist_ok=True)
    m, ex = eval_full(model, data, split, profile.amp, profile.eval_batch, check=True)
    L_T, A_T = eval_test(model, data, profile.amp, profile.eval_batch)
    tl = forward_all(model, data, data.xte, profile.eval_batch, profile.amp)
    np.savez_compressed(out_npz, train_logits=ex["logits"].cpu().numpy().astype(np.float16),
                        test_logits=tl.cpu().numpy().astype(np.float16))
    torch.save(model.state_dict(), Path(str(stem) + ".pth"))
    st = json.loads(paths.base_stats(base_seed).read_text(encoding="utf-8"))
    res = {**m, "L_T": L_T, "A_T": A_T, "fkey": fkey, "base_seed": base_seed, "rseed": rseed,
           "L_D_minus_L_D0": m["L_D"] - st["L_D0"],        # §9: độ misspecified của anchor D
           "wall_s": time.time() - t0}
    out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
    log(f"[retrain] A_T={A_T:.4f} L_F={m['L_F']:.4f} (θ0: {st['L_D0']:.4f}->L_D(θ_R)={m['L_D']:.4f})")
    return res
