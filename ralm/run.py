"""CLI điều phối (Kaggle 2×T4). Mỗi lệnh idempotent: job đã xong sẽ bị bỏ qua, nên có thể resume sau khi session chết.

Ví dụ:
  python -m ralm.run info
  python -m ralm.run base      --profile kaggle --seeds 0 1 2
  python -m ralm.run traj      --profile kaggle --methods ga ngp scrub salun tarun --fkeys random_rho0.1 random_rho0.01 --seeds 0
  python -m ralm.run retrain   --profile kaggle --fkeys random_rho0.1 random_rho0.01 --seeds 0
  python -m ralm.run aggregate --profile kaggle --tag trackA
  python -m ralm.run launch    --stage base|traj|retrain --gpus 2 ...     # chia job cho 2 GPU (2 tiến trình)
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

from .config import METHODS, Paths, get_profile


def device_of(gpu: int | None):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu or 0}")
    return torch.device("cpu")


def jobs_split(jobs: list, shard: int, nshards: int) -> list:
    return [j for i, j in enumerate(jobs) if i % nshards == shard]


def cmd_info(a):
    print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.cuda.is_available())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU{i}: {p.name} {p.total_memory / 2**30:.1f} GiB  cc{p.major}.{p.minor}")
    print("methods:", METHODS)
    if not torch.cuda.is_available():
        print("CẢNH BÁO: không có CUDA. Trên Kaggle hãy chọn Accelerator = GPU T4 x2.")


def _common(a):
    prof = get_profile(a.profile, data_root=a.data_root, n_steps=a.n_steps, eval_interval=a.eval_interval,
                       k_min=a.k_min, epochs=a.epochs)
    return prof, Paths(a.out, prof), device_of(a.gpu)


def cmd_base(a):
    from .train import train_base
    prof, paths, dev = _common(a)
    for s in jobs_split(list(a.seeds), a.shard, a.nshards):
        train_base(prof, paths, s, dev, force=a.force)


def cmd_traj(a):
    from .collect_traj import collect
    prof, paths, dev = _common(a)
    jobs = list(itertools.product(a.seeds, a.fkeys, a.methods))
    hp_cli = {"lr": a.lr}
    for seed, fkey, method in jobs_split(jobs, a.shard, a.nshards):
        try:
            collect(prof, paths, method, fkey, seed, dev, n_steps=a.n_steps, hp_cli=hp_cli,
                    save_ckpt_every=a.save_ckpt_every, force=a.force)
        except Exception:  # noqa: BLE001  (§18.3: không skip im lặng -> in lỗi, ghi file, rồi đi tiếp job khác)
            err = paths.root / "logs" / f"ERROR_{method}_{fkey}_seed{seed}.txt"
            err.parent.mkdir(parents=True, exist_ok=True)
            err.write_text(traceback.format_exc(), encoding="utf-8"); print(f"[ERROR] {method} {fkey} seed{seed} -> {err}")


def cmd_retrain(a):
    from .retrain import retrain
    prof, paths, dev = _common(a)
    jobs = list(itertools.product(a.seeds, a.fkeys, range(a.retrain_seeds)))
    for bseed, fkey, rseed in jobs_split(jobs, a.shard, a.nshards):
        retrain(prof, paths, fkey, bseed, rseed, dev, force=a.force)


def cmd_aggregate(a):
    from . import aggregate as AG
    prof, paths, _ = _common(a)
    runs = []
    for method in a.methods:
        for fkey in a.fkeys:
            for seed in a.seeds:
                rid = paths.run_id(method, fkey, seed)
                if not paths.meta(rid).exists():
                    print(f"[skip] thiếu trajectory {rid}"); continue
                o = AG.analyze_run(prof, paths, rid, fkey, seed, delta_F=0.05 if a.delta_f else None,
                                   ema=a.ema)
                if o is None:
                    print(f"[skip] chưa có retrain cho {fkey} base{seed}"); continue
                runs.append(o)
    if not runs:
        print("không có run nào để tổng hợp"); return
    out = AG.summarize(paths, a.tag, runs)
    AG.make_figs(paths, runs, a.tag)
    tm = AG.t_matrix(out["anchors"], out["rules"])
    tm.to_csv(paths.results / f"tmatrix_{a.tag}.csv", index=False)
    print((paths.results / f"rules_table_{a.tag}.md").read_text(encoding="utf-8"))
    print("\nT-matrix WIN:", tm[tm.WIN][["method", "anchor"]].values.tolist())
    print(f"\n-> {paths.results}")


def cmd_launch(a):
    """Chạy `a.stage` trên N GPU bằng N tiến trình con (mỗi tiến trình 1 GPU, 1 shard)."""
    pass_args = list(a.rest)
    n = a.gpus
    if '--gpus' in pass_args:                       # REMAINDER nuốt --gpus nếu đứng sau stage
        i = pass_args.index('--gpus'); n = int(pass_args[i + 1]); del pass_args[i:i + 2]
    n = n or max(1, torch.cuda.device_count())
    procs = []
    for g in range(n):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        cmd = [sys.executable, "-m", "ralm.run", a.stage, "--shard", str(g), "--nshards", str(n), "--gpu", "0"] + pass_args
        print("[launch]", " ".join(cmd))
        procs.append(subprocess.Popen(cmd, env=env))
    rc = [p.wait() for p in procs]
    if any(rc):
        sys.exit(max(rc))


def build_parser():
    p = argparse.ArgumentParser("ralm")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--profile", default="kaggle")
        sp.add_argument("--out", default="./ralm_out")
        sp.add_argument("--data_root", default=None)
        sp.add_argument("--gpu", type=int, default=0)
        sp.add_argument("--shard", type=int, default=0)
        sp.add_argument("--nshards", type=int, default=1)
        sp.add_argument("--seeds", type=int, nargs="+", default=[0])
        sp.add_argument("--fkeys", nargs="+", default=["random_rho0.1"])
        sp.add_argument("--methods", nargs="+", default=["ga", "ngp", "scrub", "salun", "tarun"])
        sp.add_argument("--n_steps", type=int, default=None)
        sp.add_argument("--eval_interval", type=int, default=None)
        sp.add_argument("--k_min", type=int, default=None)
        sp.add_argument("--epochs", type=int, default=None)
        sp.add_argument("--lr", type=float, default=None)
        sp.add_argument("--retrain_seeds", type=int, default=1)
        sp.add_argument("--save_ckpt_every", type=int, default=0)
        sp.add_argument("--force", action="store_true")
        sp.add_argument("--tag", default="trackA")
        sp.add_argument("--ema", type=float, default=None)
        sp.add_argument("--delta_f", action="store_true")

    for name, fn in (("info", cmd_info), ("base", cmd_base), ("traj", cmd_traj), ("retrain", cmd_retrain),
                     ("aggregate", cmd_aggregate)):
        sp = sub.add_parser(name); common(sp); sp.set_defaults(fn=fn)
    sp = sub.add_parser("launch")
    sp.add_argument("stage", choices=["base", "traj", "retrain"])
    sp.add_argument("--gpus", type=int, default=None)
    sp.add_argument("rest", nargs=argparse.REMAINDER)
    sp.set_defaults(fn=cmd_launch)
    return p


def main(argv=None):
    for s in (sys.stdout, sys.stderr):           # console Windows (cp1252) không in được θ, ρ...
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")

    a = build_parser().parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
