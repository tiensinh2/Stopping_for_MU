"""CLI: python -m pre_stop.run <lệnh> --profile kaggle --out <thư mục> --seeds 0 1 2 ..."""
from __future__ import annotations

import argparse
import json
import sys
import time

import torch

from ralm.data import Data

from . import analyze as A
from .config import FTYPES, MAIN_FTYPE, METHODS, GATE1_RATIO, Paths, get_cfg
from .train import check_stage1, prepare_config, prepare_seed, require_pre
from .unlearn import applicable, collect


def _args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["pre", "cfg", "gate1", "traj", "analyze", "all"])
    p.add_argument("--profile", default="kaggle")
    p.add_argument("--out", default="./pre_out")
    p.add_argument("--data_root", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--ftypes", nargs="+", default=list(FTYPES))
    p.add_argument("--methods", nargs="+", default=list(METHODS))
    p.add_argument("--us", type=int, nargs="+", default=[0])
    p.add_argument("--e2", type=int, default=None)
    p.add_argument("--n_epochs", type=int, default=None,
                   help="ép số epoch cố định, chạy đủ (mặc định: chạy đến khi mọi luật PRE online đã quyết định)")
    p.add_argument("--max_epochs", type=int, default=None, help="trần tuỳ chọn (mặc định: KHÔNG có trần)")
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    a = _args(argv)
    cfg = get_cfg(a.profile, e2=a.e2, n_epochs=a.n_epochs, max_epochs=a.max_epochs, data_root=a.data_root)
    paths = Paths(a.out)
    check_stage1(cfg, paths, a.seeds)
    if a.cmd in ("cfg", "gate1", "traj"):
        require_pre(paths, a.seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] {a.cmd} profile={cfg.name} device={device} seeds={a.seeds} ftypes={a.ftypes} methods={a.methods}")
    data = Data(cfg.as_profile(), device) if a.cmd != "analyze" else None
    t0 = time.time()
    if a.cmd in ("pre", "all"):
        for s in a.seeds:
            prepare_seed(cfg, paths, data, s, device, force=a.force)
    if a.cmd in ("cfg", "all"):
        for s in a.seeds:
            for ft in a.ftypes:
                prepare_config(cfg, paths, data, s, ft, device, force=a.force)
    if a.cmd in ("gate1", "all"):
        ok = gate1(cfg, paths, data, a, device)
        if not ok:      # chỉ cảnh báo, không chặn: kết quả vẫn chạy và được báo cáo cùng e_pre/ν
            print("[CẢNH BÁO] median e_pre/ν > 10 ở random-450: tiền đề 'loss trước khi học F ≈ loss oracle' "
                  "không đứng vững. Vẫn chạy tiếp; xem results/gate1.csv và đọc kết quả luật trong ngữ cảnh này.")
    if a.cmd in ("traj", "all"):
        for s in a.seeds:
            for ft in a.ftypes:
                for me in a.methods:
                    if not applicable(me, ft, MAIN_FTYPE):
                        print(f"[traj] bỏ qua {me}/{ft}: phương pháp gốc không định nghĩa cho loại F này")
                        continue
                    for u in a.us:
                        collect(cfg, paths, data, me, ft, s, device, u=u, force=a.force)
    if a.cmd in ("analyze", "all"):
        A.analyze_all(cfg, paths, a.seeds, a.ftypes, [m for m in a.methods], a.us)
        print(f"[run] báo cáo: {paths.results / 'report_auto.md'}")
    print(f"[run] xong {a.cmd} trong {time.time() - t0:.0f}s")
    return 0


def gate1(cfg, paths, data, a, device) -> bool:
    """Cổng 1 (§10): median e_pre/ν > 10 ở random-450 cho cả none và V -> dừng. Đo bằng θ0 và oracle, chưa unlearning."""
    import pandas as pd
    from .data import load_sets
    from .measure import mean_loss_acc
    from .model_util import load_theta0
    rows = []
    for s in a.seeds:
        for ft in a.ftypes:
            sets = load_sets(paths, data, cfg, s, ft)
            m0 = load_theta0(cfg, paths, s, ft, device)
            L_R0, _ = mean_loss_acc(m0, data, cfg, sets.rsub)
            L_V0, _ = mean_loss_acc(m0, data, cfg, sets.v)
            rows.append(A.gate1_row(cfg, paths, s, ft, L_R0, L_V0))
    g = pd.DataFrame(rows)
    paths.results.mkdir(parents=True, exist_ok=True)
    g.to_csv(paths.results / "gate1.csv", index=False)
    print(g.to_string(index=False))
    main = g[g.ftype == MAIN_FTYPE]
    if len(main) == 0:
        return True
    bad = min(main.e_pre_over_nu_none.median(), main.e_pre_over_nu_V.median()) > GATE1_RATIO
    return not bad


if __name__ == "__main__":
    sys.exit(main())

