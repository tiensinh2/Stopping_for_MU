"""Test PRE-stop (kịch bản §14 + Phụ lục A). Chạy: pytest tests/test_pre_stop.py -x -q  (CPU, dữ liệu giả)."""
import json
import shutil

import numpy as np
import pytest
import torch

from pre_stop import rules as R
from pre_stop.config import M_ORACLE, Paths, get_cfg
from pre_stop.data import Sets
from pre_stop.eval_oracle import noise_floor


# ---------------------------------------------------------------- Phụ lục A (test 3 §14)
def _appendix_traj():
    LF = [0.02, 0.10, 0.30, 0.52, 1.10]
    LR = [0.030, 0.031, 0.036, 0.045, 0.400]
    return [{"epoch": k, "L_F": LF[k], "L_R": LR[k], "L_V": 0.30, "A_F": 0.5, "A_V": 0.9,
             "loss_F": np.full(4, LF[k])} for k in range(5)]


def _ctx(**kw):
    return R.Ctx(t=np.full(4, 0.45), L_R0=0.03, L_V0=0.30, **kw)


def test_appendix_A_scores_four_digits():
    ctx, tr = _ctx(), _appendix_traj()
    assert ctx.s == pytest.approx(0.45)
    exp = [(0.9556, 0.0000, 0.9556), (0.7778, 0.0022, 0.7800), (0.3333, 0.0133, 0.3467),
           (0.1556, 0.0333, 0.1889), (1.4444, 0.8222, 2.2667)]
    for sn, (dF, dR, J) in zip(tr, exp):
        assert round(R.delta_F(sn, ctx), 4) == dF
        assert round(R.delta_R(sn, ctx), 4) == dR
        assert round(R.J7(sn, ctx), 4) == J


def test_appendix_A_barrier_and_choices():
    ctx, tr = _ctx(), _appendix_traj()
    assert [ctx.admissible(sn) for sn in tr] == [True, True, True, True, False]       # ngưỡng 0.05
    assert R.run_online(R.ArgMin(ctx, R.J7, "R7"), tr)["k"] == 3                       # R7 chọn k=3
    ctx1 = _ctx(p=1)                                                                    # R6 với p=1
    assert R.run_online(R.FirstHit(ctx1, lambda sn, c: sn["L_F"] >= c.s, "R6"), tr)["k"] == 3
    tr[3]["L_R"] = 0.06                                                                 # k=3 bị rào chắn loại
    out = R.run_online(R.FirstHit(ctx1, lambda sn, c: sn["L_F"] >= c.s, "R6"), tr)
    assert out["triggered"] is False and out["k"] == 2 and out["no_admissible"] is False


def test_first_hit_needs_p_consecutive():
    ctx = _ctx(p=3)
    tr = _appendix_traj()
    out = R.run_online(R.FirstHit(ctx, lambda sn, c: sn["L_F"] >= c.s, "R6"), tr)
    assert out["triggered"] is False                      # chỉ 1 snapshot chấp nhận được có L_F >= s
    assert out["k"] == 3


def test_no_admissible_flag():
    tr = _appendix_traj()
    for sn in tr:
        sn["L_V"] = 9.0
    out = R.run_online(R.ArgMin(_ctx(), R.J7, "R7"), tr)
    assert out["k"] == 0 and out["no_admissible"] is True


# ---------------------------------------------------------------- test 1: online == offline (không nhìn trước)
def test_online_equals_offline_replay():
    rs = np.random.RandomState(0)
    n = 60
    LF = np.cumsum(rs.rand(n) * 0.05) + 0.02
    tr = [{"epoch": k, "L_F": LF[k], "L_R": 0.03 + 0.0005 * k, "L_V": 0.3, "A_F": 0.5, "A_V": 0.9,
           "loss_F": LF[k] + 0.1 * rs.randn(8)} for k in range(n)]
    ctx = R.Ctx(t=np.full(8, 0.9), L_R0=0.03, L_V0=0.3)
    for name, mk in {"R6": lambda: R.FirstHit(ctx, lambda sn, c: sn["L_F"] >= c.s, "R6"),
                     "R7": lambda: R.ArgMin(ctx, R.J7, "R7"), "R8": lambda: R.ArgMin(ctx, R.J8, "R8"),
                     "R9": lambda: R.FirstHit(ctx, lambda sn, c: float(np.median(sn["loss_F"] - c.t)) >= 0, "R9")}.items():
        full = R.run_online(mk(), tr)
        if full["triggered"]:     # cắt quỹ đạo ngay sau thời điểm quyết định: phải cho cùng k
            cut = R.run_online(mk(), tr[: full["decided_at"] + 1])
            assert cut["k"] == full["k"], name


# ---------------------------------------------------------------- test 4: assert chia tập
@pytest.fixture(scope="module")
def tiny_env():
    from ralm.data import Data
    cfg = get_cfg("fake")
    data = Data(cfg.as_profile(), torch.device("cpu"))
    return cfg, data


def test_set_assertions(tiny_env):
    from pre_stop.data import choose_forget, stage_split
    cfg, data = tiny_env
    r1, p = stage_split(data, 0)
    assert len(np.intersect1d(r1, p)) == 0 and np.array_equal(np.union1d(r1, p), data.d_idx_np)
    f = choose_forget("random-450", cfg, p, data.ytr_np, 0)
    s = Sets(data, cfg, r1, p, f, "random-450")
    assert len(s.f_np) == cfg.f_small and len(np.intersect1d(s.f_np, s.v_np)) == 0
    assert len(np.intersect1d(s.f_np, s.r2_np)) == 0 and np.isin(s.f_np, p).all()
    with pytest.raises(AssertionError):                                   # F chứa mẫu của V
        Sets(data, cfg, r1, p, np.concatenate([f[:-1], data.v_idx_np[:1]]), "random-450")
    with pytest.raises(AssertionError):                                   # F ⊄ P
        Sets(data, cfg, r1, p, np.concatenate([f[:-1], r1[:1]]), "random-450")
    with pytest.raises(AssertionError):                                   # sai kích thước
        Sets(data, cfg, r1, p, f[:-1], "random-450")
    with pytest.raises(AssertionError):                                   # R1 ∩ P ≠ ∅
        Sets(data, cfg, r1, np.concatenate([p, r1[:1]]), f, "random-450")


# ---------------------------------------------------------------- test 6: eval tất định
def test_eval_deterministic(tiny_env):
    from pre_stop.measure import measure
    from ralm.model import make_model
    cfg, data = tiny_env
    m = make_model(cfg.as_profile(), torch.device("cpu"))
    m.eval()
    idx = torch.from_numpy(data.d_idx_np[:200])
    a = measure(m, data, cfg, idx)[1]
    b = measure(m, data, cfg, idx)[1]
    assert torch.equal(a, b)


# ---------------------------------------------------------------- test 7: nhiễu sàn ν từ chối M < 3
def test_noise_floor_needs_three():
    with pytest.raises(ValueError):
        noise_floor(np.random.rand(2, 10))
    x = np.random.RandomState(1).rand(3, 10)
    assert noise_floor(x) > 0


def test_noise_floor_hand():
    x = np.array([[1.0, 2.0], [2.0, 2.0], [3.0, 5.0]])
    # m=0: |1-2.5|,|2-3.5| -> 1.5, 1.5 -> 1.5 ; m=1: |2-2|,|2-3.5| -> 0, 1.5 -> 0.75 ; m=2: |3-1.5|,|5-2| -> 1.5, 3 -> 2.25
    assert noise_floor(x) == pytest.approx((1.5 + 0.75 + 2.25) / 3)


# ---------------------------------------------------------------- test 2 + 5: pipeline nhỏ end-to-end
@pytest.fixture(scope="module")
def mini_run(tmp_path_factory):
    from pre_stop.run import main
    out = str(tmp_path_factory.mktemp("pre_out"))
    common = ["--profile", "fake", "--out", out, "--seeds", "0", "--ftypes", "random-450", "--methods", "ga"]
    assert main(["pre"] + common) == 0
    assert main(["cfg"] + common) == 0
    assert main(["traj"] + common) == 0
    return out


def _decisions(paths, cfg, seed=0, ft="random-450", method="ga"):
    from pre_stop.pre_info import load_P
    rid = paths.run_id(method, ft, seed, 0)
    tr = R.load_traj(paths, rid)
    P = load_P(paths, seed, ft)
    out = {}
    for mode in ("none", "V", "log"):
        ctx = R.Ctx(R.make_target(P, mode, tr[0]["L_V"]), tr[0]["L_R"], tr[0]["L_V"])
        for name, rule in R.online_rules(ctx).items():
            out[(mode, name)] = R.run_online(rule, tr)
    return out


def test_trajectory_runs_until_pre_rules_decide(mini_run):
    cfg, paths = get_cfg("fake"), Paths(mini_run)
    rid = paths.run_id("ga", "random-450", 0, 0)
    meta = json.loads(paths.meta(rid).read_text(encoding="utf-8"))
    assert meta["stop_reason"] in ("all_rules_decided", "max_epochs", "diverged")
    assert meta["n_snapshots"] == meta["n_epochs"] + 1 <= cfg.max_epochs + 1      # profile fake tự đặt trần 6 cho test
    if meta["stop_reason"] == "all_rules_decided":
        assert meta["n_rules_undecided"] == 0
        for key, res in _decisions(paths, cfg).items():
            if key[1] in ("R6-PRE-hit", "R7-PRE-J", "R8-PRE-sample", "R9-PRE-median-hit"):
                assert res["triggered"] and res["decided_at"] <= meta["n_epochs"]


def test_rules_independent_of_oracle(mini_run):
    cfg, paths = get_cfg("fake"), Paths(mini_run)
    before = _decisions(paths, cfg)
    shutil.rmtree(paths.root / "oracle")                                  # xoá thư mục oracle
    assert not (paths.root / "oracle").exists()
    after = _decisions(paths, cfg)
    assert before == after and len(before) > 0


def test_rules_module_has_no_oracle_import():
    import inspect
    import pre_stop.rules as m
    import pre_stop.unlearn as u
    for mod in (m, u):
        src = inspect.getsource(mod)
        code = "\n".join(l for l in src.splitlines() if l.lstrip().startswith(("import ", "from ")))
        assert "eval_oracle" not in code, mod.__name__


def test_P_frozen_before_stage2_and_hash(mini_run):
    from pre_stop.pre_info import load_P
    paths = Paths(mini_run)
    st2 = json.loads(paths.stage2_stats(0, "random-450").read_text(encoding="utf-8"))
    P = json.loads(paths.P_json(0, "random-450").read_text(encoding="utf-8"))
    assert P["t_frozen"] <= st2["t_stage2_start"]                         # đóng băng TRƯỚC bước cập nhật đầu
    load_P(paths, 0, "random-450")                                        # hash khớp
    arr = np.load(paths.P_loss(0, "random-450"))
    arr[0] += 1.0
    np.save(paths.P_loss(0, "random-450"), arr)                           # sửa trộm
    with pytest.raises(AssertionError):
        load_P(paths, 0, "random-450")


def test_test_metrics_sealed(mini_run):
    paths = Paths(mini_run)
    rid = paths.run_id("ga", "random-450", 0, 0)
    row = json.loads(paths.traj_jsonl(rid).read_text(encoding="utf-8").splitlines()[0])
    assert "L_T" not in row and "A_T" not in row                          # test nằm file riêng (§7)
    t = json.loads(paths.traj_test(rid).read_text(encoding="utf-8").splitlines()[0])
    assert t["post_selection"] is True



# ---------------------------------------------------------------- epoch theo code gốc (chỉ thị: đếm theo epoch)
def _epoch_env(tmp_path_factory_dir):
    from pre_stop.data import choose_forget, stage_split
    from ralm.data import Data
    cfg = get_cfg("fake")
    data = Data(cfg.as_profile(), torch.device("cpu"))
    r1, p = stage_split(data, 0)
    f = choose_forget("random-450", cfg, p, data.ytr_np, 0)
    return cfg, data, Sets(data, cfg, r1, p, f, "random-450")


def _fresh_model(cfg):
    from ralm.model import make_model
    return make_model(cfg.as_profile(), torch.device("cpu"))


def test_epoch_semantics_match_original_code():
    """GA: 1 epoch = 1 lượt qua F. FT, SalUn, SCRUB: đếm theo số batch của loader gốc."""
    import math
    from pre_stop import unlearn as U
    cfg, data, sets = _epoch_env(None)
    dev = torch.device("cpu")
    nF, nR = len(sets.f), len(sets.r)
    exp = {"ga": math.ceil(nF / 256), "ft": math.ceil(nR / 256),
           "salun": math.ceil(nF / 256) + math.ceil(nR / 256),
           "scrub": math.ceil(nR / 32)}                      # epoch 1 của SCRUB gốc còn có pha maximize trên F
    exp["scrub"] += math.ceil(nF / 64)
    for name, n in exp.items():
        um = U.METHOD_CLS[name](cfg, data, sets, _fresh_model(cfg), dev)
        um.epoch()
        assert um.n_updates == n, (name, um.n_updates, n)


def test_scrub_maximize_only_first_msteps_epochs():
    import math
    from pre_stop import unlearn as U
    cfg, data, sets = _epoch_env(None)
    um = U.Scrub(cfg, data, sets, _fresh_model(cfg), torch.device("cpu"))
    per = []
    lrs = []
    for _ in range(6):
        before = um.n_updates
        um.epoch()
        per.append(um.n_updates - before)
        lrs.append(um.opt.param_groups[0]["lr"])
    nr, nf = math.ceil(len(sets.r) / 32), math.ceil(len(sets.f) / 64)
    assert per == [nr + nf] * 3 + [nr] * 3                   # msteps = 3
    assert lrs[4] == pytest.approx(5e-4) and lrs[5] == pytest.approx(5e-5)   # gốc: epoch > 5 mới nhân 0.1


def test_salun_mask_is_top_half():
    from pre_stop import unlearn as U
    cfg, data, sets = _epoch_env(None)
    um = U.SalUn(cfg, data, sets, _fresh_model(cfg), torch.device("cpu"))
    tot = sum(v.numel() for v in um.mask.values())
    on = sum(float(v.sum()) for v in um.mask.values())
    assert 0.45 <= on / tot <= 0.55


def test_epoch_count_and_tarun_two_epochs():
    from pre_stop.unlearn import applicable, n_epochs_for
    cfg = get_cfg("fake")
    assert n_epochs_for(get_cfg("kaggle"), "ga") is None and n_epochs_for(cfg, "scrub") == 6    # kaggle: không trần
    assert n_epochs_for(get_cfg("kaggle", n_epochs=12), "ga") == 12
    assert n_epochs_for(get_cfg("kaggle"), "tarun") == 2
    assert applicable("tarun", "class-0") and not applicable("tarun", "random-450")
    assert applicable("ft", "random-450") and not applicable("ft", "class-0")


# ---------------------------------------------------------------- lưới lr (EXPLORATORY) cho ga, scrub
def test_lr_grid_scales_lr_and_keeps_original_schedule():
    from pre_stop import unlearn as U
    cfg, data, sets = _epoch_env(None)
    dev = torch.device("cpu")
    ga = U.GA(cfg, data, sets, _fresh_model(cfg), dev, lr_scale=0.1)
    assert ga.opt.param_groups[0]["lr"] == pytest.approx(0.001)
    assert U.GA(cfg, data, sets, _fresh_model(cfg), dev).opt.param_groups[0]["lr"] == pytest.approx(0.01)
    sc = U.Scrub(cfg, data, sets, _fresh_model(cfg), dev, lr_scale=0.1)
    lrs = []
    for _ in range(6):
        sc.epoch()
        lrs.append(sc.opt.param_groups[0]["lr"])
    assert all(l == pytest.approx(5e-5) for l in lrs)         # scrub@s (s != 1): lr KHÔNG ĐỔI, không giảm ×0.1 ở epoch 5, 8, 9
    s1 = U.Scrub(cfg, data, sets, _fresh_model(cfg), dev)
    for _ in range(6):
        s1.epoch()
    assert s1.opt.param_groups[0]["lr"] == pytest.approx(5e-5)    # scrub gốc (s = 1) vẫn giữ lịch ×0.1 sau epoch 5


def test_lr_grid_method_names():
    from pre_stop.unlearn import applicable, n_epochs_for, split_method
    assert split_method("ga") == ("ga", 1.0) and split_method("scrub@0.01") == ("scrub", 0.01)
    for bad in ("salun@0.1", "tarun@0.1", "ft@0.1", "xx"):
        with pytest.raises(ValueError):
            split_method(bad)
    assert applicable("scrub@0.1", "class-0") and n_epochs_for(get_cfg("kaggle"), "ga@0.1") is None


def test_plateau_is_opt_in_and_not_an_epoch_cap():
    from pre_stop.unlearn import plateau
    flat = [{"L_F": 0.5, "L_R": 0.3, "L_V": 0.6}] * 10
    assert plateau(flat, 10, 0.05) and not plateau(flat[:9], 10, 0.05)
    noisy = [{"L_F": 0.60 + (0.02 if i % 2 else -0.02), "L_R": 0.17, "L_V": 0.59} for i in range(10)]
    assert plateau(noisy, 10, 0.05)                            # nhiễu ±3% giữa các epoch KHÔNG cản việc dừng
    drifting = [{"L_F": 0.9 - 0.05 * i, "L_R": 0.3, "L_V": 0.6} for i in range(10)]
    assert not plateau(drifting, 10, 0.05)                     # L_F còn đi xuống thì chưa dừng
    assert get_cfg("kaggle").plateau_w is None            # mặc định TẮT


def test_lr_grid_trajectory_runs_and_reports_separately(tmp_path):
    """ga@0.1 chạy trên cùng θ0, ghi báo cáo vào results_<tag>/, không đụng results/. Pipeline riêng (không dùng chung
    mini_run vì một test khác cố ý sửa P trong đó)."""
    from pre_stop.run import main
    out = str(tmp_path)
    common = ["--profile", "fake", "--out", out, "--seeds", "0", "--ftypes", "random-450"]
    assert main(["pre"] + common) == 0 and main(["cfg"] + common) == 0
    assert main(["traj", "--lr_scales", "0.1"] + common) == 0
    paths = Paths(out)
    meta = json.loads(paths.meta(paths.run_id("ga@0.1", "random-450", 0, 0)).read_text(encoding="utf-8"))
    assert meta["base_method"] == "ga" and meta["lr_scale"] == 0.1 and meta["stop_reason"]
    assert main(["analyze", "--lr_scales", "0.1", "--tag", "lrgrid_test"] + common) == 0
    rep = Paths(out, results="results_lrgrid_test").results
    assert (rep / "report_auto.md").exists() and (rep / "compare_retrain.csv").exists()
    assert not (Paths(out).results / "report_auto.md").exists()          # báo cáo chính không bị ghi đè




# ---------------------------------------------------------------- giai đoạn 1: dừng sớm theo loss train
def test_early_stop_patience_and_best():
    from pre_stop.train import EarlyStop
    es = EarlyStop(5)
    losses = [1.0, 0.8, 0.6, 0.61, 0.62, 0.60, 0.63, 0.64, 0.7]
    out = [es.update(i, l) for i, l in enumerate(losses)]
    assert [o[0] for o in out][:3] == [True, True, True]            # 3 epoch đầu đều cải thiện
    assert es.best == 0.6 and es.best_ep == 2
    first_stop = next(i for i, o in enumerate(out) if o[1])
    assert first_stop == 7, first_stop                              # 5 epoch xấu liên tiếp: 3,4,5,6,7 (vòng train break tại đây)


def test_early_stop_only_when_lr_is_low():
    from pre_stop.train import early_stop_start
    from ralm.train import lr_at
    st = early_stop_start(300, 100, 5, 0.1)
    assert 235 <= st <= 245, st                                     # ~ epoch 240
    assert lr_at(st * 100, 300 * 100, 5 * 100, 0.1) < 0.01 <= lr_at((st - 1) * 100, 300 * 100, 5 * 100, 0.1)


def test_early_stop_ignores_warmup():
    from pre_stop.train import EarlyStop
    es = EarlyStop(2, start=4)
    res = [es.update(i, l)[1] for i, l in enumerate([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])]
    assert res == [False, False, False, False, False, True]         # epoch 0..3 không bị tính là xấu


def test_stage1_runs_restores_best_and_records(tmp_path):
    from pre_stop.train import train_stage1, stage1_signature
    from pre_stop.data import stage_split
    from ralm.data import Data
    from dataclasses import replace
    cfg = replace(get_cfg("fake"), e1=40, patience1=2, warm1=1)
    # Cùng thiết bị với pipeline thật (run.py). Ép CPU trên Kaggle đã segfault ở backward của _fit.
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Data(cfg.as_profile(), dev)
    r1, _ = stage_split(data, 0)
    info = {}
    train_stage1(cfg, data, torch.from_numpy(r1).to(dev), 0, dev, lambda *a, **k: None, info)
    assert 1 <= info["best_epoch"] <= info["epochs_run"] <= 40
    assert info["stopped_early"] == (info["epochs_run"] < 40)
    assert stage1_signature(cfg)["patience1"] == 2


def test_old_stage1_cache_is_rejected(mini_run):
    from dataclasses import replace
    from pre_stop.train import check_stage1
    cfg, paths = get_cfg("fake"), Paths(mini_run)
    check_stage1(cfg, paths, [0])                                   # cùng cấu hình: được
    with pytest.raises(SystemExit):
        check_stage1(replace(cfg, patience1=cfg.patience1 + 1), paths, [0])


def test_report_compares_with_retrain(tmp_path):
    from pre_stop import analyze as A
    from pre_stop.run import main
    common = ["--profile", "fake", "--out", str(tmp_path), "--seeds", "0", "--ftypes", "random-450", "--methods", "ga"]
    for cmd in ("pre", "cfg", "traj"):
        assert main([cmd] + common) == 0
    cfg, paths = get_cfg("fake"), Paths(str(tmp_path))
    A.analyze_all(cfg, paths, [0], ["random-450"], ["ga"])
    txt = (paths.results / "report_auto.md").read_text(encoding="utf-8")
    assert "RETRAIN" in txt and "2b. So sánh với mô hình retrain" in txt
    import pandas as pd
    c = pd.read_csv(paths.results / "compare_retrain.csv")
    for m in A.CMP:
        assert m in c.columns and m + "_iqr" in c.columns
    r7 = c[c.rule == "R7-PRE-J"].iloc[0]
    assert abs(r7["gap_A_T"] - (r7["A_T"] - c[c.rule.str.startswith("RETRAIN")].iloc[0]["A_T"])) < 1e-9


def test_missing_pre_is_reported_clearly(tmp_path):
    from pre_stop.run import main
    with pytest.raises(SystemExit, match='thiếu θ_pre'):
        main(['cfg', '--profile', 'fake', '--out', str(tmp_path), '--seeds', '1', '--ftypes', 'random-450'])

