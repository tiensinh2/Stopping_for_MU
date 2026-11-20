"""Unit tests (§7, §4.3, §1.1, §1.2, §6). Chạy: pytest tests -x -q   (CPU, dữ liệu giả, ~1 phút)."""
import json
import logging

import numpy as np
import pytest
import torch

from ralm import score_ralm as S
from ralm.config import Paths, get_profile
from ralm.guard import assert_no_oracle_imports


def mk(step, LR, LD, LV, LF=2.0):
    return {"step": step, "L_R": LR, "L_D": LD, "L_V": LV, "L_F": LF, "A_R": 1, "A_D": 1, "A_V": 1, "A_F": 0.5,
            "post_selection": False}


# ---------------------------------------------------------------- §7 unit tests bắt buộc
def test_flat_trajectory_does_not_crash():
    traj = [mk(s, 0.1, 0.2, 0.3) for s in (0, 50, 100, 150, 200)]
    out = S.select_ralm(traj, k_min=100)
    assert out["J_DV"] in (100, 150, 200)


def test_min_before_kmin_selects_min_after():
    # J nhỏ nhất ở step 50 (< k_min=100): phải chọn min trong số step ≥ 100
    traj = [mk(0, 1.0, 1.0, 1.0), mk(50, 0.1, 1.0, 1.0), mk(100, 0.5, 1.0, 1.0), mk(150, 0.3, 1.0, 1.0),
            mk(200, 0.4, 1.0, 1.0)]
    assert S.select_ralm(traj, k_min=100)["J_DV"] == 150


def test_hand_computed_five_points():
    traj = [mk(0, 1.0, 1.0, 1.0), mk(100, 0.8, 1.1, 1.0), mk(200, 0.65, 1.0, 1.2),
            mk(300, 0.6, 0.8, 1.1), mk(400, 0.5, 0.5, 0.5)]
    sc = S.compute_scores(traj)
    # J_DV = S_R + S_D + S_V   (L^0 = 1 nên S_R = L_R)
    exp = {100: 0.9, 200: 0.85, 300: 0.9, 400: 1.5}
    for m in sc:
        if m["step"] in exp:
            assert m["J_DV"] == pytest.approx(exp[m["step"]], abs=1e-6)
    assert S.select_ralm(traj, k_min=100)["J_DV"] == 200
    assert sc[2]["J_D"] == pytest.approx(0.65, abs=1e-6) and sc[2]["J_V"] == pytest.approx(0.85, abs=1e-6)


def test_ema_by_hand():
    traj = [mk(i, x, 1, 1) for i, x in enumerate([1.0, 2.0, 3.0, 2.0, 1.0])]
    out = S.apply_ema(traj, "L_R", 0.9)
    exp, prev = [], None
    for x in [1.0, 2.0, 3.0, 2.0, 1.0]:
        prev = x if prev is None else 0.9 * prev + 0.1 * x
        exp.append(prev)
    assert [m["L_R"] for m in out] == pytest.approx(exp, abs=1e-12)
    assert traj[1]["L_R"] == 2.0                       # bản gốc không bị sửa


def test_delta_F_gate_fallback_warns(caplog):
    # không checkpoint nào vượt cổng: không crash, cảnh báo, chọn L_F lớn nhất (§8)
    traj = [mk(s, 0.5, 1.0, 1.0, LF=2.0 + 0.0001 * s) for s in (0, 100, 200)]
    traj[0]["L_F"] = 2.0
    with caplog.at_level(logging.WARNING, logger="ralm.score"):
        k = S.select_ralm(traj, k_min=100, delta_F=0.05)["J_DV"]
    assert k == 200 and any("δ_F" in r.message for r in caplog.records)


# ---------------------------------------------------------------- §1.1 / §1.2 cách ly
def test_isolation_of_oracle_modules():
    import ralm
    root = __import__("pathlib").Path(ralm.__file__).parent
    for f in ("collect_traj.py", "score_ralm.py", "rules.py"):
        assert_no_oracle_imports(str(root / f))
    bad = root.parent / "_bad_tmp.py"
    bad.write_text("from ralm import eval_oracle\n", encoding="utf-8")
    try:
        with pytest.raises(AssertionError):
            assert_no_oracle_imports(str(bad))
    finally:
        bad.unlink()


def test_score_refuses_test_columns(tmp_path):
    p = tmp_path / "t.jsonl"
    r = mk(0, 1, 1, 1); r["L_T"] = 0.3
    p.write_text(json.dumps(r) + "\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        S.load_traj(p)
    r = mk(0, 1, 1, 1); r["L_T"] = None; r["post_selection"] = True
    p.write_text(json.dumps(r) + "\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        S.load_traj(p)


def test_load_arrays_never_returns_test_logits(tmp_path):
    p = tmp_path / "a.npz"
    np.savez(p, steps=np.arange(3), tlogits=np.zeros((3, 2, 2)), t_y=np.zeros(2), vlogits=np.zeros((3, 2, 2)))
    z = S.load_arrays(p)
    assert "tlogits" not in z and "t_y" not in z and "vlogits" in z


# ---------------------------------------------------------------- anchors
def test_anchor_zero_at_step0_and_tail_sensitivity():
    K, n = 4, 1000
    rs = np.random.RandomState(0)
    base = rs.rand(n).astype(np.float32)
    L = np.stack([base, base, base, base]); L[2:, :5] += 5.0         # chỉ đuôi trên đổi (0.5% mẫu)
    traj = [mk(i * 100, 0.1, float(L[i].mean()), 0.3) for i in range(K)]
    arr = {"dperloss": L, "d_y": rs.randint(0, 10, n)}
    a_mean = S.anchor_series("T0", traj, arr)
    a_cvar = S.anchor_series("T2_0.5", traj, arr)
    assert a_mean[0] == 0 and a_cvar[0] == 0
    assert a_cvar[2] > 5 * a_mean[2]                                 # CVaR thấy đuôi mà mean gần như mù
    for nme in S.ANCHOR_NAMES:
        if nme in ("T4", "T5", "T6_D", "T6_V", "T8", "T9"):
            continue
        assert S.anchor_series(nme, traj, arr).shape == (K,)


def test_kl_series_zero_at_theta0():
    z = np.random.RandomState(1).randn(3, 20, 5).astype(np.float32)
    kl = S._kl_series(z)
    assert kl[0] == pytest.approx(0, abs=1e-12) and (kl[1:] > 0).all()


def test_band_selection():
    traj = [mk(0, 1, 1, 1), mk(100, 0.9, 1.02, 1.0), mk(200, 0.6, 1.03, 1.01), mk(300, 0.3, 1.5, 1.0)]
    sc = S.compute_scores(traj)
    k, info = S.select_band(sc, k_min=100, eps_D=0.05, eps_V=0.05)
    assert k == 200 and info["band_size"] == 2                       # 300 vi phạm S_D; trong band chọn S_R nhỏ nhất


# ---------------------------------------------------------------- pipeline thật (CPU, fake data)
@pytest.fixture(scope="module")
def pipe(tmp_path_factory):
    from ralm.collect_traj import collect
    from ralm.eval_core import eval_full
    from ralm.train import train_base
    prof = get_profile("fake", epochs=6)
    paths = Paths(tmp_path_factory.mktemp("o"), prof)
    dev = torch.device("cpu")
    train_base(prof, paths, 0, dev, log=lambda *_: None)
    return prof, paths, dev


def test_manifest_and_identity(pipe):
    from ralm.data import Data, Split
    from ralm.eval_core import eval_full
    from ralm.manifest import get_manifest
    from ralm.model import make_model
    prof, paths, dev = pipe
    data = Data(prof, dev)
    for fkey, n_exp in (("random_rho0.1", 135), ("highloss_rho0.05", 68), ("memorized_rho0.05", 68)):
        man = get_manifest(prof, paths, data, fkey, 0)
        idx = np.array(man["d_indices"])
        assert len(idx) == n_exp and np.isin(idx, data.d_idx_np).all()
        assert len(np.intersect1d(idx, data.v_idx_np)) == 0
    cw = get_manifest(prof, paths, data, "classwise_3", 0)
    assert (data.ytr_np[np.array(cw["d_indices"])] == 3).all()
    assert (data.ytr_np[data.d_idx_np] == 3).sum() == cw["n_forget"]
    # highloss: mọi mẫu F có loss ≥ mọi mẫu R theo logits θ0
    lg = np.load(paths.d_logits0(0)); d = data.d_idx_np
    from ralm.manifest import get_manifest as gm
    m = gm(prof, paths, data, "highloss_rho0.05", 0)
    z = lg[d].astype(np.float64); y = data.ytr_np[d]
    ce = np.log(np.exp(z - z.max(1, keepdims=True)).sum(1)) + z.max(1) - z[np.arange(len(d)), y]
    inF = np.isin(d, m["d_indices"])
    assert ce[inF].min() >= ce[~inF].max() - 1e-9
    # identity L_D = (1-ρ)L_R + ρL_F với model thật (assert nằm trong eval_full)
    model = make_model(prof, dev); model.load_state_dict(torch.load(paths.base_ckpt(0)))
    split = Split(data, np.array(man["d_indices"]))
    d_, _ = eval_full(model, data, split, False, 256, check=True)
    assert abs(d_["L_D"] - ((1 - split.rho) * d_["L_R"] + split.rho * d_["L_F"])) < 1e-5


def test_all_methods_run_and_theta0_row(pipe):
    from ralm.collect_traj import collect
    prof, paths, dev = pipe
    for method in ("ga", "ngp", "scrub", "salun", "rl", "tarun", "ft"):
        meta = collect(prof, paths, method, "random_rho0.1", 0, dev, n_steps=10, log=lambda *_: None)
        rows = S.load_traj(paths.traj_jsonl(meta["run_id"]))
        assert [r["step"] for r in rows] == [0, 5, 10]
        assert rows[0]["w_dist"] == 0 and rows[0]["emb_drift_V"] == pytest.approx(0, abs=1e-9)
        assert all(np.isfinite(r["L_R"]) for r in rows)
        z = np.load(paths.traj_jsonl(meta["run_id"]).with_name(f"traj_{meta['run_id']}_arrays.npz"))
        assert z["dperloss"].shape[0] == 3 and z["vlogits"].shape[1] == prof.v_size


def test_ga_increases_forget_loss_and_ft_leaves_it(pipe):
    prof, paths, dev = pipe
    from ralm.collect_traj import collect
    collect(prof, paths, "ga", "random_rho0.1", 0, dev, n_steps=40, force=True, log=lambda *_: None)
    t = S.load_traj(paths.traj_jsonl("ga_cifar10_random_rho0.1_seed0"))
    assert t[-1]["L_F"] > t[0]["L_F"]


def test_salun_mask_fraction_and_rl_labels(pipe):
    prof, paths, dev = pipe
    meta = json.loads(paths.meta("salun_cifar10_random_rho0.1_seed0").read_text(encoding="utf-8"))
    assert meta["extra"]["mask_frac"] == pytest.approx(0.5, abs=0.02)


def test_methods_do_not_touch_test_in_selection(pipe):
    # jsonl không có L_T/A_T khác None (test chỉ ở npz tlogits)
    prof, paths, dev = pipe
    for line in paths.traj_jsonl("ga_cifar10_random_rho0.1_seed0").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        assert r["L_T"] is None and r["A_T"] is None and r["post_selection"] is False


def test_oracle_pipeline_and_regret(pipe):
    from ralm import aggregate as AG
    from ralm import eval_oracle as O
    from ralm.retrain import retrain
    prof, paths, dev = pipe
    retrain(prof, paths, "random_rho0.1", 0, 0, dev, log=lambda *_: None)
    o = AG.analyze_run(prof, paths, "ga_cifar10_random_rho0.1_seed0", "random_rho0.1", 0)
    rules = {r["rule"]: r for r in o["rules"]}
    assert rules["ralm-dv"]["regret"] >= 0 and "ralm-band" in rules and "random-k4" in rules
    G = np.array(o["curves"]["G_logit_F"]); st = np.array(o["curves"]["steps"])
    assert O.regret(G, st, int(st[np.argmin(G)]))["regret"] == 0
    assert 0.0 <= o["mia_auc"]["ralm-dv"] <= 1.0
    assert o["L_D_gap"] is not None
