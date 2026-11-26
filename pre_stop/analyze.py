"""Phân tích (kịch bản §9, §12, §15, §18): metric chuẩn hoá theo ν, regret, rollback, R14/R15, tiêu chí thành công.

File này là nơi duy nhất ghép luật (rules.py) với oracle (eval_oracle.py). Luật chạy trước, oracle chỉ chấm điểm sau.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import rules as R
from .config import FTYPES, MAIN_FTYPE, P_HIT, GATE1_RATIO, Cfg
from .eval_oracle import Oracle, auc_conf, conf_of, oracle_checkpoint
from .pre_info import load_P

PRE_ONLINE = ("R6-PRE-hit", "R7-PRE-J", "R8-PRE-sample", "R9-PRE-median-hit")
PRE_GLOBAL = ("R7g-PRE-J-global", "R8g-PRE-sample-global")      # argmin thuần trên toàn quỹ đạo, offline
PRE_RULES = PRE_ONLINE + PRE_GLOBAL
FWD = {  # số ảnh phải forward thêm mỗi snapshot khi chạy online (R_sub=5000, V=5000, |F| tuỳ cấu hình)
    "R3-retain-only": "R", "R4-val-only": "V", "R5-heldout-hit": "FV", "R6-PRE-hit": "FR", "R7-PRE-J": "FR",
    "R8-PRE-sample": "FR", "R9-PRE-median-hit": "FR", "R12-acc-hit": "FV"}


def _iqr(x):
    x = np.asarray(x, dtype=float)
    return float(np.percentile(x, 75) - np.percentile(x, 25)) if len(x) else float("nan")


def read_test(paths, rid):
    rows = [json.loads(l) for l in paths.traj_test(rid).read_text(encoding="utf-8").splitlines() if l.strip()]
    return np.array([r["A_T"] for r in rows]), np.array([r["L_T"] for r in rows])


# các chỉ số đem so với mô hình retrain (oracle θ_R2); R = R_sub (5000 mẫu), V = tập validation, T = test
CMP = ("L_F", "A_F", "L_R", "A_R", "L_V", "A_V", "L_T", "A_T")


def oracle_metrics(orc) -> dict:
    """Trung bình và độ lệch chuẩn của M oracle cho từng chỉ số trong CMP."""
    out = {}
    for m in CMP:
        v = getattr(orc, m)
        out[m] = float(np.mean(v))
        out[m + "_sd"] = float(np.std(v, ddof=1))
    return out


def gate1_row(cfg, paths, seed, ftype, L_R0=None, L_V0=None) -> dict:
    """e_pre/ν, e_pre,agg, e_R và (khoảng cách, utility) của rollback, tính SAU θ0 và oracle, TRƯỚC unlearning."""
    orc = Oracle(paths, seed, ftype)
    P = load_P(paths, seed, ftype)
    st2 = json.loads(paths.stage2_stats(seed, ftype).read_text(encoding="utf-8"))
    pre = json.loads(paths.pre_stats(seed).read_text(encoding="utf-8"))
    cV = (L_V0 / P["L_V_pre"]) if L_V0 is not None else float("nan")
    t_none = P["loss_F_pre"]
    row = {"seed": seed, "ftype": ftype, "w_F": st2["w_F"], "gate_F_learned": st2["gate_passed"],
           "nu": orc.nu, "L_F_pre": P["L_F_pre"], "L_F_oracle": float(orc.L_F.mean()),
           "e_pre_over_nu_none": orc.e_pre(t_none), "e_pre_over_nu_V": orc.e_pre(cV * t_none),
           "e_pre_agg_none": orc.e_pre_agg(t_none), "e_pre_agg_V": orc.e_pre_agg(cV * t_none),
           "c_V": cV, "d_hat_rollback": orc.d_hat(t_none), "g_T_rollback": orc.g_T(pre["A_T_pre"])}
    if L_R0 is not None:
        row["e_R"] = orc.e_R(L_R0)
    return row


def analyze_run(cfg: Cfg, paths, seed, ftype, method, u=0) -> list[dict]:
    rid = paths.run_id(method, ftype, seed, u)
    traj = R.load_traj(paths, rid)
    A_T, L_T = read_test(paths, rid)                  # post-selection: chỉ analyze đọc
    orc, P = Oracle(paths, seed, ftype), load_P(paths, seed, ftype)
    pre = json.loads(paths.pre_stats(seed).read_text(encoding="utf-8"))
    n = len(traj)
    L_R0, L_V0 = traj[0]["L_R"], traj[0]["L_V"]
    z = np.load(paths.traj_arrays(rid))
    d_hat = np.array([orc.d_hat(sn["loss_F"]) for sn in traj])
    g_T = np.array([orc.g_T(a) for a in A_T])
    k_star, flag_noutil = oracle_checkpoint(d_hat, g_T)
    kl = np.array([orc.g_kl(z["logits_F"][i]) for i in range(n)])
    conf_v = [conf_of(z["logits_V"][i]) for i in range(n)]
    conf_f = [conf_of(z["logits_F"][i]) for i in range(n)]
    auc_k = np.array([auc_conf(conf_f[i], conf_v[i]) for i in range(n)])
    auc_or = float(np.mean([auc_conf(conf_of(orc.logits_F[m]), conf_of(orc.logits_V[m].astype(np.float64)))
                            for m in range(orc.M)]))
    nF, nV, nR = len(traj[0]["loss_F"]), z["loss_V"].shape[1], 5000
    orc_m = oracle_metrics(orc)
    # số snapshot k>=1 qua rào G_R và G_V (rào chỉ phụ thuộc L_R(0), L_V(0), không phụ thuộc đích). k=0 luôn qua rào nên cờ
    # no_admissible không bao giờ bật khi luật rơi về k=0; chỉ số này mới cho biết luật có điểm unlearn nào để chọn hay không.
    gate_ctx = R.Ctx(np.ones(nF), L_R0, L_V0)
    n_adm_after0 = int(sum(gate_ctx.admissible(sn) for sn in traj[1:]))
    recs = []

    def add(rule, target, res, extra=None):
        k = res["k"]
        row = {"seed": seed, "ftype": ftype, "method": method, "u": u, "rule": rule, "target": target,
               "k": k, "epoch": traj[k]["epoch"], "decided_at_epoch": traj[res["decided_at"]]["epoch"],
               "pick_k0": int(k == 0), "triggered": int(res["triggered"]),
               "no_admissible": int(res["no_admissible"]), "flag_no_util_k_star": int(flag_noutil),
               "n_adm_after0": n_adm_after0, "n_snap_after0": n - 1,
               "d_hat": d_hat[k], "d_hat_k_star": d_hat[k_star], "k_star": k_star,
               "epoch_star": traj[k_star]["epoch"], "regret": d_hat[k] - d_hat[k_star],
               "delta0": d_hat[0] - d_hat[k], "g_T": g_T[k], "G_KL": kl[k],
               "auc_gap": abs(auc_k[k] - auc_or), "L_F": traj[k]["L_F"], "A_F": traj[k]["A_F"],
               "L_R": traj[k]["L_R"], "A_R": traj[k]["A_R"], "L_V": traj[k]["L_V"], "A_V": traj[k]["A_V"],
               "A_T": A_T[k], "L_T": L_T[k]}
        # GapAvg (kịch bản §19.2), đơn vị điểm phần trăm: trung bình |lệch so với retrain| của FA, RA, TA và AUC(MIA);
        # GapAvg-TA-MIA chỉ lấy TA và MIA. Luật PRE không dùng chúng để chọn k.
        d_fa, d_ra = abs(row["A_F"] - orc_m["A_F"]), abs(row["A_R"] - orc_m["A_R"])
        d_ta, d_mia = abs(row["A_T"] - orc_m["A_T"]), row["auc_gap"]
        row["gapavg"] = 100.0 * float(np.mean([d_fa, d_ra, d_ta, d_mia]))
        row["gapavg_ta_mia"] = 100.0 * float(np.mean([d_ta, d_mia]))
        code = FWD.get(rule, "")
        row["fwd_images_per_snapshot"] = ("F" in code) * nF + ("R" in code) * nR + ("V" in code) * nV
        if extra:
            row.update(extra)
        for m in CMP:                                  # chênh lệch có dấu so với trung bình oracle (retrain)
            row["retrain_" + m] = orc_m[m]
            row["gap_" + m] = row[m] - orc_m[m]
        recs.append(row)

    add("R0-do-nothing", "-", R.r0_do_nothing(traj))
    # R1 rollback: không phải luật trên quỹ đạo; k_star/regret tính cùng thang để so sánh
    rb = {"k": 0, "decided_at": 0, "triggered": True, "no_admissible": False}
    # rollback = chính θ_pre. Nếu đã chạy `prefull` thì có đủ L/A trên F, R_sub, V, test và logits -> G_KL, auc_gap, gapavg;
    # nếu chưa thì các chỉ số đó là NaN (không lấy nhầm giá trị của R0).
    nan = float("nan")
    r1 = {"d_hat": orc.d_hat(P["loss_F_pre"]), "g_T": orc.g_T(pre["A_T_pre"]),
          "regret": orc.d_hat(P["loss_F_pre"]) - d_hat[k_star], "delta0": d_hat[0] - orc.d_hat(P["loss_F_pre"]),
          "epoch": -1, "k": -1, "pick_k0": 0,
          "L_F": P["L_F_pre"], "A_F": P["A_F_pre"], "L_V": P["L_V_pre"], "A_V": nan,
          "L_R": nan, "A_R": nan, "A_T": pre["A_T_pre"], "L_T": pre["L_T_pre"],
          "gapavg": nan, "gapavg_ta_mia": nan, "G_KL": nan, "auc_gap": nan}
    pf = paths.pre_full(seed, ftype)
    if pf.exists():
        zp = np.load(pf)
        r1.update({m: float(zp[m]) for m in ("L_R", "A_R", "A_V")})
        r1["G_KL"] = orc.g_kl(zp["logits_F"])
        r1["auc_gap"] = abs(auc_conf(conf_of(zp["logits_F"]), conf_of(zp["logits_V"].astype(np.float64))) - auc_or)
        d_ = [abs(r1["A_F"] - orc_m["A_F"]), abs(r1["A_R"] - orc_m["A_R"]), abs(r1["A_T"] - orc_m["A_T"]), r1["auc_gap"]]
        r1["gapavg"] = 100.0 * float(np.mean(d_))
        r1["gapavg_ta_mia"] = 100.0 * float(np.mean(d_[2:]))
    add("R1-rollback", "-", rb, r1)
    for K in cfg.fixed_k:
        add(f"R2-fixed-{K}", "-", R.r2_fixed(traj, K))
    ctx0 = R.Ctx(R.make_target(P, "none", L_V0), L_R0, L_V0)
    for name, rule in R.online_rules(ctx0).items():
        if name in PRE_RULES:
            continue
        add(name, "-", R.run_online(rule, traj))
    add("R10-pareto-knee", "-", R.r10_pareto_knee(traj, ctx0))
    add("R11-scrubR-rewind", "-", R.r11_rewind(traj, ctx0))
    for mode in ("none", "V", "log"):
        ctx = R.Ctx(R.make_target(P, mode, L_V0), L_R0, L_V0)
        for name, rule in R.online_rules(ctx).items():
            if name in PRE_ONLINE:
                add(name, mode, R.run_online(rule, traj))
        add("R7g-PRE-J-global", mode, R.argmin_global(traj, ctx, R.J7))
        add("R8g-PRE-sample-global", mode, R.argmin_global(traj, ctx, R.J8))
    # R14 (oracle-aggregate-hit): trần của luật đích tổng hợp; chỉ để đo
    ctx14 = R.Ctx(np.full(nF, float(orc.L_F.mean())), L_R0, L_V0)
    add("R14-oracle-agg-hit", "oracle",
        R.run_online(R.FirstHit(ctx14, lambda sn, c: sn["L_F"] >= c.s, "R14"), traj))
    # đường cong để vẽ / R15
    curve = {"seed": seed, "ftype": ftype, "method": method, "u": u, "epochs": [sn["epoch"] for sn in traj],
             "d_hat": d_hat.tolist(), "g_T": g_T.tolist(), "k_star": k_star}
    return recs, curve


def add_r15(df: pd.DataFrame, curves: list[dict], cfg: Cfg) -> pd.DataFrame:
    """R15 oracle-tuned fixed-K, leave-one-seed-out theo (method, ftype) (§18.2). Cần >= 2 seed."""
    extra = []
    by = {}
    for c in curves:
        by.setdefault((c["method"], c["ftype"]), []).append(c)
    for (method, ftype), cs in by.items():
        seeds = sorted({c["seed"] for c in cs})
        if len(seeds) < 2:
            continue
        for s in seeds:
            others = [c for c in cs if c["seed"] != s]
            best = None
            for K in cfg.fixed_k:
                vals = []
                for c in others:
                    st = c["epochs"]
                    i = st.index(K) if K in st else len(st) - 1
                    vals.append(c["d_hat"][i])
                m = float(np.median(vals))
                if best is None or m < best[0]:
                    best = (m, K)
            row_src = df[(df.rule == f"R2-fixed-{best[1]}") & (df.seed == s) & (df.method == method)
                         & (df.ftype == ftype)]
            for _, r in row_src.iterrows():
                r = r.copy()
                r["rule"], r["target"] = "R15-oracle-tuned-fixedK", "oracle"
                extra.append(r)
    return pd.concat([df, pd.DataFrame(extra)], ignore_index=True) if extra else df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Median + IQR theo (ftype, method, rule, target) (§9.6, §15)."""
    g = df.groupby(["ftype", "method", "rule", "target"], sort=False)
    out = g.agg(n=("regret", "size"), regret_med=("regret", "median"), regret_iqr=("regret", _iqr),
                epoch_med=("epoch", "median"), pct_k0=("pick_k0", "mean"), delta0_med=("delta0", "median"),
                gT_med=("g_T", "median"), gKL_med=("G_KL", "median"), auc_gap_med=("auc_gap", "median"),
                gapavg_med=("gapavg", "median"), gapavg_ta_mia_med=("gapavg_ta_mia", "median"),
                A_F_med=("A_F", "median"), no_adm=("no_admissible", "mean"),
                fwd=("fwd_images_per_snapshot", "max")).reset_index()
    out["pct_k0"] *= 100
    out["no_adm"] *= 100
    return out


def compare_retrain(df: pd.DataFrame, rules=None) -> pd.DataFrame:
    """Median (và IQR) qua seed của 8 chỉ số tại điểm dừng, kèm dòng retrain = trung bình M oracle.

    Mỗi (ftype, method) một nhóm; cột gap_* là chênh lệch có dấu so với retrain (median qua seed).
    """
    rows = []
    for (ft, me), g in df.groupby(["ftype", "method"], sort=False):
        r = {"ftype": ft, "method": me, "rule": "RETRAIN (oracle, trung bình M=3)", "target": "-", "n": g.seed.nunique()}
        for m in CMP:
            r[m] = float(g.groupby("seed")["retrain_" + m].first().median())
            r[m + "_iqr"] = _iqr(g.groupby("seed")["retrain_" + m].first())
        rows.append(r)
        for (rule, tg), h in g.groupby(["rule", "target"], sort=False):
            if rules is not None and rule not in rules:
                continue
            r = {"ftype": ft, "method": me, "rule": rule, "target": tg, "n": len(h)}
            for m in CMP:
                r[m] = float(h[m].median())
                r[m + "_iqr"] = _iqr(h[m].dropna())
                r["gap_" + m] = float(h["gap_" + m].median())
            for m in ("g_T", "G_KL", "auc_gap", "gapavg", "gapavg_ta_mia"):
                r[m] = float(h[m].median())
                r[m + "_iqr"] = _iqr(h[m].dropna())
            rows.append(r)
    return pd.DataFrame(rows)


def sign_test(df: pd.DataFrame, a: str, b: str) -> dict:
    """Kiểm định dấu: tỉ lệ cấu hình mà luật a có regret nhỏ hơn b (§18.4)."""
    from math import comb
    key = ["seed", "ftype", "method", "u"]
    A = df[df.rule == a].groupby(key).regret.min()
    B = df[df.rule == b].groupby(key).regret.min()
    j = pd.concat([A, B], axis=1, keys=["a", "b"]).dropna()
    w, l = int((j.a < j.b).sum()), int((j.a > j.b).sum())
    n = w + l
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(0, min(w, l) + 1)) / 2 ** n)
    return {"a": a, "b": b, "wins": w, "losses": l, "ties": len(j) - n, "p": p}


def criteria(df: pd.DataFrame, gate: pd.DataFrame) -> pd.DataFrame:
    """§12.1 cho từng biến thể PRE. Không chọn 'biến thể tốt nhất' sau khi xem kết quả: báo đủ cả 8 biến thể."""
    rows = []
    c1 = gate[gate.ftype == MAIN_FTYPE]
    prem = {m: float(c1[f"e_pre_over_nu_{m}"].median()) if len(c1) else float("nan") for m in ("none", "V")}
    for rule in PRE_RULES:
        for mode in ("none", "V"):
            sub = df[(df.rule == rule) & (df.target == mode)]
            cell = {}
            for (ft, me), g in sub.groupby(["ftype", "method"]):
                b = df[(df.ftype == ft) & (df.method == me)]
                med = lambda r: float(b[b.rule == r].regret.median())            # noqa: E731
                fixed = [float(b[b.rule == r].regret.median()) for r in b.rule.unique() if r.startswith("R2-fixed")]
                rg = float(g.regret.median())
                cell[(ft, me)] = {"c2": rg <= 2 and rg < min(med("R0-do-nothing"), med("R3-retain-only"),
                                                             med("R4-val-only"), float(np.median(fixed))),
                                  "c3": float(g.g_T.median()) <= 1.0,
                                  "c4": (float(g.g_T.median()) <= float(b[b.rule == "R1-rollback"].g_T.median()) - 1.0
                                         and float(g.d_hat.median()) <= float(b[b.rule == "R1-rollback"].d_hat.median()) + 2)}
            n_ok = sum(1 for (ft, me), c in cell.items() if c["c2"] and c["c3"])
            fts = {ft for ft, _ in cell}
            ok_ft = {ft for ft in fts if all(cell[(ft, me)]["c2"] and cell[(ft, me)]["c3"]
                                              for me in ("ga", "scrub") if (ft, me) in cell)}
            rows.append({"rule": rule, "target": mode, "c1_premise_med_e_pre_over_nu": prem[mode],
                         "c1_ok": prem[mode] <= 5, "c2_cells": sum(c["c2"] for c in cell.values()),
                         "c3_cells": sum(c["c3"] for c in cell.values()), "c4_cells": sum(c["c4"] for c in cell.values()),
                         "cells": len(cell), "c5_ftypes_ok": len(ok_ft), "c5_ok": len(ok_ft) >= 3})
    return pd.DataFrame(rows)


def reject_checks(df: pd.DataFrame, gate: pd.DataFrame) -> dict:
    """§12.2: điều kiện bác bỏ."""
    pre = df[df.rule.isin(PRE_RULES) & df.target.isin(["none", "V"])]
    r0 = df[df.rule == "R0-do-nothing"].regret.median()
    med_by = pre.groupby(["rule", "target"]).regret.median()
    c1 = gate[gate.ftype == MAIN_FTYPE]
    return {"premise_fail_main": bool(len(c1) and min(c1.e_pre_over_nu_none.median(), c1.e_pre_over_nu_V.median()) > GATE1_RATIO),
            "frac_pre_k0": float(pre.pick_k0.mean()) if len(pre) else float("nan"),
            "reject_k0_ge_50pct": bool(len(pre) and pre.pick_k0.mean() >= 0.5),
            "no_pre_variant_beats_R0": bool(len(med_by) and (med_by >= r0).all()),
            "rollback_wins_everywhere": bool(
                len(pre) and all(float(pre[(pre.ftype == ft) & (pre.method == me)].g_T.median())
                                 > float(df[(df.rule == "R1-rollback") & (df.ftype == ft) & (df.method == me)].g_T.median()) - 1.0
                                 for ft, me in pre[["ftype", "method"]].drop_duplicates().itertuples(index=False)))}


def analyze_all(cfg: Cfg, paths, seeds, ftypes, methods, us=(0,), log=print) -> dict:
    recs, curves, gates = [], [], []
    for s in seeds:
        for ft in ftypes:
            try:
                first = paths.run_id(methods[0], ft, s, us[0])
                tr0 = R.load_traj(paths, first)
                gates.append(gate1_row(cfg, paths, s, ft, tr0[0]["L_R"], tr0[0]["L_V"]))
            except FileNotFoundError:
                log(f"[analyze] thiếu dữ liệu cho gate {ft} seed {s}, bỏ qua")
            for me in methods:
                for u in us:
                    try:
                        r, c = analyze_run(cfg, paths, s, ft, me, u)
                    except FileNotFoundError as e:
                        log(f"[analyze] thiếu {me}/{ft}/seed{s}/u{u}: {e}")
                        continue
                    recs += r
                    curves.append(c)
    df = pd.DataFrame(recs)
    if df.empty:
        raise SystemExit("[analyze] không có quỹ đạo nào để phân tích")
    df = add_r15(df, curves, cfg)
    gate = pd.DataFrame(gates)
    paths.results.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.results / "summary_runs.csv", index=False)
    gate.to_csv(paths.results / "gate1.csv", index=False)
    summ = summarize(df)
    summ.to_csv(paths.results / "summary_rules.csv", index=False)
    compare_retrain(df).to_csv(paths.results / "compare_retrain.csv", index=False)
    crit = criteria(df, gate) if len(gate) else pd.DataFrame()
    crit.to_csv(paths.results / "criteria.csv", index=False)
    rej = reject_checks(df, gate) if len(gate) else {}
    st = [sign_test(df, "R7-PRE-J", b) for b in ("R0-do-nothing", "R3-retain-only", "R4-val-only", "R5-heldout-hit")
          if (df.rule == "R7-PRE-J").any()]
    (paths.results / "curves.json").write_text(json.dumps(curves), encoding="utf-8")
    lrsel = lr_selection(paths, seeds, ftypes, methods, us)
    lrsel.to_csv(paths.results / "lr_selection.csv", index=False)
    write_report(paths, cfg, gate, summ, crit, rej, st, df, lrsel)
    return {"df": df, "gate": gate, "summary": summ, "criteria": crit, "reject": rej}


def lr_selection(paths, seeds, ftypes, methods, us=(0,)) -> pd.DataFrame:
    """Chọn lr KHÔNG dùng oracle, test hay khoảng cách tới retrain (đăng ký trước khi chạy lưới).

    Một điểm lưới (method@s) ĐỦ ĐIỀU KIỆN ở một loại F nếu ở MỌI seed tồn tại snapshot k ≥ 1 vừa qua rào G_R, G_V, vừa có
    L_F ≥ s_P, với s_P = trung bình đích P (loss của θ_pre trên F, chế độ none). Nghĩa là: quỹ đạo có điểm F đã quên tới mức P
    mà không làm hỏng R, V. Trong các điểm đủ điều kiện chọn lr LỚN NHẤT (ít epoch nhất). Không có điểm nào đủ thì không chọn.
    Chỉ đọc quỹ đạo và P; đây là bước chọn siêu tham số, kết quả lr vẫn phải ghi nhãn đã chọn bằng tiêu chí này."""
    rows = []
    for ft in ftypes:
        for me in methods:
            base, _, sc = me.partition("@")
            hits, n_adm, ok_all, found = [], [], True, 0
            for s in seeds:
                for u in us:
                    try:
                        tr = R.load_traj(paths, paths.run_id(me, ft, s, u))
                        P = load_P(paths, s, ft)
                    except FileNotFoundError:
                        ok_all = False
                        continue
                    found += 1
                    ctx = R.Ctx(np.asarray(P["loss_F_pre"], dtype=np.float64), tr[0]["L_R"], tr[0]["L_V"])
                    adm = [sn for sn in tr[1:] if ctx.admissible(sn)]
                    n_adm.append(len(adm))
                    hits.append(any(sn["L_F"] >= ctx.s for sn in adm))
            if not found:
                continue
            rows.append({"ftype": ft, "method": me, "base": base, "scale": float(sc) if sc else 1.0, "runs": found,
                         "adm": "/".join(map(str, n_adm)), "runs_hit_sP": int(sum(hits)),
                         "eligible": bool(ok_all and all(hits))})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["chosen"] = False
    for (_, _), g in out[out.eligible].groupby(["ftype", "base"]):
        out.loc[g.scale.idxmax(), "chosen"] = True
    return out


def gate_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Mỗi (ftype, method): số snapshot k ≥ 1 qua rào theo từng seed, số seed có adm = 0, và tỉ lệ luật PRE online chọn k=0."""
    runs = df.drop_duplicates(["seed", "ftype", "method", "u"])
    rows = []
    for (ft, me), g in runs.groupby(["ftype", "method"], sort=False):
        g = g.sort_values("seed")
        pre = df[(df.ftype == ft) & (df.method == me) & df.rule.isin(PRE_ONLINE)]
        rows.append({"ftype": ft, "method": me, "seeds": len(g),
                     "adm": "/".join(f"{int(a)}" for a in g.n_adm_after0),
                     "of": "/".join(f"{int(a)}" for a in g.n_snap_after0),
                     "seeds_adm0": int((g.n_adm_after0 == 0).sum()),
                     "pct_PRE_online_k0": 100.0 * float(pre.pick_k0.mean()) if len(pre) else float("nan")})
    return pd.DataFrame(rows)


def _md(df: pd.DataFrame, floatfmt="{:.3g}") -> str:
    if df.empty:
        return "_(trống)_"
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                      for v in r) + " |")
    return "\n".join(out)


def write_report(paths, cfg, gate, summ, crit, rej, st, df, lrsel=None):
    L = ["# Báo cáo PRE-stop (tự sinh)\n",
         f"- Seed: {sorted(df.seed.unique().tolist())}; loại F: {sorted(df.ftype.unique().tolist())}; "
         f"phương pháp: {sorted(df.method.unique().tolist())}; số dòng (run×luật): {len(df)}",
         f"- Mọi khoảng cách chuẩn hoá bằng nhiễu sàn ν giữa M={3} oracle (§9.1). Median và IQR qua seed (§9.6).",
         "- Kết quả khám phá phải ghi nhãn exploratory; ngưỡng ở §12 chốt trước khi chạy.\n",
         "## 1. Bảng tiền đề (Cổng 1), trước mọi kết quả luật\n", _md(gate), "",
         "## 2. Bảng chính: median regret chuẩn hoá theo luật\n"]
    for (ft, me), g in summ.groupby(["ftype", "method"]):
        L += [f"### {ft} · {me}\n", _md(g.drop(columns=["ftype", "method"])), ""]
    L += ["## 2b. So sánh với mô hình retrain (oracle θ_R2, trung bình M=3), median qua seed\n",
          "Mỗi nhóm có một dòng RETRAIN (mốc so sánh) và các luật dừng. L = loss, A = accuracy; "
          "F = forget set, R = retain (R_sub 5000 mẫu), V = validation, T = test. "
          "Cột phụ: g_T = |A_T − A_T(retrain)| (điểm %), G_KL = KL(softmax oracle ‖ mô hình) trên F, "
          "auc_gap = |AUC MIA(F vs V, max-softmax) − AUC của oracle|, gapavg = trung bình |lệch| của FA, RA, TA, AUC (điểm %); "
          "MIA tổng thể là chỉ báo yếu, đọc kèm g_T và G_KL. "
          "Bản đầy đủ kèm IQR và chênh lệch có dấu `gap_*` nằm ở `results/compare_retrain.csv`. "
          "R1-rollback chính là θ_pre: L_R, A_R, A_V, G_KL, auc_gap, gapavg chỉ có khi đã chạy lệnh `prefull` "
          "(không thì NaN).\n"]
    cmp_ = compare_retrain(df)
    show = ["rule", "target", "n"] + list(CMP) + ["g_T", "G_KL", "auc_gap", "gapavg"]
    for (ft, me), g in cmp_.groupby(["ftype", "method"], sort=False):
        L += [f"### {ft} · {me}\n", _md(g[show]), ""]
    L += ["## 3. Tiêu chí thành công §12.1 (báo đủ các biến thể, không chọn biến thể tốt nhất sau khi xem)\n",
          _md(crit), "", "## 4. Điều kiện bác bỏ §12.2\n", "```json\n" + json.dumps(rej, indent=2) + "\n```", "",
          "## 5. Kiểm định dấu (R7 so với baseline)\n", _md(pd.DataFrame(st)), "",
          "## 5b. Độ phủ của rào G_R và G_V (số snapshot k ≥ 1 được chấp nhận)\n",
          "Cờ `no_admissible` không bật khi luật rơi về k=0, vì k=0 luôn qua rào. Cột `adm` cho biết luật có điểm unlearn nào để chọn: "
          "`adm = 0` nghĩa là mọi luật online buộc phải chọn k=0 dù quỹ đạo có điểm tốt, nên kết quả k=0 ở ô đó "
          "không đánh giá luật PRE. Rào là hằng số [FIXED], không phụ thuộc đích.\n",
          _md(gate_coverage(df)), "",
          "## 5c. Chọn lr không dùng oracle (chỉ có ý nghĩa khi chạy lưới lr)\n",
          "Điểm lưới đủ điều kiện nếu ở MỌI seed có snapshot k ≥ 1 qua rào G_R, G_V và có L_F ≥ s_P (F đã quên tới mức loss của θ_pre). "
          "Chọn lr lớn nhất trong các điểm đủ điều kiện. Tiêu chí chỉ đọc quỹ đạo và P, không đọc oracle, test hay khoảng cách tới retrain. "
          "`adm` = số snapshot k ≥ 1 qua rào theo từng seed; `runs_hit_sP` = số seed có snapshot như vậy.\n",
          _md(lrsel if lrsel is not None else pd.DataFrame()), "",
          "## 6. Cảnh báo pipeline\n",
          f"- Số lần cờ no_admissible: {int(df.no_admissible.sum())} (không phát hiện được trường hợp mọi snapshot k ≥ 1 bị rào chặn, xem mục 5b); "
          f"cờ k* không có k thoả utility: "
          f"{int(df.groupby(['seed', 'ftype', 'method', 'u']).flag_no_util_k_star.max().sum())}",
          "- R13 (MU-Mis) chưa cài (cần gradient theo mẫu); R14 và R15 dùng oracle, chỉ để đo trần, không tuyên bố.",
          "- R11 dùng Err_V khớp histogram lớp của F tại snapshot cuối; R10 chuẩn hoá theo giá trị lớn nhất trên quỹ đạo (offline).",
          "- Mô hình ở chế độ BN=train khi unlearn (ghi rõ theo §6)."]
    (paths.results / "report_auto.md").write_text("\n".join(L), encoding="utf-8")

