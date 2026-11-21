"""Aggregate (§15): đọc trajectory + retrain, chọn k̂ theo mọi luật (chỉ từ metrics), rồi MỚI chấm bằng oracle.

Đây là nơi duy nhất ghép score_ralm với eval_oracle; mỗi bên vẫn không import bên kia.
Đầu ra: results/runs/{rid}.json, results/summary_{tag}.csv, results/anchors_{tag}.csv, results/rules_table_{tag}.md, figs/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import eval_oracle as O
from . import rules as R
from . import score_ralm as S
from .config import BAND_EPS_D, BAND_EPS_V, Paths, Profile


# ----------------------------------------------------------------------------- MIA proxy (§10.3)
def auc_rank(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC(pos > neg) bằng thống kê hạng (Mann–Whitney)."""
    from scipy.stats import rankdata
    x = np.concatenate([pos, neg])
    r = rankdata(x)
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logit_gap(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64)
    zt = z[np.arange(len(y)), y]
    z2 = z.copy(); z2[np.arange(len(y)), y] = -np.inf
    return zt - z2.max(1)


def mia_auc(f_logits, f_y, h_logits, h_y) -> float:
    """AUC phân biệt F (thành viên cũ) với holdout bằng logit-gap. 0.5 = không rò rỉ."""
    return auc_rank(logit_gap(f_logits, f_y), logit_gap(h_logits, h_y))


# ----------------------------------------------------------------------------- phân tích một run
def analyze_run(profile: Profile, paths: Paths, rid: str, fkey: str, base_seed: int, k_min: int | None = None,
                delta_F: float | None = None, ema: float | None = None) -> dict | None:
    k_min = profile.k_min if k_min is None else k_min
    traj = S.load_traj(paths.traj_jsonl(rid))
    meta = json.loads(paths.meta(rid).read_text(encoding="utf-8"))
    arr = S.load_arrays(paths.traj_jsonl(rid).with_name(f"traj_{rid}_arrays.npz"))      # không có tlogits
    full = np.load(paths.traj_jsonl(rid).with_name(f"traj_{rid}_arrays.npz"))            # oracle-side: có tlogits
    arr_o = {**arr, "tlogits": full["tlogits"], "t_y": full["t_y"]}
    retrains = O.load_retrains(paths, fkey, base_seed)
    if not retrains:
        return None
    steps = np.array([m["step"] for m in traj])
    cur = O.oracle_curves(arr_o, arr_o["tlogits"], traj, retrains)
    G = cur["G_logit_F"]
    n_D, n_V = int(arr["dperloss"].shape[1]), int(arr["vlogits"].shape[1])
    n_R = n_D - len(arr["f_y"])
    sel = R.run_rules(traj, profile.fixed_steps, k_min, n_D, n_V, n_R, delta_F, ema)
    sc = S.compute_scores(traj, ema_beta=ema)
    band = S.band_mask(sc, BAND_EPS_D, BAND_EPS_V, 0)
    k_star = int(steps[np.argmin(G)])
    rows = []
    t_logits, t_y = full["tlogits"], full["t_y"]
    # tham chiếu retrain (trung vị qua retrain seed) cho các chỉ số chất lượng unlearn
    rmeta = [rt["meta"] for rt in retrains]
    ref = {k: float(np.median([m[k] for m in rmeta])) for k in ("A_R", "A_F", "A_V", "A_T")}
    f_idx_ = arr["f_idx"] if "f_idx" in arr else full["f_idx"]
    ref["MIA"] = float(np.median([mia_auc(rt["train_logits"][f_idx_], arr["f_y"], rt["test_logits"], t_y)
                                  for rt in retrains]))
    for rule, v in sel.items():
        r = O.regret(G, steps, v["k"])
        i = int(np.argmin(np.abs(steps - v["k"])))
        q = {"A_R": traj[i]["A_R"], "A_F": traj[i]["A_F"], "A_V": traj[i]["A_V"], "A_T": float(cur["A_T_k"][i]),
             "MIA": mia_auc(arr["flogits"][i], arr["f_y"], t_logits[i], t_y)}
        gaps = {f"gap_{k}": abs(q[k] - ref[k]) for k in ("A_R", "A_F", "A_T", "MIA")}
        rows.append({"run_id": rid, "method": meta["method"], "fkey": fkey, "seed": base_seed, "rho": meta["rho"],
                     "rule": rule, "k_hat": v["k"], "online_fwd_per_snap": v["fwd"],
                     "at_kmin": int(v["k"] == int(steps[steps >= k_min][0]) if (steps >= k_min).any() else 0),
                     "G_F_at_k": float(cur["G_F"][i]), "G_test_acc": float(cur["G_test_acc"][i]),
                     "G_logit_V": float(cur["G_logit_V"][i]), "A_T": q["A_T"],
                     "A_T_retrain": cur["A_T_retrain"],
                     "A_R": q["A_R"], "A_F": q["A_F"], "A_V": q["A_V"], "MIA": q["MIA"],
                     "A_R_retrain": ref["A_R"], "A_F_retrain": ref["A_F"], "MIA_retrain": ref["MIA"],
                     **gaps, "avg_gap": float(np.mean(list(gaps.values()))), **r})
    # --- T-series (re-analysis, không chạy thêm GPU)
    arows = []
    base_k = {}
    SR = np.array([m["S_R"] for m in sc])
    SD = np.array([m["S_D"] for m in sc])
    for a in S.ANCHOR_NAMES:
        try:
            A = S.anchor_series(a, traj, {**arr, **{k: full[k] for k in ("d_y",) if k in full.files}})
        except Exception as e:  # noqa: BLE001
            arows.append({"run_id": rid, "anchor": a, "error": str(e)}); continue
        for mode in ("replace_D", "only"):
            k = S.select_with_anchor(traj, A, mode, k_min, delta_F=delta_F)
            r = O.regret(G, steps, k)
            arows.append({"run_id": rid, "method": meta["method"], "fkey": fkey, "seed": base_seed,
                          "anchor": a, "mode": mode, "k_hat": k, **r,
                          "corr_SR": S.redundancy(A, SR),
                          "corr_Glogit": (float(np.corrcoef(A, G)[0, 1]) if A.std() > 1e-15 else float("nan"))})
    # --- band H4 (§0A.7): coverage, width, in-band dispersion của G_logit_F
    in_b, out_b = G[band], G[~band]
    near = bool(band.any() and np.min(np.abs(steps[band] - k_star)) <= 100)
    band_info = {"run_id": rid, "method": meta["method"], "fkey": fkey, "seed": base_seed,
                 "band_n": int(band.sum()), "path_width": S.longest_run(band), "k_star": k_star,
                 "coverage": near,
                 "G_in_mean": float(in_b.mean()) if in_b.size else float("nan"),
                 "G_in_var": float(in_b.var()) if in_b.size else float("nan"),
                 "G_out_mean": float(out_b.mean()) if out_b.size else float("nan"),
                 "G_out_var": float(out_b.var()) if out_b.size else float("nan")}
    # --- privacy proxy tại k̂ của ralm-dv / retain-only / fixed-last / retrain
    priv = {}
    for rule in ("ralm-dv", "retain-only", "fixed-last"):
        i = int(np.argmin(np.abs(steps - sel[rule]["k"])))
        priv[rule] = mia_auc(arr["flogits"][i], arr["f_y"], t_logits[i], t_y)
    rt0 = retrains[0]["train_logits"]
    priv["retrain"] = mia_auc(rt0[arr["f_idx"] if "f_idx" in arr else full["f_idx"]], arr["f_y"],
                              retrains[0]["test_logits"], t_y)
    priv["theta0"] = mia_auc(arr["flogits"][0], arr["f_y"], t_logits[0], t_y)
    out = {"run_id": rid, "k_star": k_star, "G_range": [float(G.min()), float(G.max())],
           "L_D_gap": cur["L_D_gap"], "rules": rows, "anchors": arows, "band": band_info, "mia_auc": priv,
           "curves": {"steps": steps.tolist(), "G_logit_F": G.tolist(), "G_logit_V": cur["G_logit_V"].tolist(),
                      "J_DV": [m["J_DV"] for m in sc], "J_D": [m["J_D"] for m in sc], "J_V": [m["J_V"] for m in sc],
                      "L_R": [m["L_R"] for m in traj], "L_V": [m["L_V"] for m in traj],
                      "L_F": [m["L_F"] for m in traj], "L_D": [m["L_D"] for m in traj],
                      "band": band.tolist(), "A_T": cur["A_T_k"].tolist()},
           "go_nogo": {"corr_J_G": float(np.corrcoef([m["J_DV"] for m in sc], G)[0, 1]) if np.std(G) > 0 else float("nan")}}
    p = paths.results / "runs" / f"{rid}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out), encoding="utf-8")
    return out


# ----------------------------------------------------------------------------- tổng hợp
def iqr(x):
    x = np.asarray(x, dtype=float)
    return f"{np.nanpercentile(x, 25):.4g}–{np.nanpercentile(x, 75):.4g}"


def summarize(paths: Paths, tag: str, runs: list[dict]) -> dict:
    paths.results.mkdir(parents=True, exist_ok=True)
    rules_df = pd.DataFrame([r for o in runs for r in o["rules"]])
    anc_df = pd.DataFrame([r for o in runs for r in o["anchors"] if "error" not in r])
    band_df = pd.DataFrame([o["band"] for o in runs])
    mia_df = pd.DataFrame([{"run_id": o["run_id"], **o["mia_auc"]} for o in runs])
    rules_df.to_csv(paths.results / f"summary_{tag}.csv", index=False)
    anc_df.to_csv(paths.results / f"anchors_{tag}.csv", index=False)
    band_df.to_csv(paths.results / f"band_{tag}.csv", index=False)
    mia_df.to_csv(paths.results / f"mia_{tag}.csv", index=False)
    # bảng chính theo luật, tách (method, fkey)  — median + IQR (§18.4)
    md = []
    # gộp random-k0..4 thành 1 luật "random-k" (mỗi mẫu = 1 seed k)
    rd = rules_df.copy()
    rd["rule"] = rd["rule"].str.replace(r"random-k\d", "random-k", regex=True)
    # bảng chất lượng unlearn tại k̂ (median qua run): accuracy F/R/test, MIA, khoảng cách tới retrain
    qm = ["### Chất lượng mô hình tại k̂ — median qua run (gap = |giá trị − retrain|; avg_gap = trung bình 4 gap; nhỏ hơn tốt hơn)"]
    for (m, f), g in rd.groupby(["method", "fkey"]):
        h0 = g.iloc[0]
        qm.append(f"\n#### {m} / {f}  (n_runs={g.run_id.nunique()})\n")
        qm.append(f"Retrain: A_F={h0.A_F_retrain:.4f} A_R={h0.A_R_retrain:.4f} A_T={h0.A_T_retrain:.4f} MIA={h0.MIA_retrain:.3f}\n")
        qm.append("| Rule | k̂ | A_F | A_R | A_T | MIA | gap A_F | gap A_R | gap A_T | gap MIA | avg_gap |")
        qm.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for rule, h in g.groupby("rule", sort=False):
            qm.append(f"| {rule} | {h.k_hat.median():.0f} | {h.A_F.median():.4f} | {h.A_R.median():.4f} | "
                      f"{h.A_T.median():.4f} | {h.MIA.median():.3f} | {h.gap_A_F.median():.4f} | {h.gap_A_R.median():.4f} | "
                      f"{h.gap_A_T.median():.4f} | {h.gap_MIA.median():.3f} | {h.avg_gap.median():.4f} |")
    (paths.results / f"quality_table_{tag}.md").write_text("\n".join(qm), encoding="utf-8")
    for (m, f), g in rd.groupby(["method", "fkey"]):
        md.append(f"\n### {m} / {f}  (n_runs={g.run_id.nunique()})\n")
        md.append("| Rule | median regret | IQR | median regret_rel | median k̂ | % k̂=k_min | online fwd/snap | median |A_T−A_T(θR)| |")
        md.append("|---|---|---|---|---|---|---|---|")
        for rule, h in g.groupby("rule", sort=False):
            md.append(f"| {rule} | {h.regret.median():.4g} | {iqr(h.regret)} | {h.regret_rel.median():.3g} | "
                      f"{h.k_hat.median():.0f} | {100 * h.at_kmin.mean():.0f}% | {int(h.online_fwd_per_snap.median())} | "
                      f"{(h.A_T - h.A_T_retrain).abs().median() * 100:.2f}pp |")
    (paths.results / f"rules_table_{tag}.md").write_text("\n".join(md), encoding="utf-8")
    return {"rules": rules_df, "anchors": anc_df, "band": band_df, "mia": mia_df}


def t_matrix(anc_df: pd.DataFrame, rules_df: pd.DataFrame) -> pd.DataFrame:
    """§0A.6.2: anchor 'thắng' T0 nếu median regret giảm ≥ 20% trên highloss VÀ classwise, không tệ > 10% trên random.

    So sánh trong cùng mode='replace_D' (anchor thay S_D, giữ S_V)."""
    a = anc_df[anc_df["mode"] == "replace_D"].copy()
    a["ftype"] = a["fkey"].str.split("_").str[0]
    med = a.groupby(["method", "ftype", "anchor"])["regret"].median().unstack("anchor")
    rows = []
    for (m, ft), r in med.iterrows():
        for an in r.index:
            rows.append({"method": m, "ftype": ft, "anchor": an, "median_regret": r[an],
                         "vs_T0": (r[an] - r["T0"]) / (abs(r["T0"]) + 1e-12)})
    d = pd.DataFrame(rows)
    out = []
    for (m, an), g in d.groupby(["method", "anchor"]):
        g = g.set_index("ftype")["vs_T0"]
        ok_hl = g.get("highloss", np.nan) <= -0.20
        ok_cw = g.get("classwise", np.nan) <= -0.20
        ok_rd = g.get("random", np.nan) <= 0.10
        out.append({"method": m, "anchor": an, "highloss": g.get("highloss"), "classwise": g.get("classwise"),
                    "random": g.get("random"), "WIN": bool(ok_hl and ok_cw and ok_rd)})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- hình
def make_figs(paths: Paths, runs: list[dict], tag: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths.figs.mkdir(parents=True, exist_ok=True)
    for o in runs:
        c = o["curves"]; st = np.array(c["steps"])
        rl = {r["rule"]: r["k_hat"] for r in o["rules"]}
        fig, ax = plt.subplots(1, 2, figsize=(13, 4))
        ax[0].plot(st, c["G_logit_F"], "k", label="G_logit_F (oracle)")
        ax[0].plot(st, c["G_logit_V"], "gray", ls=":", label="G_logit_V")
        for name, col in (("ralm-dv", "r"), ("ralm-d", "orange"), ("retain-only", "b"), ("val-only", "g"),
                          ("fixed-last", "m")):
            ax[0].axvline(rl[name], color=col, ls="--", alpha=.8, label=f"{name} k̂={rl[name]}")
        ax[0].axvline(o["k_star"], color="k", lw=2, alpha=.3, label=f"k*={o['k_star']}")
        ax[0].set_xlabel("step"); ax[0].legend(fontsize=6); ax[0].set_title(f"regret curve · {o['run_id']}", fontsize=8)
        for key in ("J_DV", "J_D", "J_V"):
            ax[1].plot(st, c[key], label=key)
        ax[1].plot(st, np.array(c["L_R"]) / c["L_R"][0], ":", label="L_R/L_R0")
        ax[1].plot(st, np.array(c["L_V"]) / c["L_V"][0], ":", label="L_V/L_V0")
        for key, mk in (("J_DV", "r*"),):
            i = int(np.argmin(np.abs(st - rl["ralm-dv"]))); ax[1].plot(st[i], c[key][i], mk, ms=12)
        ax[1].legend(fontsize=6); ax[1].set_xlabel("step"); ax[1].set_title("J overlay", fontsize=8)
        fig.tight_layout(); fig.savefig(paths.figs / f"regret_curve_{o['run_id']}.png", dpi=110); plt.close(fig)
    rd = pd.DataFrame([r for o in runs for r in o["rules"]])
    if rd.empty:
        return
    rd["rule_g"] = rd["rule"].str.replace(r"random-k\d", "random-k", regex=True)
    order = [r for r in ("fixed-100", "fixed-200", "fixed-500", "fixed-last", "random-k", "retain-only", "val-only",
                         "pareto-knee", "ralm-d", "ralm-v", "ralm-dv", "ralm-band") if r in set(rd.rule_g)]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.boxplot([rd[rd.rule_g == r].regret.values for r in order], showfliers=True)
    ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels(order)
    ax.set_ylabel("regret (G_logit_F)"); ax.set_yscale("symlog", linthresh=1e-3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(paths.figs / f"box_regret_by_rule_{tag}.png", dpi=110); plt.close(fig)
    d = rd[rd.rule == "ralm-dv"]
    if len(d):
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        for m, g in d.groupby("method"):
            ax.scatter(g.k_star, g.k_hat, label=m, s=30)
        mx = max(d.k_hat.max(), d.k_star.max()) * 1.05
        ax.plot([0, mx], [0, mx], "k--", lw=.7); ax.set_xlabel("k*"); ax.set_ylabel("k̂ (ralm-dv)"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(paths.figs / f"scatter_khat_vs_kstar_{tag}.png", dpi=110); plt.close(fig)
