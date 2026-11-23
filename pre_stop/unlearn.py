"""Unlearning ga / scrub / salun / tarun / ft, đơn vị là EPOCH, theo code gốc của từng phương pháp.

Nguồn code gốc (đã clone vào _orig/ khi viết file này):
  ga      OPTML-Group/Unlearn-Saliency  Classification/unlearn/GA.py + impl.py
          SGD lr 0.01, momentum 0.9, wd 5e-4, batch 256, 10 epoch; mỗi epoch duyệt CHỈ F, loss = -CE, BN train.
  ft      Unlearn-Saliency FT.py: SGD lr 0.01, momentum 0.9, wd 5e-4, batch 256, 10 epoch trên R, BN train.
  salun   Unlearn-Saliency generate_mask.py + RL.py (nhánh cifar10): mask = top 50% |grad(-CE_F)| (model.eval),
          SGD lr 0.013, momentum 0.9, wd 5e-4, batch 256, 10 epoch; mỗi epoch: F với nhãn ngẫu nhiên rồi R,
          grad nhân mask, tham số ngoài mask trả về giá trị đầu epoch.
  scrub   meghdadk/SCRUB small_scale_unlearning.ipynb + thirdparty/repdistiller/helper/loops.py:
          Adam lr 5e-4, wd 0.1, 10 epoch, msteps 3, lr nhân 0.1 sau epoch 5, 8, 9, alpha 0.5, gamma 1, KD T=2,
          batch R 32 / F 64. Epoch e<=3: maximize (-KL) trên F rồi minimize (gamma*CE + alpha*KL) trên R.
  tarun   vikram2000b/Fast-Machine-Unlearning (UNSIR) notebook: học noise cho từng lớp cần quên (Adam 0.1, 5x8 bước),
          epoch 1 = impair (noise + 1000 mẫu R/lớp, Adam 0.02), epoch 2 = repair (1000 mẫu R/lớp, Adam 0.01), batch 256.

Mọi quyết định "epoch" đều lấy từ vòng for epoch trong code gốc. Snapshot lấy ở CUỐI mỗi epoch (và epoch 0).
CÁCH LY (§0.4.1): file này không import eval_oracle và không đụng tới θ_R2. Metric test ghi vào file niêm phong.
"""
from __future__ import annotations

import copy
import itertools
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from ralm.eval_core import seed_all
from ralm.model import make_model

from .config import Cfg
from .data import load_sets
from .measure import measure, test_metrics

assert "eval_oracle" not in globals(), "unlearn.py không được import eval_oracle (§0.4.1)"

BN_MODE = {"ga": "train", "ft": "train", "salun": "train", "scrub": "train (student), eval (teacher)",
           "tarun": "train (noise: eval)"}


LR_GRID_METHODS = ("ga", "scrub")       # chỉ hai phương pháp này có lưới lr (EXPLORATORY), tên dạng "ga@0.1"


def split_method(name: str) -> tuple[str, float]:
    """"ga@0.1" -> ("ga", 0.1): phương pháp gốc + hệ số nhân lr so với lr của code gốc. "ga" -> ("ga", 1.0)."""
    base, _, sc = name.partition("@")
    scale = float(sc) if sc else 1.0
    if base not in METHOD_CLS:
        raise ValueError(f"phương pháp không biết: {name}")
    if sc and base not in LR_GRID_METHODS:
        raise ValueError(f"lưới lr chỉ có cho {LR_GRID_METHODS}, không có cho {base}")
    if scale <= 0:
        raise ValueError("hệ số lr phải > 0")
    return base, scale


def n_epochs_for(cfg: Cfg, method: str) -> int | None:
    """Số epoch của quỹ đạo. Tarun/UNSIR gốc đúng 2 epoch (impair, repair) và không kéo dài được.
    cfg.n_epochs (nếu đặt) là số epoch cố định, chạy đủ không dừng sớm. Nếu None: KHÔNG có trần, chạy đến khi mọi
    luật PRE online đã quyết định (hoặc phân kỳ); cfg.max_epochs chỉ là trần tuỳ chọn do người dùng tự đặt."""
    if method.partition("@")[0] == "tarun":
        return 2
    return cfg.n_epochs or cfg.max_epochs


def plateau(rows: list[dict], w: int, tol: float) -> bool:
    """Quỹ đạo "đứng yên": trong w epoch gần nhất, L_F, L_R, L_V đều có (max-min) <= tol * max(|trung bình|, 0.1).
    KHÔNG phải trần epoch: chỉ dừng khi quỹ đạo không còn thay đổi (các luật không thể quyết định khác đi nữa).
    Tiêu chí này do người triển khai đề xuất; không dùng oracle / test."""
    if w is None or len(rows) < w + 1:
        return False
    last = rows[-(w + 1):]
    for k in ("L_F", "L_R", "L_V"):
        v = np.array([r[k] for r in last], dtype=np.float64)
        if v.max() - v.min() > tol * max(abs(v.mean()), 0.1):
            return False
    return True


_PRE_STOP_RULES = ("R6-PRE-hit", "R7-PRE-J", "R8-PRE-sample", "R9-PRE-median-hit")


def _online_rules_for_stop(paths, seed, ftype, L_R0, L_V0) -> list:
    """Các luật PRE online (R6-R9, mỗi luật ở các chế độ đích none/V/log). Chỉ để quyết định khi nào ngừng thu
    quỹ đạo: không dùng oracle / test. Quỹ đạo kết thúc khi tất cả đã chọn xong checkpoint của mình.
    Không chờ R3/R4/R5/R12: chúng thường không kích hoạt (vd. F không bao giờ bị quên) nên sẽ không bao giờ dừng;
    các baseline đó được chấm trên cùng quỹ đạo, nếu chưa kích hoạt thì dùng finish() (ghi cờ triggered=0)."""
    from . import rules as R
    from .pre_info import load_P
    P = load_P(paths, seed, ftype)
    out = []
    for mode in ("none", "V", "log"):
        if mode == "log" and P.get("loss_F_log") is None:
            continue
        ctx = R.Ctx(R.make_target(P, mode, L_V0), L_R0, L_V0)
        out += [r for n, r in R.online_rules(ctx).items() if n in _PRE_STOP_RULES]
    return out


def _batches(idx: torch.Tensor, bs: int) -> list:
    """Một lượt qua idx đã xáo trộn; batch cuối giữ nguyên (DataLoader drop_last=False)."""
    perm = idx[torch.randperm(len(idx), device=idx.device)]
    return [perm[i:i + bs] for i in range(0, len(perm), bs)]


class Base:
    name = ""

    def __init__(self, cfg: Cfg, data, sets, model, device):
        self.cfg, self.data, self.sets, self.model, self.dev = cfg, data, sets, model, device
        self.use_amp = cfg.amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.n_updates = 0
        self.opt = None

    def _fwd(self, x):
        with torch.autocast(device_type=self.dev.type, dtype=torch.float16, enabled=self.use_amp):
            out = self.model(x)
        return out.float()

    def _ce(self, idx, y=None):
        x, yy = self.data.batch(idx, augment=True)
        return F.cross_entropy(self._fwd(x), yy if y is None else y)

    def _update(self, loss, pre=None, post=None):
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        if pre is not None:
            self.scaler.unscale_(self.opt)
            pre()
        self.scaler.step(self.opt)
        self.scaler.update()
        if post is not None:
            post()
        self.n_updates += 1
        return float(loss.detach())


def _sgd(model, lr, wd):
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)


class GA(Base):
    name = "ga"

    def __init__(self, *a, lr_scale: float = 1.0):
        super().__init__(*a)
        self.opt = _sgd(self.model, 0.01 * lr_scale, 5e-4)       # lr_scale != 1: lưới lr EXPLORATORY

    def epoch(self) -> float:
        self.model.train()
        last = 0.0
        for b in _batches(self.sets.f, 256):                 # CHỈ forget_loader
            last = self._update(-self._ce(b))
        return last


class FT(Base):
    name = "ft"

    def __init__(self, *a):
        super().__init__(*a)
        self.opt = _sgd(self.model, 0.01, 5e-4)

    def epoch(self) -> float:
        self.model.train()
        last = 0.0
        for b in _batches(self.sets.r, 256):                 # CHỈ retain_loader
            last = self._update(self._ce(b))
        return last


class SalUn(Base):
    name = "salun"
    THRESHOLD = 0.5

    def __init__(self, *a):
        super().__init__(*a)
        self.opt = _sgd(self.model, 0.013, 5e-4)
        self.mask = self._mask()
        self.num_classes = self.cfg.num_classes

    def _mask(self) -> dict:
        m = copy.deepcopy(self.model).eval()
        g = {n: torch.zeros_like(p) for n, p in m.named_parameters()}
        for b in _batches(self.sets.f, 256):
            x, y = self.data.batch(b, augment=True)
            loss = -F.cross_entropy(m(x).float(), y)
            m.zero_grad(set_to_none=True)
            loss.backward()
            for n, p in m.named_parameters():
                if p.grad is not None:
                    g[n] += p.grad
        flat = torch.cat([v.abs().flatten() for v in g.values()])
        k = max(1, int(len(flat) * self.THRESHOLD))
        cut = flat.sort(descending=True).values[k - 1]
        return {n: (v.abs() >= cut).to(v.dtype) for n, v in g.items()}

    def epoch(self) -> float:
        self.model.train()
        theta0 = {n: p.detach().clone() for n, p in self.model.named_parameters() if n in self.mask}
        params = dict(self.model.named_parameters())

        def pre():
            for n, p in params.items():
                if p.grad is not None:
                    p.grad *= self.mask[n]

        def post():
            with torch.no_grad():
                for n, p in params.items():
                    inv = 1 - self.mask[n]
                    if torch.count_nonzero(inv) == 0:
                        continue
                    p.mul_(self.mask[n]).add_(theta0[n] * inv)
                    st = self.opt.state.get(p)
                    if st is not None and "momentum_buffer" in st:
                        st["momentum_buffer"].mul_(self.mask[n])

        last = 0.0
        for b in _batches(self.sets.f, 256):                 # F với nhãn ngẫu nhiên
            y = torch.randint(0, self.num_classes, (len(b),), device=self.dev)
            last = self._update(self._ce(b, y), pre, post)
        for b in _batches(self.sets.r, 256):                 # rồi R
            last = self._update(self._ce(b), pre, post)
        return last


class Scrub(Base):
    """SCRUB gốc (meghdadk): teacher = θ0 đóng băng, student = bản sao; không có SWA/param_dist ở bản gốc này."""
    name = "scrub"
    LR, WD, T = 5e-4, 0.1, 2.0
    ALPHA, GAMMA = 0.5, 1.0            # beta = 0 trong notebook gốc nên không có số hạng kd
    MSTEPS = 3
    DECAY_EPOCHS, DECAY_RATE = (5, 8, 9), 0.1
    BS_R, BS_F = 32, 64

    def __init__(self, *a, lr_scale: float = 1.0):
        super().__init__(*a)
        self.teacher = copy.deepcopy(self.model).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.lr0 = self.LR * lr_scale                             # lr_scale != 1: lưới lr EXPLORATORY
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr0, weight_decay=self.WD)
        self.ep = 0

    def _kl(self, s, t):
        T = self.T
        return F.kl_div(F.log_softmax(s / T, 1), F.softmax(t / T, 1), reduction="sum") * (T * T) / s.shape[0]

    def _pair(self, idx):
        x, y = self.data.batch(idx, augment=False)       # notebook gốc: augment=False
        s = self._fwd(x)
        with torch.no_grad():
            t = self.teacher(x) if not self.use_amp else self._fwd_t(x)
        return s, t.float(), y

    def _fwd_t(self, x):
        with torch.autocast(device_type=self.dev.type, dtype=torch.float16, enabled=self.use_amp):
            return self.teacher(x)

    def epoch(self) -> float:
        self.ep += 1
        k = sum(self.ep > e for e in self.DECAY_EPOCHS)           # sgda_adjust_learning_rate
        if k > 0:
            for g in self.opt.param_groups:
                g["lr"] = self.lr0 * (self.DECAY_RATE ** k)
        self.model.train()
        self.teacher.eval()
        last = 0.0
        if self.ep <= self.MSTEPS:                                # maximize trên F
            for b in _batches(self.sets.f, self.BS_F):
                s, t, _ = self._pair(b)
                last = self._update(-self._kl(s, t))
        for b in _batches(self.sets.r, self.BS_R):                # minimize trên R
            s, t, y = self._pair(b)
            last = self._update(self.GAMMA * F.cross_entropy(s, y) + self.ALPHA * self._kl(s, t))
        return last


class Tarun(Base):
    """UNSIR (Tarun et al.) theo notebook gốc. Chỉ định nghĩa cho quên THEO LỚP: classes_to_forget = lớp của F."""
    name = "tarun"
    PER_CLASS_RETAIN, NOISE_BS, NOISE_BATCHES, BS = 1000, 256, 20, 256

    def __init__(self, cfg, data, sets, model, device):
        super().__init__(cfg, data, sets, model, device)
        self.ep = 0
        self.classes = sorted(int(c) for c in np.unique(data.ytr_np[sets.f_np]))
        ytr = data.ytr_np[sets.r_np]
        take = [sets.r_np[ytr == c][:self.PER_CLASS_RETAIN] for c in range(cfg.num_classes) if c not in self.classes]
        self.ret = torch.from_numpy(np.concatenate(take)).to(device)
        assert len(self.ret) > 0, "tarun cần ít nhất một lớp giữ lại"
        self.noises = self._learn_noise()

    def _learn_noise(self) -> dict:
        self.model.eval()
        out = {}
        for c in self.classes:
            noise = torch.nn.Parameter(torch.randn(self.NOISE_BS, 3, 32, 32, device=self.dev))
            opt = torch.optim.Adam([noise], lr=0.1)
            lab = torch.full((self.NOISE_BS,), c, dtype=torch.long, device=self.dev)
            for _ in range(5):                                    # num_epochs = 5
                for _ in range(8):                                # num_steps = 8
                    loss = (-F.cross_entropy(self.model(noise).float(), lab)
                            + 0.1 * torch.mean(torch.sum(noise ** 2, [1, 2, 3])))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            self.model.zero_grad(set_to_none=True)
            out[c] = noise.detach()
        return out

    def _run(self, x, y, lr):
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        last = 0.0
        perm = torch.randperm(len(x), device=self.dev)
        for i in range(0, len(x), self.BS):
            b = perm[i:i + self.BS]
            last = self._update(F.cross_entropy(self._fwd(x[b]), y[b]))
        return last

    def epoch(self) -> float:
        self.ep += 1
        xr, yr = self.data.batch(self.ret, augment=False)
        if self.ep == 1:                                          # impair: noise (nhãn = lớp) + retain, Adam 0.02
            xn = torch.cat([self.noises[c].repeat(self.NOISE_BATCHES, 1, 1, 1) for c in self.classes])
            yn = torch.cat([torch.full((self.NOISE_BS * self.NOISE_BATCHES,), c, dtype=torch.long, device=self.dev)
                            for c in self.classes])
            return self._run(torch.cat([xn, xr]), torch.cat([yn, yr]), 0.02)
        return self._run(xr, yr, 0.01)                            # repair: chỉ retain, Adam 0.01


METHOD_CLS = {"ga": GA, "ft": FT, "salun": SalUn, "scrub": Scrub, "tarun": Tarun}
TARUN_FTYPES = ("class-0",)       # UNSIR gốc chỉ quên theo lớp: không áp dụng được cho F ngẫu nhiên / hard


def applicable(method: str, ftype: str, main_ftype: str = "random-450") -> bool:
    method = method.partition("@")[0]
    if method == "tarun":
        return ftype in TARUN_FTYPES
    if method == "ft":
        return ftype == main_ftype
    return True


def _load_theta0(cfg, paths, seed, ftype, device):
    m = make_model(cfg.as_profile(), device)
    m.load_state_dict(torch.load(paths.theta0(seed, ftype), map_location=device))
    return m


def collect(cfg: Cfg, paths, data, method: str, ftype: str, seed: int, device, u: int = 0,
            log=print, force: bool = False) -> dict:
    rid = paths.run_id(method, ftype, seed, u)
    jl, arr_p, meta_p, test_p = paths.traj_jsonl(rid), paths.traj_arrays(rid), paths.meta(rid), paths.traj_test(rid)
    if meta_p.exists() and arr_p.exists() and not force:
        old = json.loads(meta_p.read_text(encoding="utf-8"))
        if "stop_reason" in old:
            log(f"[traj] {rid} đã xong, bỏ qua")
            return old
        log(f"[traj] {rid} là quỹ đạo kiểu cũ (cắt cứng số epoch, không có stop_reason): chạy lại")
    sets = load_sets(paths, data, cfg, seed, ftype)
    seed_all(seed + 1000 * u)
    model = _load_theta0(cfg, paths, seed, ftype, device)
    base, scale = split_method(method)
    um = METHOD_CLS[base](cfg, data, sets, model, device, **({"lr_scale": scale} if base in LR_GRID_METHODS else {}))
    n_ep = n_epochs_for(cfg, method)
    jl.parent.mkdir(parents=True, exist_ok=True)
    jl.write_text("", encoding="utf-8")
    test_p.write_text("", encoding="utf-8")
    arrays = {k: [] for k in ("epochs", "loss_F", "loss_V", "logits_F", "logits_V")}

    def snapshot(ep: int, t_wall: float, last_loss):
        model.eval()
        lgR, lsR = measure(model, data, cfg, sets.rsub)
        lgF, lsF = measure(model, data, cfg, sets.f)
        lgV, lsV = measure(model, data, cfg, sets.v)
        for t in (lsR, lsF, lsV):
            if not torch.isfinite(t).all():
                raise FloatingPointError("per-sample loss không hữu hạn (model phân kỳ)")
        acc = lambda lg, idx: float((lg.argmax(1) == data.ytr[idx]).double().mean())    # noqa: E731
        row = {"run_id": rid, "epoch": ep, "n_updates": um.n_updates, "wall_time_s": round(t_wall, 3),
               "L_R": float(lsR.double().mean()), "A_R": acc(lgR, sets.rsub),
               "L_F": float(lsF.double().mean()), "A_F": acc(lgF, sets.f),
               "L_V": float(lsV.double().mean()), "A_V": acc(lgV, sets.v),
               "mean_conf_F": float(F.softmax(lgF, 1).max(1).values.double().mean()),
               "train_loss_last": last_loss}
        L_T, A_T = test_metrics(model, data, cfg)                         # niêm phong: file riêng (§7)
        with open(jl, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n"); f.flush()
        with open(test_p, "a", encoding="utf-8") as f:
            f.write(json.dumps({"run_id": rid, "epoch": ep, "L_T": L_T, "A_T": A_T,
                                "post_selection": True}) + "\n"); f.flush()
        arrays["epochs"].append(ep)
        arrays["loss_F"].append(lsF.cpu().numpy().astype(np.float32))
        arrays["loss_V"].append(lsV.cpu().numpy().astype(np.float32))
        arrays["logits_F"].append(lgF.cpu().numpy().astype(np.float16))
        arrays["logits_V"].append(lgV.cpu().numpy().astype(np.float16))
        return row

    t0 = time.time()
    row0 = snapshot(0, 0.0, None)
    fixed = cfg.n_epochs is not None or base == "tarun"
    rows = [row0]
    stop_rules = [] if fixed else _online_rules_for_stop(paths, seed, ftype, row0["L_R"], row0["L_V"])
    pending = list(stop_rules)

    def feed_rules(k: int, sn_row: dict):
        nonlocal pending
        sn = dict(sn_row, loss_F=arrays["loss_F"][-1].astype(np.float64))
        pending = [r for r in pending if r.feed(k, sn) is None]

    feed_rules(0, row0)                     # giống run_online: luật thấy snapshot 0, 1, 2, ... theo thứ tự
    log(f"[traj] {rid} |F|={len(sets.f)} "
        f"{n_ep if n_ep else 'không giới hạn'} epoch "
        f"({'cố định' if fixed else 'chạy đến khi mọi luật PRE online đã quyết định'}; code gốc: {base}, "
        f"hệ số lr {scale:g}{' [EXPLORATORY]' if scale != 1 else ''}, BN {BN_MODE[base]})")
    diverged, stop_reason, ep = None, ("fixed_epochs" if fixed else "max_epochs"), 0
    for ep in (range(1, n_ep + 1) if n_ep else itertools.count(1)):
        try:
            last = um.epoch()
            row = snapshot(ep, time.time() - t0, last)
            log(f"  epoch {ep:2d} L_R {row['L_R']:.4f} L_F {row['L_F']:.4f} L_V {row['L_V']:.4f} "
                f"A_F {row['A_F']:.3f} upd {um.n_updates} [{time.time() - t0:.0f}s]")
        except FloatingPointError as e:     # ghi rõ, không bỏ qua im lặng (§17.3)
            diverged = {"epoch": ep, "error": str(e)}
            stop_reason = "diverged"
            log(f"[traj] {rid} PHÂN KỲ ở epoch {ep}: quỹ đạo bị cắt ở đây (ghi cờ diverged)")
            break
        rows.append(row)
        if stop_rules:
            feed_rules(ep, row)
            if not pending:
                stop_reason = "all_rules_decided"
                log(f"[traj] {rid} mọi luật PRE online đã quyết định ở epoch {ep}: dừng quỹ đạo")
                break
            if cfg.plateau_w and plateau(rows, cfg.plateau_w, cfg.plateau_tol):
                stop_reason = "plateau"
                log(f"[traj] {rid} quỹ đạo đứng yên {cfg.plateau_w} epoch (tol {cfg.plateau_tol}) ở epoch {ep}: dừng "
                    f"({len(pending)}/{len(stop_rules)} luật chưa quyết định)")
                break
    if stop_reason == "max_epochs":      # chỉ xảy ra khi người dùng tự đặt --max_epochs
        log(f"[traj] {rid} chạm trần {n_ep} epoch do người dùng đặt; {len(pending)}/{len(stop_rules)} luật chưa quyết định")
    np.savez_compressed(arr_p, epochs=np.array(arrays["epochs"]), loss_F=np.stack(arrays["loss_F"]),
                        loss_V=np.stack(arrays["loss_V"]), logits_F=np.stack(arrays["logits_F"]),
                        logits_V=np.stack(arrays["logits_V"]), f_idx=sets.f_np, v_idx=sets.v_np,
                        f_y=data.ytr_np[sets.f_np], v_y=data.ytr_np[sets.v_np])
    meta = {"run_id": rid, "method": method, "base_method": base, "lr_scale": scale, "ftype": ftype, "seed": seed, "u": u, "n_epochs": ep if stop_reason != "diverged" else ep - 1,
            "max_epochs": n_ep, "stop_reason": stop_reason, "n_rules_undecided": len(pending),
            "n_snapshots": len(arrays["epochs"]), "n_updates": um.n_updates, "diverged": diverged,
            "wall_s": time.time() - t0, "arrays": arr_p.name, "bn_mode": BN_MODE[base]}
    meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
