"""Các phương pháp unlearning dưới dạng "máy bước": prepare() một lần, rồi step() lặp lại.

collect_traj.py điều khiển snapshot; file này KHÔNG đụng tới đo lường (patch tối thiểu §0B.2).
Recipe (lr, alpha, ...) lấy từ config.METHOD_HP, giữ nguyên giữa các seed (§18.5).

  ga      -CE(F)                                              (§5.1)
  ngp     -CE(F) + alpha*CE(R)                                (§5.2)
  scrub   max-step: -KL(t‖s) trên F ; min-step: alpha*KL(t‖s)+gamma*CE trên R   (Kurmanji 2023)
  salun   random-label trên F + CE(R), gradient mask theo saliency của F          (Fan 2024)
  rl      như salun nhưng không mask (đối chứng)
  tarun   UNSIR: học noise tăng loss theo class -> impair (noise + R) -> repair (R)  (Tarun 2023)
  ft      CE(R)  (đối chứng)
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F


class Cycler:
    """Lấy batch vô hạn từ một tập chỉ mục (shuffle lại sau mỗi vòng)."""

    def __init__(self, idx: torch.Tensor, bs: int):
        self.idx, self.bs, self.pos, self.perm = idx, min(bs, len(idx)), 0, None
        self._new()

    def _new(self):
        self.perm = self.idx[torch.randperm(len(self.idx), device=self.idx.device)]
        self.pos = 0

    def next(self) -> torch.Tensor:
        if self.pos + self.bs > len(self.perm):
            self._new()
        b = self.perm[self.pos:self.pos + self.bs]
        self.pos += self.bs
        return b


def kd_kl(student_logits, teacher_logits, T: float):
    """KL(teacher ‖ student) với nhiệt độ T, nhân T² (SCRUB)."""
    return F.kl_div(F.log_softmax(student_logits / T, 1), F.softmax(teacher_logits / T, 1),
                    reduction="batchmean") * (T * T)


class Unlearner:
    name = "base"

    def __init__(self, model, theta0_state, data, split, hp, profile, device):
        self.model, self.data, self.split, self.hp, self.profile, self.device = model, data, split, hp, profile, device
        self.amp = profile.amp and device.type == "cuda"
        self.bs = profile.unlearn_batch
        self.aug = profile.unlearn_augment
        self.theta0_state = theta0_state
        self.fc = Cycler(split.f, self.bs)
        self.rc = Cycler(split.r, self.bs)
        self.opt = torch.optim.SGD(model.parameters(), lr=hp["lr"], momentum=hp.get("momentum", 0.0),
                                   weight_decay=hp.get("wd", 0.0))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.t = 0

    # ---- tiện ích
    def _fwd(self, x):
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
            return self.model(x).float()

    def _fb(self, cyc):
        return self.data.batch(cyc.next(), augment=self.aug)

    def _update(self, loss, grad_hook=None):
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        if grad_hook is not None:
            grad_hook()
        self.scaler.step(self.opt)
        self.scaler.update()

    def prepare(self):
        """Chuẩn bị một lần trước step 1 (không tính vào số bước). Mô hình không được đổi."""

    def step(self) -> float:
        raise NotImplementedError

    def lr(self) -> float:
        return self.opt.param_groups[0]["lr"]


class GA(Unlearner):
    name = "ga"

    def step(self):
        x, y = self._fb(self.fc)
        loss = -F.cross_entropy(self._fwd(x), y)
        self._update(loss)
        return float(loss.detach())


class NGPlus(Unlearner):
    name = "ngp"

    def step(self):
        xf, yf = self._fb(self.fc)
        xr, yr = self._fb(self.rc)
        loss = -F.cross_entropy(self._fwd(xf), yf) + self.hp["alpha"] * F.cross_entropy(self._fwd(xr), yr)
        self._update(loss)
        return float(loss.detach())


class FT(Unlearner):
    name = "ft"

    def step(self):
        x, y = self._fb(self.rc)
        loss = F.cross_entropy(self._fwd(x), y)
        self._update(loss)
        return float(loss.detach())


class SCRUB(Unlearner):
    name = "scrub"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.teacher = copy.deepcopy(self.model).eval()
        self.teacher.load_state_dict(self.theta0_state)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _t(self, x):
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
            return self.teacher(x).float()

    def step(self):
        hp, T = self.hp, self.hp["temp"]
        if self.t < hp["max_iters"]:                 # max-step: đẩy student ra xa teacher trên F
            xf, _ = self._fb(self.fc)
            loss_max = -kd_kl(self._fwd(xf), self._t(xf), T)
            self._update(loss_max)
        xr, yr = self._fb(self.rc)                   # min-step: giữ gần teacher + CE trên R
        so = self._fwd(xr)
        loss = hp["alpha"] * kd_kl(so, self._t(xr), T) + hp["gamma"] * F.cross_entropy(so, yr)
        self._update(loss)
        return float(loss.detach())


class RandomLabel(Unlearner):
    """RL (không mask) và SalUn (có mask): random-label trên F + CE trên R, một bước/batch ghép."""
    name = "rl"
    use_mask = False

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.masks = None
        self.nc = self.profile.num_classes

    def prepare(self):
        if not self.use_mask:
            return
        m = self.model
        was = m.training
        m.eval()
        grads = [torch.zeros_like(p) for p in m.parameters()]
        n_b = 0
        for s in range(0, len(self.split.f), 256):
            idx = self.split.f[s:s + 256]
            x, y = self.data.batch(idx, augment=False)
            m.zero_grad(set_to_none=True)
            F.cross_entropy(m(x), y).backward()
            for g, p in zip(grads, m.parameters()):
                g += p.grad.detach()
            n_b += 1
        m.zero_grad(set_to_none=True)
        sal = torch.cat([g.abs().reshape(-1) for g in grads])
        k = int(self.hp["mask_ratio"] * sal.numel())          # giữ top mask_ratio theo |grad|
        thr = torch.kthvalue(sal, sal.numel() - k + 1).values if k > 0 else sal.max() + 1
        self.masks = [(g.abs() >= thr).to(g.dtype) for g in grads]
        self.mask_frac = float(sum(m_.sum() for m_ in self.masks) / sal.numel())
        m.train(was)

    def _apply_mask(self):
        if self.masks is None:
            return
        for p, mk in zip(self.model.parameters(), self.masks):
            if p.grad is not None:
                p.grad.mul_(mk)

    def step(self):
        h = self.bs // 2
        xf, yf = self.data.batch(self.fc.next()[:h], augment=self.aug)
        xr, yr = self.data.batch(self.rc.next()[:h], augment=self.aug)
        y_rand = (yf + torch.randint(1, self.nc, yf.shape, device=yf.device)) % self.nc   # ≠ nhãn thật
        out = self._fwd(torch.cat([xf, xr]))
        loss = F.cross_entropy(out, torch.cat([y_rand, yr]))
        self._update(loss, self._apply_mask if self.use_mask else None)
        return float(loss.detach())


class SalUn(RandomLabel):
    name = "salun"
    use_mask = True


class Tarun(Unlearner):
    """UNSIR (Tarun et al. 2023). Bước 1..impair_steps: impair (noise đã học + R); sau đó repair (chỉ R)."""
    name = "tarun"

    def prepare(self):
        hp, m, dev = self.hp, self.model, self.device
        was = m.training
        m.eval()
        classes = torch.unique(self.data.ytr[self.split.f])
        n_each = hp["noise_per_class"]
        self.noise_y = classes.repeat_interleave(n_each)
        noise = (0.1 * torch.randn(len(self.noise_y), 3, 32, 32, device=dev)).requires_grad_(True)
        opt = torch.optim.Adam([noise], lr=hp["noise_lr"])
        for _ in range(hp["noise_iters"]):          # tối đa hoá CE trên noise, phạt năng lượng noise
            opt.zero_grad(set_to_none=True)
            for s in range(0, len(noise), 512):
                nz, ny = noise[s:s + 512], self.noise_y[s:s + 512]
                ce = F.cross_entropy(m(nz.contiguous(memory_format=torch.channels_last)), ny)
                loss = (-ce + hp["noise_lam"] * nz.pow(2).mean()) * (len(nz) / len(noise))
                loss.backward(inputs=[noise])
            opt.step()
        self.noise = noise.detach()
        m.zero_grad(set_to_none=True)
        m.train(was)

    def step(self):
        hp = self.hp
        impair = self.t < hp["impair_steps"]
        if not impair and self.opt.param_groups[0]["lr"] != hp["repair_lr"]:
            for g in self.opt.param_groups:
                g["lr"] = hp["repair_lr"]
        if impair:
            h = self.bs // 2
            ni = torch.randint(0, len(self.noise), (min(h, len(self.noise)),), device=self.device)
            xr, yr = self._fb(self.rc)
            x = torch.cat([self.noise[ni].contiguous(memory_format=torch.channels_last), xr[:h]])
            y = torch.cat([self.noise_y[ni], yr[:h]])
        else:
            x, y = self._fb(self.rc)
        loss = F.cross_entropy(self._fwd(x), y)
        self._update(loss)
        return float(loss.detach())


REGISTRY = {"ga": GA, "ngp": NGPlus, "scrub": SCRUB, "salun": SalUn, "rl": RandomLabel,
            "tarun": Tarun, "ft": FT}


def make_unlearner(method, model, theta0_state, data, split, hp, profile, device) -> Unlearner:
    return REGISTRY[method](model, theta0_state, data, split, hp, profile, device)
