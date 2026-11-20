"""Dữ liệu: giữ cả tập train/test dạng uint8 trên GPU, augmentation chạy trên GPU.

Quy ước chỉ mục: mọi index (D, V, F, R) đều là chỉ mục vào mảng train gốc (0..N-1).
Split cố định theo §3.1: perm = RandomState(12345).permutation(N); V = perm[-5000:], D = perm[:-5000].
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .config import SPLIT_SEED, Profile

STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def _fake(n: int, num_classes: int, seed: int):
    """Dữ liệu giả học được (chỉ để test logic trên CPU)."""
    rs = np.random.RandomState(seed)
    proto = np.random.RandomState(999).randint(0, 255, (num_classes, 32, 32, 3)).astype(np.float32)
    y = rs.randint(0, num_classes, n)
    x = 0.45 * proto[y] + 0.55 * rs.randint(0, 255, (n, 32, 32, 3)).astype(np.float32)
    return np.clip(x, 0, 255).astype(np.uint8), y.astype(np.int64)


def load_raw(profile: Profile):
    if profile.fake_train > 0:
        xtr, ytr = _fake(profile.fake_train, profile.num_classes, 1)
        xte, yte = _fake(profile.fake_test, profile.num_classes, 2)
        return xtr, ytr, xte, yte
    import torchvision
    cls = {"cifar10": torchvision.datasets.CIFAR10, "cifar100": torchvision.datasets.CIFAR100}[profile.dataset]
    try:
        tr = cls(profile.data_root, train=True, download=True)
        te = cls(profile.data_root, train=False, download=True)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Không tải được {profile.dataset} vào {profile.data_root}: {e}\n"
            "Kaggle: bật Internet, hoặc Add Data một dataset chứa thư mục cifar-10-batches-py "
            "rồi chạy với --data_root <thư mục cha của nó>.") from e
    return (np.asarray(tr.data, dtype=np.uint8), np.asarray(tr.targets, dtype=np.int64),
            np.asarray(te.data, dtype=np.uint8), np.asarray(te.targets, dtype=np.int64))


class Data:
    def __init__(self, profile: Profile, device: torch.device):
        self.profile, self.device = profile, device
        xtr, ytr, xte, yte = load_raw(profile)
        self.ytr_np, self.yte_np = ytr, yte
        to_dev = lambda a: torch.from_numpy(a).permute(0, 3, 1, 2).contiguous().to(device)  # noqa: E731
        self.xtr, self.xte = to_dev(xtr), to_dev(xte)
        self.ytr, self.yte = torch.from_numpy(ytr).to(device), torch.from_numpy(yte).to(device)
        mean, std = STATS[profile.dataset]
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        perm = np.random.RandomState(SPLIT_SEED).permutation(len(xtr))
        self.v_idx_np = np.sort(perm[-profile.v_size:])
        self.d_idx_np = np.sort(perm[:-profile.v_size])
        assert len(np.intersect1d(self.v_idx_np, self.d_idx_np)) == 0
        self.n_train = len(xtr)

    # --- chuẩn hoá / augment (augmentation CHỈ khi train — §1.5)
    def prep(self, xb_u8: torch.Tensor) -> torch.Tensor:
        x = (xb_u8.float() / 255.0 - self.mean) / self.std
        return x.contiguous(memory_format=torch.channels_last)

    def augment(self, xb_u8: torch.Tensor) -> torch.Tensor:
        """RandomCrop(32, pad=4) + RandomHorizontalFlip, trên GPU, per-sample."""
        b, dev = xb_u8.shape[0], xb_u8.device
        xp = F.pad(xb_u8, (4, 4, 4, 4))
        i = torch.randint(0, 9, (b,), device=dev)
        j = torch.randint(0, 9, (b,), device=dev)
        ar = torch.arange(32, device=dev)
        rows = (i[:, None] + ar)[:, :, None]
        cols = (j[:, None] + ar)[:, None, :]
        out = xp[torch.arange(b, device=dev)[:, None, None], :, rows, cols].permute(0, 3, 1, 2)
        flip = torch.rand(b, device=dev) < 0.5
        return torch.where(flip[:, None, None, None], out.flip(3), out)

    def batch(self, idx: torch.Tensor, augment: bool):
        x = self.xtr[idx]
        if augment:
            x = self.augment(x)
        return self.prep(x), self.ytr[idx]


class Split:
    """Chỉ mục D, V, F, R dạng tensor trên device. F ⊂ D, R = D \\ F."""

    def __init__(self, data: Data, f_idx_np: np.ndarray):
        dev = data.device
        f = np.sort(np.asarray(f_idx_np, dtype=np.int64))
        assert len(np.intersect1d(f, data.v_idx_np)) == 0, "F ∩ V ≠ ∅"
        assert np.isin(f, data.d_idx_np).all(), "F ⊄ D"
        r = np.setdiff1d(data.d_idx_np, f)
        assert len(r) + len(f) == len(data.d_idx_np) and len(np.intersect1d(r, f)) == 0
        self.f_np, self.r_np, self.d_np, self.v_np = f, r, data.d_idx_np, data.v_idx_np
        t = lambda a: torch.from_numpy(a).to(dev)  # noqa: E731
        self.f, self.r, self.d, self.v = t(f), t(r), t(self.d_np), t(self.v_np)
        self.rho = len(f) / len(self.d_np)
