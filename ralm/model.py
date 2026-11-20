"""ResNet-18 biến thể CIFAR (stem 3x3, không maxpool)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + (x if self.short is None else self.short(x)))


class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 10, width: int = 64):
        super().__init__()
        w = width
        self.conv1 = nn.Conv2d(3, w, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(w)
        cfg, layers, cin = [(w, 1), (2 * w, 2), (4 * w, 2), (8 * w, 2)], [], w
        for cout, stride in cfg:
            layers += [BasicBlock(cin, cout, stride), BasicBlock(cout, cout, 1)]
            cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x, return_feat: bool = False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        feat = F.adaptive_avg_pool2d(out, 1).flatten(1)     # penultimate feature (T9)
        logits = self.fc(feat)
        return (logits, feat) if return_feat else logits


def make_model(profile, device) -> ResNet18:
    m = ResNet18(profile.num_classes, profile.width).to(device)
    return m.to(memory_format=torch.channels_last)
