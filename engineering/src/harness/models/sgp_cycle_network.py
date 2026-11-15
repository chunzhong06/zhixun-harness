"""同一物理加热周期的四路原始响应归一化及浓度回归网络。"""

import torch
from torch import nn


class CycleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 12, (1, 15), stride=(1, 6)), nn.BatchNorm2d(12), nn.ReLU(),
            nn.Conv2d(12, 16, (4, 7), stride=(1, 4)), nn.BatchNorm2d(16), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(16 * 58, 48), nn.ReLU(), nn.Dropout(.1), nn.Linear(48, 1))

    def forward(self, x):
        return self.head(self.encoder(x)).flatten()


def normalize(raw):
    x = torch.tensor(raw, dtype=torch.float32)[:, None]
    return (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True, unbiased=False).clamp_min(1e-8)
