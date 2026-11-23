# 讲义 P101：深度可分离卷积（Depthwise + Pointwise，无中间 ReLU）
# Conv(c_in, c_out, k×k) ->
#   Conv(c_in, c_in,   k×k, groups=c_in)    # Depthwise：每个通道独立卷
#   Conv(c_in, c_out, 1×1)                   # Pointwise：1×1 改通道数
# 两层都 padding=0：DW 收缩 k-1，PW 不收缩，总收缩 k-1 与基线相同
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size):
        super().__init__()
        k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.dw = nn.Conv2d(c_in, c_in, kernel_size=k, padding=0, groups=c_in)
        self.pw = nn.Conv2d(c_in, c_out, kernel_size=1, padding=0)

    def forward(self, x):
        return self.pw(self.dw(x))


class DepthwiseSeparableNet(nn.Module):
    def __init__(self, num_classes=63):
        super().__init__()
        self.features = nn.Sequential(
            DepthwiseSeparableConv(1,  16, kernel_size=5),   # 80 -> 76
            nn.ReLU(),
            nn.MaxPool2d(2),                                   # -> 38
            DepthwiseSeparableConv(16, 32, kernel_size=5),   # 38 -> 34
            nn.ReLU(),
            nn.MaxPool2d(2),                                   # -> 17
            DepthwiseSeparableConv(32, 64, kernel_size=3),   # 17 -> 15
            nn.ReLU(),
            nn.MaxPool2d(2),                                   # -> 7
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.shape[0], -1)
        return self.classifier(x)
