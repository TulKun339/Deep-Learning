# 讲义 P100：空间分离卷积（无中间 ReLU，无 padding 保持与 baseline 一致的收缩）
# Conv(c_in, c_out, k×k) -> Conv(c_in, c_out, k×1) → Conv(c_out, c_out, 1×k)
# 两层都 padding=0：k×1 收缩高 k-1，1×k 收缩宽 k-1，总收缩 (k-1,k-1) 与基线相同
import torch.nn as nn


class SpatialSeparableConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size):
        super().__init__()
        # 正确的空间分离：c_in→c_in（垂直）→c_out（水平）
        # 第一层沿垂直方向聚合，不改变通道数；第二层沿水平方向聚合，同时把通道从 c_in 变成 c_out
        # 和讲义 P100 的伪代码不同，讲义是教学示意（c_in→c_out→c_out），
        # 这里的实现才是真正省参数的秩1分解思路
        k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.conv1 = nn.Conv2d(c_in,  c_in,  kernel_size=(k, 1), padding=0)   # 垂直方向聚合，通道不变
        self.conv2 = nn.Conv2d(c_in,  c_out, kernel_size=(1, k), padding=0)   # 水平方向聚合，通道升/降到目标 c_out

    def forward(self, x):
        return self.conv2(self.conv1(x))


class SpatialSeparableNet(nn.Module):
    def __init__(self, num_classes=63):
        super().__init__()
        # 每一层都 k-1 的收缩，和 baseline 完全对齐
        self.features = nn.Sequential(
            SpatialSeparableConv(1,  16, kernel_size=5),   # 80 -> 76
            nn.ReLU(),
            nn.MaxPool2d(2),                                 # -> 38
            SpatialSeparableConv(16, 32, kernel_size=5),   # 38 -> 34
            nn.ReLU(),
            nn.MaxPool2d(2),                                 # -> 17
            SpatialSeparableConv(32, 64, kernel_size=3),   # 17 -> 15
            nn.ReLU(),
            nn.MaxPool2d(2),                                 # -> 7
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
