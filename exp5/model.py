# 讲义 P74：自己创造的网络，同时使用 Conv2d 和 Linear
# 设计：两组卷积+池化抽特征 → 拉平 → 两层全连接分类
# 输入：(B, 1, 80, 80)；输出：(B, 10)
import torch
import torch.nn as nn


class MyNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 1*80*80 -> 16*76*76 -> 池化 -> 16*38*38
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        # 16*38*38 -> 32*34*34 -> 池化 -> 32*17*17
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        # 32*17*17 = 9248
        self.linear1 = nn.Linear(32 * 17 * 17, 512)
        self.linear2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


if __name__ == "__main__":
    m = MyNet()
    print("参数量：", sum(p.numel() for p in m.parameters()))
    dummy = torch.randn(4, 1, 80, 80)
    out = m(dummy)
    print("输入：", dummy.shape, "→ 输出：", out.shape)
