# 自建网络：只用 Conv2d + Linear（外加 ReLU / 池化）
# 输入 (B, 1, 80, 80) -> 输出 (B, 63)：63 句的分类概率
import torch.nn as nn


class QianziwenNet(nn.Module):
    def __init__(self, num_classes=63):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5),    # 80 -> 76
            nn.ReLU(),
            nn.MaxPool2d(2),                    # 76 -> 38
            nn.Conv2d(16, 32, kernel_size=5),   # 38 -> 34
            nn.ReLU(),
            nn.MaxPool2d(2),                    # 34 -> 17
            nn.Conv2d(32, 64, kernel_size=3),   # 17 -> 15
            nn.ReLU(),
            nn.MaxPool2d(2),                    # 15 -> 7
        )
        # 64 * 7 * 7 = 3136
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.shape[0], -1)
        return self.classifier(x)
