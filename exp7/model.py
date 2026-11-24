# exp7：P123 练习——2 个卷积 + 1 个双向 LSTM + 1 个分类层
# 卷积部分沿用 exp6 基线的前两层，把后面的 Conv+Linear 换成 BiLSTM + Linear
# LSTM 写法参考 P122：输入形状 (B, T, C)，nn.LSTM 用 batch_first=True；
# P122 的示例是单向的，这里改成 bidirectional=True
import torch
import torch.nn as nn


class ConvBiLSTMNet(nn.Module):
    def __init__(self, num_classes=10, hidden_size=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5),    # 80 -> 76
            nn.ReLU(),
            nn.MaxPool2d(2),                    # 76 -> 38
            nn.Conv2d(16, 32, kernel_size=5),   # 38 -> 34
            nn.ReLU(),
            nn.MaxPool2d(2),                    # 34 -> 17
        )
        # 特征图 (B, 32, 17, 17)：把宽度方向当时间轴，每帧特征 32*17=544
        self.lstm = nn.LSTM(
            input_size=32 * 17,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        # 前向末态 + 后向末态拼接，共 hidden_size*2 维
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.features(x)                        # (B, 32, 17, 17)
        x = x.permute(0, 3, 1, 2)                   # (B, 17, 32, 17) 宽度做时间轴
        x = x.reshape(x.shape[0], x.shape[1], -1)   # (B, 17, 544)
        _, (h_n, _) = self.lstm(x)                  # h_n: (2, B, hidden_size)
        # h_n[0] 是前向走完整句的末态，h_n[1] 是后向走完整句的末态
        h = torch.cat([h_n[0], h_n[1]], dim=1)      # (B, 256)
        return self.classifier(h)                   # logits
