# 讲义 P73：自定义 Dataset
# 目前没有真实语音，用带固定模板的随机假数据生成 (80,80) 特征图
# 以后换真实语音时，只需把 __getitem__ 里的假数据替换成：
#   torchaudio.load → 单声道 → 重采样 → MelSpectrogram → Resize
import torch
from torch.utils.data import Dataset


num_classes = 10
feature_shape = (1, 80, 80)


class MyData(Dataset):
    def __init__(self, n_per_class=200, noise=0.5):
        # 为每个类别造一个固定模板，样本 = 模板 + 噪声
        torch.manual_seed(42)
        self.templates = torch.randn(num_classes, *feature_shape)
        self.ys = []
        for c in range(num_classes):
            self.ys += [c] * n_per_class
        self.ys = torch.tensor(self.ys, dtype=torch.long)
        self.noise = noise

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        y = self.ys[idx]
        x = self.templates[y] + self.noise * torch.randn(*feature_shape)
        return x, y


if __name__ == "__main__":
    ds = MyData(n_per_class=200)
    print("数据集大小：", len(ds))
    x, y = ds[0]
    print("单条样本：", x.shape, "标签 =", y)
