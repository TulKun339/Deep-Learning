# exp7：在模拟数据上跑通"2 卷积 + 双向 LSTM + 分类"
# 数据是随机数：每类一个随机模板 + 噪声（与 exp3/exp4 的假数据一致），不读真实录音
import copy
import os

import torch
from torch.utils.data import DataLoader, TensorDataset

from model import ConvBiLSTMNet

torch.manual_seed(0)

# ---------- 超参数 ----------
num_classes = 10
side = 80
batch_size = 32
lr = 1e-3
epochs = 10

# ---------- 造模拟数据：每个类一个随机模板 ----------
centroids = torch.randn(num_classes, 1, side, side)


def make_dataset(n_per_class, noise=0.5):
    xs, ys = [], []
    for c in range(num_classes):
        x = centroids[c].unsqueeze(0).repeat(n_per_class, 1, 1, 1)
        xs.append(x + noise * torch.randn_like(x))
        ys.append(torch.full((n_per_class,), c, dtype=torch.long))
    return torch.cat(xs), torch.cat(ys)


x_train, y_train = make_dataset(100)  # 每类 100 条，共 1000 条
x_test, y_test = make_dataset(20)     # 每类 20 条，共 200 条
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size)

# ---------- 模型 ----------
model = ConvBiLSTMNet(num_classes=num_classes)
print("输入形状：", tuple(x_train.shape))
print("参数量：", sum(p.numel() for p in model.parameters()))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

# ---------- 训练 ----------
best_acc, best_state = 0.0, None
for epoch in range(1, epochs + 1):
    model.train()
    total_loss, n = 0.0, 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    acc = correct / total

    if acc > best_acc:
        best_acc = acc
        best_state = copy.deepcopy(model.state_dict())

    print(f"epoch {epoch:2d}  loss={total_loss / n:.4f}  test_acc={acc:.4f}")

# ---------- 保存 ----------
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pth")
torch.save({"state_dict": best_state, "num_classes": num_classes}, save_path)
print(f"最佳准确率 {best_acc:.4f}，模型已保存到 {save_path}")
