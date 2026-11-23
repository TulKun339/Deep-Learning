# 讲义 P75 + P76：DataLoader + 自定义网络训练
# 拆成了三个文件：dataset.py（数据）、model.py（网络）、train.py（训练）
import torch
from torch.utils.data import DataLoader, random_split

from dataset import MyData
from model import MyNet

torch.manual_seed(0)

# ---------- 构造 Dataset 和 DataLoader ----------
full_ds = MyData(n_per_class=200)   # 共 2000 条

# 80% 训练 / 20% 测试
train_size = int(len(full_ds) * 0.8)
test_size = len(full_ds) - train_size
train_ds, test_ds = random_split(full_ds, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

print("训练/测试：", train_size, "/", test_size)

# ---------- 网络、损失、优化器 ----------
model = MyNet(num_classes=10)
print("网络参数量：", sum(p.numel() for p in model.parameters()))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))

# ---------- 训练循环 ----------
epochs = 5
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    n = 0
    for xb, yb in train_loader:        # DataLoader 自动分批 + 打乱
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
    train_loss = total_loss / n

    # 测试
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    acc = correct / total

    print(f"epoch {epoch:2d}  loss={train_loss:.4f}  test_acc={acc:.4f}")
