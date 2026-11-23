# 11/10 任务：63 句《千字文》整句识别
# 用 Dataset + DataLoader 训练，网络只用 Conv2d + Linear，全程 CPU
#
# 用法：
#   python train.py [录音目录]
#   默认读取 exp5/千字文20261103（含 recording_mapping.txt）
import copy
import os
import sys

import torch
from torch.utils.data import DataLoader, random_split

from dataset import FakeQianziwen, Qianziwen
from model import QianziwenNet

root = os.path.dirname(os.path.abspath(__file__))
data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "千字文20261103")

device = torch.device("cpu")
torch.manual_seed(0)

# ---------- 数据 ----------
if os.path.isdir(data_dir):
    dataset = Qianziwen(data_dir, cache_dir=os.path.join(root, "features_cache"))
    classes = dataset.classes
    epochs = 15
    print(f"真实数据：{len(dataset)} 条，{len(classes)} 类")
    print("开始缓存特征（首次较慢，之后直接读缓存）...")
    print("新缓存：", dataset.build_cache(), "条")
else:
    dataset = FakeQianziwen(num_classes=63)
    classes = [f"第{i}句" for i in range(63)]
    epochs = 5
    print(f"未找到 {data_dir}，改用假数据验证流程（准确率无意义）")

num_classes = len(classes)
n_train = int(len(dataset) * 0.8)
train_set, test_set = random_split(
    dataset,
    [n_train, len(dataset) - n_train],
    generator=torch.Generator().manual_seed(0),
)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)
print(f"训练/测试：{len(train_set)} / {len(test_set)}")

# ---------- 模型：Conv2d + Linear ----------
model = QianziwenNet(num_classes=num_classes).to(device)
print("参数量：", sum(p.numel() for p in model.parameters()))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))

# ---------- 训练 ----------
best_acc = 0.0
best_state = None
for epoch in range(1, epochs + 1):
    model.train()
    total_loss, n = 0.0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
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
            pred = model(xb.to(device)).argmax(dim=1).cpu()
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    acc = correct / total

    if acc > best_acc:
        best_acc = acc
        best_state = copy.deepcopy(model.state_dict())

    print(f"epoch {epoch:2d}  loss={total_loss / n:.4f}  test_acc={acc:.4f}")

# ---------- 保存 ----------
save_path = os.path.join(root, "best_model.pth")
torch.save({"state_dict": best_state, "classes": classes}, save_path)
print(f"最佳准确率 {best_acc:.4f}，模型已保存到 {save_path}")
