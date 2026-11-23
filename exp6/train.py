# exp6：对比三种卷积的参数量和准确率
# 复用 exp5 的 Dataset + DataLoader（cache 也在 exp5 目录）
# 三个模型结构完全对齐：输入 (B,1,80,80)、输出 (B,63)
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, random_split

# ---------- 复用 exp5 的数据集代码 ----------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "exp5"))
from dataset import Qianziwen

from baseline import BaselineNet
from spatial import SpatialSeparableNet
from depthwise import DepthwiseSeparableNet

root = os.path.dirname(os.path.abspath(__file__))
# 录音在 exp5/千字文20261103，缓存也复用 exp5 的 features_cache
exp5_dir = os.path.join(root, "..", "exp5")
data_dir = os.path.join(exp5_dir, "千字文20261103")
cache_dir = os.path.join(exp5_dir, "features_cache")

device = torch.device("cpu")
torch.manual_seed(0)

# ---------- 数据 ----------
dataset = Qianziwen(data_dir, cache_dir=cache_dir)
print(f"数据集：{len(dataset)} 条，{len(dataset.classes)} 类")

n_train = int(len(dataset) * 0.8)
train_set, test_set = random_split(
    dataset,
    [n_train, len(dataset) - n_train],
    generator=torch.Generator().manual_seed(0),
)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)
print(f"训练/测试：{len(train_set)} / {len(test_set)}")


def train_one(name, model, epochs=10, lr=1e-3):
    """训练单个模型并返回 (最佳准确率, 参数量, 总耗时)"""
    print(f"\n{'='*50}")
    print(f"模型：{name}    参数量：{sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*50}")

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    crit = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
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
            import copy
            best_state = copy.deepcopy(model.state_dict())
        print(f"epoch {epoch:2d}  loss={total_loss/n:.4f}  test_acc={acc:.4f}")

    elapsed = time.time() - t0
    save_path = os.path.join(root, f"{name}_best.pth")
    torch.save({"state_dict": best_state}, save_path)
    print(f"最佳准确率 {best_acc:.4f}，耗时 {elapsed:.0f}s，已保存 {save_path}")
    return best_acc, sum(p.numel() for p in model.parameters()), elapsed


# ---------- 跑三个模型 ----------
results = []
for name, ModelClass in [
    ("baseline", BaselineNet),
    ("spatial", SpatialSeparableNet),
    ("depthwise", DepthwiseSeparableNet),
]:
    acc, params, elapsed = train_one(name, ModelClass(num_classes=63), epochs=10)
    results.append((name, params, acc, elapsed))

# ---------- 汇总 ----------
print("\n" + "=" * 65)
print(f"{'模型':<12}{'参数量':>12}{'相对基线':>10}{'最佳准确率':>10}{'耗时(s)':>10}")
print("-" * 65)
base_params = results[0][1]
for name, params, acc, elapsed in results:
    print(f"{name:<12}{params:>12,}{params/base_params:>9.1%}{acc:>10.4f}{elapsed:>10.0f}")
print("=" * 65)
