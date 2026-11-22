# 讲义 P35 题目二：用 Adam 优化器求解神经网络参数
# 没有真实语音，用带可学习模式的随机数据把训练框架搭起来
import torch

torch.manual_seed(0)

# ---------- 超参数 ----------
num_classes = 10
feature_dim = 80 * 80      # 6400，与 exp2 输出的特征维度对齐
hidden_size = 128
batch_size = 32
lr = 1e-3
epochs = 10

# ---------- 造一份带规律的假数据（让 loss 能真正下降，验证框架可用）----------
# 思路：给每个数字固定一个“模板”特征，样本 = 模板 + 噪声
centroids = torch.randn(num_classes, feature_dim)


def make_dataset(n_per_class, noise=0.5):
    xs, ys = [], []
    for c in range(num_classes):
        x = centroids[c].unsqueeze(0).repeat(n_per_class, 1)
        x = x + noise * torch.randn_like(x)
        xs.append(x)
        ys.append(torch.full((n_per_class,), c, dtype=torch.long))
    return torch.cat(xs), torch.cat(ys)


x_train, y_train = make_dataset(200)  # 每类 200 条，共 2000 条
x_test, y_test = make_dataset(50)     # 每类 50 条，共 500 条

# ---------- 网络（与 exp2/分类网络.py 同结构：Linear -> Sigmoid -> Linear）----------
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(feature_dim, hidden_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.linear2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.linear2(x)  # 输出 logits，Softmax 交给 CrossEntropyLoss 内部完成
        return x


model = Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))  # 讲义里的 Adam 参数

# ---------- 训练循环 ----------
for epoch in range(1, epochs + 1):
    # 训练
    model.train()
    perm = torch.randperm(x_train.size(0))
    total_loss = 0.0
    for i in range(0, x_train.size(0), batch_size):
        idx = perm[i:i + batch_size]
        xb, yb = x_train[idx], y_train[idx]

        optimizer.zero_grad()      # 1. 清空上一步的梯度
        out = model(xb)            # 2. 前向传播
        loss = criterion(out, yb)  # 3. 算损失
        loss.backward()            # 4. 反向传播，算梯度
        optimizer.step()           # 5. Adam 更新参数

        total_loss += loss.item() * xb.size(0)
    train_loss = total_loss / x_train.size(0)

    # 测试集准确率
    model.eval()
    with torch.no_grad():
        pred = model(x_test).argmax(dim=1)
        acc = (pred == y_test).float().mean().item()

    print(f"epoch {epoch:2d}  loss={train_loss:.4f}  test_acc={acc:.4f}")
