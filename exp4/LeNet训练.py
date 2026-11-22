# 讲义 P56：把 exp3 的全连接网络换成 LeNet
# 结构：Conv2d -> MaxPool -> Conv2d -> MaxPool -> Reshape -> Linear -> Softmax
# 仍用带规律的随机假数据，输入形状改为 (B, 1, 80, 80)
import torch

torch.manual_seed(0)

# ---------- 超参数 ----------
num_classes = 10
channels = 1
side = 80
feature_shape = (channels, side, side)  # 与 exp2 输出的 (1,80,80) 对齐
batch_size = 32
lr = 1e-3
epochs = 5

# ---------- 造带规律的假数据（每类一个固定模板 + 噪声）----------
centroids = torch.randn(num_classes, *feature_shape)


def make_dataset(n_per_class, noise=0.5):
    xs, ys = [], []
    for c in range(num_classes):
        x = centroids[c].unsqueeze(0).repeat(n_per_class, 1, 1, 1)
        x = x + noise * torch.randn_like(x)
        xs.append(x)
        ys.append(torch.full((n_per_class,), c, dtype=torch.long))
    return torch.cat(xs), torch.cat(ys)


x_train, y_train = make_dataset(100)  # 1000 条
x_test, y_test = make_dataset(20)     # 200 条

# ---------- LeNet ----------
class LeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 1*80*80 -> 6*76*76 -> 池化 6*38*38
        self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=5)
        self.pool = torch.nn.MaxPool2d(2)
        # 6*38*38 -> 16*34*34 -> 池化 16*17*17
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=5)
        # 16*17*17 = 4624，拉平后接全连接
        self.linear = torch.nn.Linear(16 * 17 * 17, num_classes)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.reshape(x.shape[0], -1)  # (B, 4624)
        x = self.linear(x)             # logits，Softmax 交给 CrossEntropyLoss
        return x


model = LeNet()
print("LeNet 参数量：", sum(p.numel() for p in model.parameters()))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

# ---------- 训练循环（与 exp3 相同）----------
for epoch in range(1, epochs + 1):
    model.train()
    perm = torch.randperm(x_train.size(0))
    total_loss = 0.0
    for i in range(0, x_train.size(0), batch_size):
        idx = perm[i:i + batch_size]
        xb, yb = x_train[idx], y_train[idx]

        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
    train_loss = total_loss / x_train.size(0)

    model.eval()
    with torch.no_grad():
        pred = model(x_test).argmax(dim=1)
        acc = (pred == y_test).float().mean().item()

    print(f"epoch {epoch:2d}  loss={train_loss:.4f}  test_acc={acc:.4f}")
