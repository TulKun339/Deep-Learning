# 讲义 P24 题目二：设计接受 6400 维输入、产生 10 类输出的神经网络
import torch

batch_size = 4
hidden_size = 128  # 隐藏层宽度可以自行调整

# 网络结构：Linear -> Sigmoid -> Linear -> Softmax
linear1 = torch.nn.Linear(80 * 80, hidden_size)
sigmoid = torch.nn.Sigmoid()
linear2 = torch.nn.Linear(hidden_size, 10)
softmax = torch.nn.Softmax(dim=1)

# 没有语音，用随机数模拟一个 batch，只验证网络的输入和输出
x = torch.randn(batch_size, 80 * 80)
h = sigmoid(linear1(x))
y = softmax(linear2(h))

print("输入形状：", x.shape)
print("隐藏层形状：", h.shape)
print("输出形状：", y.shape)
print("每条样本的类别概率之和：", y.sum(dim=1))
