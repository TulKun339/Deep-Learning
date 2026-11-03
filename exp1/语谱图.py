# ---------- ① 读文件 ----------
import torchaudio

x, fs = torchaudio.load('e.wav')
# x = 波形张量，形状 [通道数, 采样点数]。单声道就是 [1, 32000]
# fs = 采样率，每秒采多少个点。fs=16000 时，32000 点 = 2 秒

# ---------- ② 造一台"机器"（还不算） ----------
spec = torchaudio.transforms.Spectrogram(1024)
# 1024 是 n_fft：每次分析多长的一段。
# 这句只是把机器配置好，还没碰数据。

# ---------- ③ 开动机器 ----------
y = spec(x)
# 形状 [1, 513, 128]：1 个样本 × 513 个频率 × 128 个时间帧
# 这就是"语谱图"本身，只不过还是裸数字。

# ---------- ④ 画图 ----------
import matplotlib
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(20, 10))
# fig = 整张画布，ax = 画布上的一个坐标系。1,1 = 画 1 行 1 列

im = ax.imshow(
    y[0].log().numpy(),
    # y[0]      去掉 batch 维 → [513, 128]
    # .log()    取对数，把百万倍的能量差距压到屏幕上看得见
    # .numpy()  torch 张量 → numpy 数组，matplotlib 只认后者
    aspect="auto",    # 不保持宽高比，让图铺满整个框
    origin="lower",   # 第 0 行画在底部，否则频率轴上下颠倒
)
print(x.shape, fs, y.shape)

fig.canvas.draw()          # 先渲染一次（某些版本不这样会存出空白图）
fig.savefig('e.png')   # 存文件
plt.close()                # 关闭，释放内存
