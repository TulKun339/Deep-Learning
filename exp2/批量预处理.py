# ---------- exp2：在 exp1 代码基础上完成讲义 P18 ----------
# exp1：读一条语音 → 生成语谱图 → 画图
# exp2：遍历多条语音 → 重采样 → 生成语谱图 → 统一大小 → 保存 pth

import os

import soundfile
import torch
import torchaudio
import torchvision

wav_dir = os.path.join(os.path.dirname(__file__), "..", "exp1")
out_dir = os.path.join(os.path.dirname(__file__), "features")
os.makedirs(out_dir, exist_ok=True)

# ---------- ① 造一台“机器”（沿用 exp1） ----------
spec = torchaudio.transforms.Spectrogram(1024)
resize = torchvision.transforms.Resize((80, 80), antialias=True)

# ---------- ② 遍历目录下所有语音（exp1 只读一条） ----------
for filename in os.listdir(wav_dir):
    if not filename.lower().endswith(".wav"):
        continue
    path = os.path.join(wav_dir, filename)

    # ---------- ③ 读文件（沿用 exp1） ----------
    try:
        x, fs = torchaudio.load(path)
    except (ImportError, RuntimeError):
        # 本机 torchaudio 缺少 FFmpeg 共享库时，用 soundfile 读出同样的数据
        data, fs = soundfile.read(path, dtype="float32")
        x = torch.from_numpy(data).T

    # ---------- ④ 重采样（exp2 新增） ----------
    if x.shape[0] > 1:                 # 双声道合成单声道
        x = x.mean(dim=0, keepdim=True)
    if fs != 16000:
        x = torchaudio.transforms.Resample(fs, 16000)(x)
        fs = 16000

    # ---------- ⑤ 开动机器（沿用 exp1） ----------
    y = spec(x)
    y = y.clamp_min(1e-10).log()       # clamp 防止 log(0) = -inf
    print(filename, x.shape, fs, y.shape)

    # ---------- ⑥ 统一大小并保存（替代 exp1 的画图） ----------
    y = resize(y)
    out_path = os.path.join(out_dir, os.path.splitext(filename)[0] + ".pth")
    torch.save(y, out_path)
    print("保存：", out_path, y.shape)
