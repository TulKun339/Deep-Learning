# 11/10 任务：给定一句《千字文》朗读，判断是 63 句中的哪一句
# 数据来源：千字文20261103/recording_mapping.txt（含 sentence_index 与 audio_file）
# 录音是 webm，用 ffmpeg 解码成 16 kHz 单声道
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile
import torch
import torchaudio
import torchvision
from torch.utils.data import Dataset

# ---------- 特征提取（讲义第七节：梅尔谱）----------
mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,        # 25 ms 一帧
    hop_length=160,   # 10 ms 帧移
    n_mels=24,        # 讲义用的 24 组梅尔滤波器
)
resize = torchvision.transforms.Resize((80, 80), antialias=True)


def decode_audio(path):
    """读任意格式录音 -> 16 kHz 单声道波形"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".wav", ".flac"):
        try:
            x, fs = torchaudio.load(path)
        except (ImportError, RuntimeError):
            data, fs = soundfile.read(path, dtype="float32")
            x = torch.from_numpy(data).T
        if x.shape[0] > 1:
            x = x.mean(dim=0, keepdim=True)
        if fs != 16000:
            x = torchaudio.transforms.Resample(fs, 16000)(x)
        return x

    # webm 等格式交给 ffmpeg：直接输出 16 kHz 单声道 float32
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path,
         "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
        capture_output=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"ffmpeg 解码失败：{path}\n{out.stderr.decode(errors='ignore')}")
    x = torch.from_numpy(np.frombuffer(out.stdout, dtype=np.float32).copy())
    return x.unsqueeze(0)


def load_feature(path):
    """一条录音 -> (1, 80, 80) 的 log 梅尔谱"""
    x = decode_audio(path)
    y = mel(x).clamp_min(1e-10).log()
    return resize(y)


class Qianziwen(Dataset):
    """按 recording_mapping.txt 组织的《千字文》数据集，标签是 sentence_index。"""

    def __init__(self, data_dir, cache_dir=None):
        mapping_path = os.path.join(data_dir, "recording_mapping.txt")
        with open(mapping_path, encoding="utf-8") as f:
            records = json.load(f)["recordings"]

        self.items = []
        texts = {}
        for r in records:
            audio = os.path.join(data_dir, "recordings", r["audio_file"].replace("/", os.sep))
            self.items.append((audio, r["sentence_index"]))
            texts[r["sentence_index"]] = r["original_text"]

        self.classes = [texts[i] for i in sorted(texts)]
        self.cache_dir = cache_dir or os.path.join(data_dir, "features_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.items)

    def cache_path(self, path):
        key = path.split(os.sep + "recordings" + os.sep)[-1].replace(os.sep, "__")
        return os.path.join(self.cache_dir, os.path.splitext(key)[0] + ".pt")

    def __getitem__(self, idx):
        path, label = self.items[idx]
        cache = self.cache_path(path)
        if os.path.exists(cache):
            x = torch.load(cache)
        else:
            try:
                x = load_feature(path)
                torch.save(x, cache)
            except Exception as e:                   # 少数录音文件损坏
                print("读取失败，用空特征代替：", path, e)
                x = torch.zeros(1, 80, 80)
        return x, label

    def build_cache(self, workers=8):
        """首次运行把所有录音解码成特征存盘，后面的 epoch 就很快"""
        todo = [(p, self.cache_path(p)) for p, _ in self.items if not os.path.exists(self.cache_path(p))]
        if not todo:
            return 0

        def work(item):
            path, cache = item
            try:
                torch.save(load_feature(path), cache)
                return True
            except Exception as e:                       # 个别文件坏了就跳过
                print("跳过", path, e)
                return False

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for ok in pool.map(work, todo):
                done += ok
                if done % 500 == 0:
                    print(f"已缓存 {done}/{len(todo)}")

        # 损坏的录音没有缓存，直接从数据集里去掉
        before = len(self.items)
        self.items = [it for it in self.items if os.path.exists(self.cache_path(it[0]))]
        if len(self.items) < before:
            print(f"跳过损坏录音 {before - len(self.items)} 条")
        return done


class FakeQianziwen(Dataset):
    """没有录音时的占位数据集：63 类，每类一个固定模板 + 噪声。"""

    def __init__(self, num_classes=63, n_per_class=20, noise=0.5):
        torch.manual_seed(42)
        self.templates = torch.randn(num_classes, 1, 80, 80)
        self.ys = torch.arange(num_classes).repeat_interleave(n_per_class)
        self.noise = noise

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        y = int(self.ys[idx])
        x = self.templates[y] + self.noise * torch.randn(1, 80, 80)
        return x, y
