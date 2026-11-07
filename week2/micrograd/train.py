"""micrograd 引擎上的二维二分类训练。

纯 Python，不 import torch。
全批量梯度下降（不做 minibatch），数据是两层 moons。

用法：
    python train.py
    python train.py --lr 0.1 --steps 300 --hidden 16
"""

import argparse
import math
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nn import MLP
from value import Value


# ---------- 数据 ----------

def make_moons(n=100, noise=0.1, seed=0):
    """两层交错的半月。标签是 +1 / -1，配 hinge loss。"""
    # rng: random number generator
    rng = np.random.default_rng(seed)
    n_out = n // 2
    t = np.linspace(0, math.pi, n_out)
    outer = np.stack([np.cos(t), np.sin(t)], axis=1)
    inner = np.stack([1 - np.cos(t), 0.5 - np.sin(t)], axis=1)
    X = np.concatenate([outer, inner], axis=0)
    X += rng.normal(0, noise, X.shape)
    y = np.array([1.0] * n_out + [-1.0] * n_out)
    # 归一化到 [-1, 1]，否则 relu 之后数值尺度失控
    X = (X - X.mean(0)) / X.std(0)
    X = X / np.abs(X).max()
    return X, y


# ---------- 损失 ----------

def hinge_loss(model, X, y):
    """mean(max(0, 1 - y * pred))。全部用现有的 Value 算子拼出来。"""
    losses = []
    for xi, yi in zip(X, y):
        pred = model(xi)[0]                 # MLP 输出是长度为 1 的列表
        losses.append((1 - yi * pred).relu())   # hinge：预测对了且间隔够大就是 0
    return sum(losses, Value(0.0)) * (1.0 / len(X))


def accuracy(model, X, y):
    correct = 0
    for xi, yi in zip(X, y):
        pred = model(xi)[0].data
        correct += (1 if pred > 0 else -1) == yi
    return correct / len(X)


# ---------- 训练 ----------

def train(model, X, y, lr=0.1, steps=200, log_every=20):
    hist = []
    t0 = time.perf_counter()

    for step in range(steps):
        loss = hinge_loss(model, X, y)

        model.zero_grad()      # 必须在 backward 之前：梯度是累加的
        loss.backward()

        for p in model.parameters():
            p.data -= lr * p.grad   # 唯一的更新规则，一行

        hist.append(loss.data)
        if step % log_every == 0 or step == steps - 1:
            acc = accuracy(model, X, y)
            print(f"step {step:4d}  loss {loss.data:.4f}  acc {acc:.3f}")

    return hist, time.perf_counter() - t0


# ---------- 画图 ----------

def plot(model, hist, X, y, elapsed, path="moons.png", grid_n=40):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(hist)
    ax1.set_xlabel("step")
    ax1.set_ylabel("hinge loss")
    ax1.set_title(f"loss  ({elapsed:.2f}s for {len(hist)} steps)")
    ax1.grid(alpha=0.3)

    # 决策边界：在网格上逐点前向，纯 Python，很慢
    t0 = time.perf_counter()
    lo, hi = X.min(0) - 0.2, X.max(0) + 0.2
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], grid_n),
                         np.linspace(lo[1], hi[1], grid_n))
    Z = np.zeros_like(gx)
    for i in range(grid_n):
        for j in range(grid_n):
            Z[i, j] = model(np.array([gx[i, j], gy[i, j]]))[0].data
    grid_time = time.perf_counter() - t0

    ax2.contourf(gx, gy, Z, levels=20, cmap="coolwarm", alpha=0.6)
    ax2.contour(gx, gy, Z, levels=[0], colors="k", linewidths=1.5)
    ax2.scatter(X[y > 0][:, 0], X[y > 0][:, 1], c="tab:red", s=15, label="+1")
    ax2.scatter(X[y < 0][:, 0], X[y < 0][:, 1], c="tab:blue", s=15, label="-1")
    ax2.set_title(f"decision boundary  ({grid_n**2} fwd = {grid_time:.2f}s)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")
    return grid_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    X, y = make_moons(args.n, seed=args.seed)
    model = MLP(2, [args.hidden, args.hidden, 1])
    print(f"model: {model}")
    print(f"params: {len(model.parameters())}\n")

    hist, elapsed = train(model, X, y, lr=args.lr, steps=args.steps)
    plot(model, hist, X, y, elapsed)

    print(f"\ntotal train time   {elapsed:.2f} s")
    print(f"per step           {elapsed / args.steps * 1e3:.1f} ms  ({args.n} samples)")
