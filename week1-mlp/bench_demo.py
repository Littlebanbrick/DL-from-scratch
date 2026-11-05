"""perf/bench.py 用法示范。

跑法（在 week1-mlp/ 目录下）：
    python bench_demo.py

数据是随机生成的，不下载数据集，十秒内跑完。
"""

import pathlib
import statistics
import sys
import time

# perf 在仓库根目录，不在 week1-mlp/ 里，所以要把它加进 import 路径
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from perf.bench import bench, device_info, peak_mem_mb, reset_peak_mem, set_threads


class MLP(nn.Module):
    def __init__(self, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 10),
        )

    def forward(self, x):
        return self.net(x)


torch.manual_seed(0)
set_threads(8)                      # 第一步永远是钉线程数，否则数字不可信
model = MLP(hidden=512)
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=1e-3)

N, B = 8192, 64                     # 假装这是 8192 张 28x28 的图
fake_x = torch.randn(N, 784)
fake_y = torch.randint(0, 10, (N,))
loader = DataLoader(TensorDataset(fake_x, fake_y), batch_size=B, shuffle=True)

xb, yb = next(iter(loader))
print(f"device: {device_info()}")
print(f"batch : x{tuple(xb.shape)}  y{tuple(yb.shape)}\n")


# ---------- 1. 怎么用 bench ----------
# 传一个「零参数的可调用对象」进去，它内部负责 warmup、跑 n 次、取中位数和最小值。

def one_step():
    """一次完整训练步：清梯度 -> 前向 -> loss -> 反向 -> 更新。"""
    optim.zero_grad()
    loss = loss_fn(model(xb), yb)
    loss.backward()
    optim.step()
    return loss


bench(lambda: model(xb), label="forward (no grad)")        # 单语句用 lambda
bench(one_step, label="train step (fwd+bwd+upd)")          # 多语句用 def
bench(one_step, label="train step, n=50", n=50)            # 多测几次更稳

reset_peak_mem()
bench(one_step, label="train step, with peak mem")
print(f"{'peak mem':<28} {peak_mem_mb()}   (always None on CPU)\n")


# ---------- 2. 为什么不能拿 bench 测整个 epoch ----------
# bench 会把同一个函数重复跑 n 次，前提是「跑多少次结果都一样」。
# epoch 有状态：参数每步都在变。所以 epoch 用 time.perf_counter 手掐。

t0 = time.perf_counter()
for X, y in loader:
    pass
print(f"{'dataloader only (pass)':<28} total {time.perf_counter() - t0:8.3f} s   ({len(loader)} batches)\n")


# ---------- 3. 线程数会毁掉中位数 ----------
# 这是本机实测：默认 16 线程时，同一份代码的中位数能在 1 ms 和 90 ms 之间跳。
# 最小值反而稳定 —— 这就是「最小值报性能，中位数判优劣」的由来。

def spread(threads, repeats=3):
    set_threads(threads)
    meds, mins = [], []
    for _ in range(repeats):
        r = bench(one_step, warmup=5, n=20, label=f"  threads={threads}")
        meds.append(r["median_ms"])
        mins.append(r["min_ms"])
    return meds, mins


for t in (16, 8):
    meds, mins = spread(t)
    print(f"threads={t:2d}  median {[round(m, 1) for m in meds]}  min {[round(m, 2) for m in mins]}"
          f"   spread {max(meds) / min(meds):.1f}x")
set_threads(8)
print()


# ---------- 4. 时间账要能算平 ----------
# Q1 的验收方式：epoch 时间 ≈ 步数 × 单步时间 + 数据加载

res = bench(one_step, warmup=5, n=20, label="step (for extrapolation)")
steps = len(loader)
predict = steps * res["min_ms"] / 1e3          # 用最小值外推，中位数会被抖动污染

t0 = time.perf_counter()
for X, y in loader:
    optim.zero_grad()
    loss_fn(model(X), y).backward()
    optim.step()
actual = time.perf_counter() - t0

print(f"\n{steps} steps x {res['min_ms']:.3f} ms/step = {predict:.2f} s   (extrapolated)")
print(f"measured full loop                = {actual:.2f} s")
print(f"delta {actual - predict:+.2f} s  --  dataloader + Python loop + fresh-batch alloc")
