import matplotlib
matplotlib.use("Agg")  # 无窗口环境必需，且要在 import pyplot 之前
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import v2

# 固定种子
import random, numpy as np
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

import time

import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from perf.bench import bench, device_info, peak_mem_mb, reset_peak_mem, set_threads
# 固定线程数
set_threads(8)


ds_train = datasets.FashionMNIST(
    root="notes/data",
    train=True,
    download=True,
    transform=v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]),
    target_transform=v2.Lambda(
        lambda y: F.one_hot(torch.tensor(y), num_classes=10).float()
    ),
)

ds_test = datasets.FashionMNIST(
    root="notes/data",
    train=False,
    download=True,
    transform=v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]),
    target_transform=v2.Lambda(
        lambda y: F.one_hot(torch.tensor(y), num_classes=10).float()
    ),
)

'''
python train.py 命令接收四个参数：--batch-size、--epochs、--lr、--hidden
'''
import argparse
parser = argparse.ArgumentParser(description="PyTorch FashionMNIST Example")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--hidden", type=int, default=512)
'''
action="store_true"
表示这个参数出现时返回 True, 否则 False。不吃值。
'''
parser.add_argument("--profile", action="store_true")

''' 精简模式：只输出单 epoch 时间、每步时间、最终 val acc'''
parser.add_argument("--elite", action="store_true")
''' 只测单步时间，出表后退出（Q2 用）'''
parser.add_argument("--bench", action="store_true")

batch_size   = parser.parse_args().batch_size
epochs       = parser.parse_args().epochs
lr           = parser.parse_args().lr
hidden       = parser.parse_args().hidden
profile_mode = parser.parse_args().profile
elite_mode   = parser.parse_args().elite
bench_mode   = parser.parse_args().bench

dl_train = torch.utils.data.DataLoader(
    ds_train,
    batch_size=batch_size,
    shuffle=True
)

dl_test = torch.utils.data.DataLoader(
    ds_test,
    batch_size=batch_size,
    shuffle=False
)

'''
不使用 nn.Sequential
'''
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(28*28, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.linear3 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        logits = self.linear3(x)
        return logits

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

model = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    total, n = 0.0, 0
    for X, y in dataloader:          # 不要 enumerate，否则 X 拿到的是下标
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()
        n += 1

    return total / n                 # 该 epoch 的平均训练 loss

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0.0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # pred 形状是 [batch, 10], argmax(1) 取 dim = 1 的每行最大值的下标 - 把 one-hot 编码还原成类别索引
            # pred.argmax(1) == y.argmax(1) 逐元素比较，返回一个 bool 张量，True 表示预测正确
            # 转成浮点数求和
            # item() 把 0-dim 张量转成 Python 数字
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct       # 返回给主循环画图用

'''
python train.py --profile
出表后直接退出，不跑训练。
'''
if profile_mode:
    from torch.profiler import profile, ProfilerActivity

    X, y = next(iter(dl_train))
    model.train()

    # 先跑 warmup，不录进 profile
    for _ in range(2):
        optimizer.zero_grad()
        loss_fn(model(X), y).backward()
        optimizer.step()

    # profile 是 torch.profiler 中的类，这里创建一个对象
    # 参数：
    # activities=[ProfilerActivity.CPU] 只记录 CPU 上的运算
    # record_shapes=True 记录每个算子的输入输出形状
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        for _ in range(8):
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()

    print(f"\n=== Q1: profiler  (bs={batch_size}, hidden={hidden}) ===")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))
    prof.export_chrome_trace("trace.json")
    print("saved trace.json — open at https://ui.perfetto.dev/")
    sys.exit(0)

'''
Q2: 单步时间 —— 只测步，不测 epoch
python train.py --bench --batch-size 128
'''
if bench_mode:
    X, y = next(iter(dl_train))   # 固定一批，排除数据加载
    model.train()

    def one_step():
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()

    r = bench(one_step, warmup=5, n=20, label=f"step bs={batch_size} hidden={hidden}")

    # 再用真实数据循环量一次，用来核对 bench 差在哪
    t0 = time.perf_counter()
    for bx, by in dl_train:
        optimizer.zero_grad()
        loss_fn(model(bx), by).backward()
        optimizer.step()
    loop = time.perf_counter() - t0
    steps = len(dl_train)

    print(f"\n=== Q2: step time  (bs={batch_size}, hidden={hidden}) ===")
    print(f"steps/epoch        {steps}")
    print(f"bench step         min {r['min_ms']:.3f} ms   median {r['median_ms']:.3f} ms")
    print(f"real loop step     {loop / steps * 1e3:.3f} ms   (includes dataloader)")
    print(f"extrapolated epoch {steps * r['min_ms'] / 1e3:.2f} s   (steps x min step)")
    print(f"measured epoch     {loop:.2f} s")
    sys.exit(0)

train_losses = []
test_losses = []
accuracies = []

for t in range(epochs):
    print(f"=============================")
    print(f"Epoch {t+1} / {epochs}")
    print(f"=============================")

    t0 = time.perf_counter()
    train_loss = train(dl_train, model, loss_fn, optimizer)
    t1 = time.perf_counter()

    if elite_mode:
        print(f"epoch time: {t1-t0:.2f}s")
    else:
        print(f"train loss: {train_loss:.4f}  time: {t1-t0:.2f}s")
    train_losses.append(train_loss)

    t0 = time.perf_counter()
    test_loss, acc = test(dl_test, model, loss_fn)
    t1 = time.perf_counter()

    if elite_mode:
        print(f"val acc: {acc:.4f}")
    else:
        print(f"test loss: {test_loss:.4f}  acc: {acc:.4f}  time: {t1-t0:.2f}s")
    test_losses.append(test_loss)
    accuracies.append(acc)

# '''
# train / val loss 曲线储存在 curve.png
# '''
# plt.figure(figsize=(6, 4))
# plt.plot(range(1, epochs + 1), train_losses, marker="o", label="train")
# plt.plot(range(1, epochs + 1), test_losses, marker="s", label="val (test set)")
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.title(f"FashionMLP  hidden={hidden}  bs={batch_size}  lr={lr}")
# plt.legend()
# plt.grid(alpha=0.3)
# plt.savefig("curve.png", dpi=150, bbox_inches="tight")
# print("saved curve.png")
