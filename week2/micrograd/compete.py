import gc
import pathlib
import sys

sys.path.insert(0, ".")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import argparse

import torch
import torch.nn as nn

from perf.bench import bench, device_info, set_threads

set_threads(8)

from nn import MLP
from value import Value
from train import make_moons

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001)
args = parser.parse_args()
lr = args.lr


X, y = make_moons(n=100, seed=0)
X_pt = torch.from_numpy(X).double()
y_pt = torch.from_numpy(y).double()


# My custom model
mg_model = MLP(2, [8, 8, 1])

def mg_loss():
    losses = [(1 - yi * mg_model(xi)[0]).relu() for xi, yi in zip(X, y)]
    return sum(losses, Value(0.0)) * (1.0 / len(X))

def mg_forward():
    return mg_loss()

def mg_fwd_bwd():
    loss = mg_loss()
    mg_model.zero_grad()
    loss.backward()

def mg_step():
    loss = mg_loss()
    mg_model.zero_grad()
    loss.backward()
    for p in mg_model.parameters():
        p.data -= lr * p.grad


# Pytorch model
torch_model = nn.Sequential(
    nn.Linear(2, 8), nn.ReLU(),
    nn.Linear(8, 8), nn.ReLU(),
    nn.Linear(8, 1),
).double()
optimizer = torch.optim.SGD(torch_model.parameters(), lr=lr)

def pt_loss():
    pred = torch_model(X_pt).squeeze(-1)
    return (1 - y_pt * pred).clamp_min(0).mean()

def pt_forward():
    # no "no_grad()" here
    return pt_loss()

def pt_fwd_bwd():
    optimizer.zero_grad()
    pt_loss().backward()

def pt_step():
    optimizer.zero_grad()
    loss = pt_loss()
    loss.backward()
    optimizer.step()

print(device_info())
print(f"params  mine {len(mg_model.parameters())}  torch {sum(p.numel() for p in torch_model.parameters())}")
print()

results = {}
gc.disable()
try:
    for name, mg_fn, pt_fn in [
        ("forward",  mg_forward, pt_forward),
        ("fwd+bwd",  mg_fwd_bwd, pt_fwd_bwd),
        ("full step", mg_step,   pt_step),
    ]:
        mine = bench(mg_fn, warmup=3, n=20, label=f"mine  {name}")
        torc = bench(pt_fn, warmup=5, n=20, label=f"torch {name}")
        results[name] = (mine["min_ms"], torc["min_ms"])
        print()
finally:
    gc.enable()


print("=" * 58)
print(f"{'stage':<12}{'mine (ms)':>12}{'torch (ms)':>13}{'ratio':>10}")
print("-" * 58)
for name, (m, t) in results.items():
    print(f"{name:<12}{m:>12.3f}{t:>13.4f}{m / t:>9.0f}x")
print("=" * 58)
