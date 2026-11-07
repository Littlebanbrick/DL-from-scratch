import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

BASELINE = 2.4540       # list.py 算出来的计数模型 NLL，这是要复现的目标

stoi = {c: i for i, c in enumerate(".abcdefghijklmnopqrstuvwxyz")}
itos = {i: c for c, i in stoi.items()}
V = len(stoi)           # 27

with open("data/names.txt", "r", encoding="utf-8") as f:
    names = [line.strip() for line in f]


xs, ys = [], []
for name in names:
    seq = ["."] + list(name) + ["."]
    for a, b in zip(seq, seq[1:]):
        xs.append(stoi[a])
        ys.append(stoi[b])

X = torch.tensor(xs)                    # [N]
Y = torch.tensor(ys)                    # [N]
X_oh = F.one_hot(X, num_classes=V).float()   # [N, 27]

print(f"names   {len(names)}")
print(f"pairs   {len(X)}")
print(f"X       {tuple(X_oh.shape)}   Y {tuple(Y.shape)}")


class Bigram(nn.Module):
    def __init__(self):
        super().__init__()
        # 只有一个线性层
        # 输入 one-hot (V=27维), 输出27维
        self.W = nn.Linear(V, V, bias=False)

    def forward(self, x):
        return self.W(x)                # logits [N, 27]


model = Bigram()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


def nll_loss():
    return F.cross_entropy(model(X_oh), Y)


def loss_fn():
    # 先不 + 0.01 * (model.W.weight ** 2).mean() 看看会怎么样
    return nll_loss()


def report(step):
    with torch.no_grad():
        print(f"step {step:5d}  nll {nll_loss().item():.4f}   total {loss_fn().item():.4f}")


report(0)       # 预测：初始化附近应该接近 log(27) = 3.296

for step in range(1, 6001):
    loss = loss_fn()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        report(step)


print()
print(f"baseline    (counts)   {BASELINE:.4f}")
print(f"final nll   (nn)       {nll_loss().item():.4f}")
print(f"final total (including L2)   {loss_fn().item():.4f}")
