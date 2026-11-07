import torch
import torch.nn as nn
import torch.nn.functional as F

from model import MLP

torch.manual_seed(0)

BASELINE = 2.4540       # list.py 算出来的计数模型 NLL

stoi = {c: i for i, c in enumerate(".abcdefghijklmnopqrstuvwxyz")}
itos = {i: c for c, i in stoi.items()}
V = len(stoi)           # 27

with open("data/names.txt", "r", encoding="utf-8") as f:
    names = [line.strip() for line in f]

xs, ys = [], []
for name in names:
    seq = [".", ".", "."] + list(name) + ["."]
    for a, b, c, d in zip(seq, seq[1:], seq[2:], seq[3:]):
        xs.append([stoi[a], stoi[b], stoi[c]])
        ys.append(stoi[d])

X = torch.tensor(xs)                            # [N, 3]
Y = torch.tensor(ys)                            # [N]

print(f"names   {len(names)}")
print(f"pairs   {len(X)}")
print(f"X       {tuple(X.shape)}   Y {tuple(Y.shape)}")

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def nll_loss():
    return F.cross_entropy(model(X), Y)

def loss_fn():
    return nll_loss()

def report(step):
    with torch.no_grad():
        print(f"step {step:5d}  nll {nll_loss().item():.4f}   total {loss_fn().item():.4f}")



if __name__ == "__main__":
    report(0)

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


    CKPT = "mlp.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "V": model.C.weight.shape[0],
                "emb": model.C.weight.shape[1],
                "hidden": model.fc1.out_features,
                "block": 3,
            },
            "stoi": stoi,
            "nll": nll_loss().item(),
        },
        CKPT,
    )
    print(f"saved {CKPT}")
