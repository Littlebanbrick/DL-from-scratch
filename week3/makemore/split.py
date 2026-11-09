import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import model as M

SEED = 0
RATIO = (0.7, 0.2, 0.1)

stoi = {c: i for i, c in enumerate(".abcdefghijklmnopqrstuvwxyz")}
itos = {i: c for c, i in stoi.items()}
V = len(stoi)


def load_names(path="data/names.txt"):
    with open(path, "r", encoding="utf-8") as f:
        rows = [line.strip() for line in f if line.strip()]
    return sorted(set(rows))


def get_splits(path="data/names.txt", seed=SEED, ratio=RATIO):
    names = load_names(path)
    random.Random(seed).shuffle(names)

    n = len(names)
    n_train = int(ratio[0] * n)
    n_dev = int(ratio[1] * n)

    train = names[:n_train]
    dev = names[n_train:n_train + n_dev]
    test = names[n_train + n_dev:]
    return train, dev, test


def no_overlap(train, dev, test):
    # 确认没有重叠
    all_names = train + dev + test
    assert not (set(train) & set(dev)), "train/dev overlap"
    assert not (set(train) & set(test)), "train/test overlap"
    assert not (set(dev) & set(test)), "dev/test overlap"
    assert len(all_names) == len(set(all_names)), "duplicates in source"
    print(f"\ntotal names {len(all_names)}   no overlap, seed={SEED}")


def baseline(train, dev, smooth=0.0):
    counts = np.zeros((V, V), dtype=float)
    for name in train:
        first = name[0]
        last = name[-1]
        counts[stoi['.']][stoi[first]] += 1
        counts[stoi[last]][stoi['.']] += 1

        for i in range(len(name)-1):
            counts[stoi[name[i]]][stoi[name[i+1]]] += 1

    # 防止后面取对数时出现除零错误, 给 counts 中每个位置加上一点偏移量
    if smooth:
        counts += smooth

    P = counts / counts.sum(1, keepdims=True).clip(min=1)
    nll_sum, n_pairs = 0.0, 0

    # eval in dev set, not train set
    for name in dev:
        seq = ['.'] + list(name) + ['.']
        for a, b in zip(seq, seq[1:]):
            nll_sum += -np.log(P[stoi[a], stoi[b]])
            n_pairs += 1

    return nll_sum / n_pairs


def nll_loss(model, X, Y):
    return F.cross_entropy(model(X), Y)

def loss_fn(model, X, Y):
    return nll_loss(model, X, Y)

def report(step, model, X, Y):
    with torch.no_grad():
        print(f"step {step:5d}  nll {nll_loss(model, X, Y).item():.4f}   total {loss_fn(model, X, Y).item():.4f}")


@torch.no_grad()
def evaluate(model, X, Y):
    was_training = model.training
    model.eval()
    logits = model(X)
    nll = F.cross_entropy(logits, Y).item()
    model.train(was_training)          # 恢复原状态
    return nll


def train(model, train, dev, LR=0.1, MAXSTEPS=100000, PATIENCE=3, EVALEVERY=100):
    Xtr, Ytr = train
    Xdev, Ydev = dev
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_nll, best_state, stale = float('inf'), None, 0

    for step in range(1, MAXSTEPS+1):
        optimizer.zero_grad()
        loss = loss_fn(model, Xtr, Ytr)
        loss.backward()
        optimizer.step()

        if step % EVALEVERY == 0:
            dev_nll = evaluate(model, Xdev, Ydev)

            if dev_nll < best_nll - 1e-4:
                best_nll = dev_nll
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1

            print(f"step {step:6d}  train {loss.item():.4f}  dev {dev_nll:.4f}  best {best_nll:.4f}  stale {stale}/{PATIENCE}")

            if stale >= PATIENCE:
                break

    return best_state, best_nll


if __name__ == "__main__":
    torch.manual_seed(SEED)
    train_set, dev, test = get_splits()
    no_overlap(train_set, dev, test)

    model = M.MLP()

    BASELINE = baseline(train_set, dev, smooth=1.0)
    MAXSTEPS = 100000
    EVALEVERY = 100
    PATIENCE = 20

    def to_tensors(names, block=3):
        xs, ys = [], []
        for name in names:
            seq = ["."] * (block - 1) + list(name) + ["."]
            for ctx in zip(*(seq[i:] for i in range(block + 1))):
                xs.append([stoi[c] for c in ctx[:block]])
                ys.append(stoi[ctx[block]])
        return torch.tensor(xs), torch.tensor(ys)

    Xtr, Ytr = to_tensors(train_set)
    Xdev, Ydev = to_tensors(dev)
    Xte, Yte = to_tensors(test)

    total = len(train_set) + len(dev) + len(test)
    for tag, names, X in (("train", train_set, Xtr), ("dev", dev, Xdev), ("test", test, Xte)):
        print(f"{tag:5s}  names {len(names):6d}  pairs {len(X):7d}  ({len(names) / total:.1%})")

    print(f"\nbaseline (train counts -> dev nll)  {BASELINE:.4f}")

    # MAXSTEPS should be greater than EVALEVERY
    assert MAXSTEPS > EVALEVERY, f"MAXSTEPS {MAXSTEPS} should be greater than EVALEVERY {EVALEVERY}"
    best_state, best_dev = train(
        model, (Xtr, Ytr), (Xdev, Ydev),
        MAXSTEPS=MAXSTEPS, PATIENCE=PATIENCE, EVALEVERY=EVALEVERY,
    )

    # 回到最佳状态
    model.load_state_dict(best_state)

    # 测试
    test_nll = evaluate(model, Xte, Yte)

    print(f"\n{'':28s}{'dev':>8s}{'test':>8s}")
    print(f"{'bigram (train counts)':28s}{BASELINE:8.4f}{'':>8s}")
    print(f"{'mlp (3-char context)':28s}{best_dev:8.4f}{test_nll:8.4f}")
    print(f"\ngap dev-test {test_nll - best_dev:+.4f}")

    torch.save(
        {
            "state_dict": best_state,
            "config": {
                "V": model.C.weight.shape[0],
                "emb": model.C.weight.shape[1],
                "hidden": model.fc1.out_features,
                "block": 3,
            },
            "stoi": stoi,
            "nll": test_nll,
        },
        "split.pt",
    )
    print("saved split.pt")

