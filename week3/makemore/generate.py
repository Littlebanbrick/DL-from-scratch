import argparse

import torch
import torch.nn.functional as F

from model import MLP

DOT = "."


def load_mlp(path="mlp.pt"):
    """返回 (model, stoi, itos)。model 已切到 eval() 模式。"""
    ck = torch.load(path, map_location="cpu", weights_only=False)

    cfg = ck["config"]
    model = MLP(cfg["V"], cfg["emb"], cfg["hidden"], cfg["block"])
    model.load_state_dict(ck["state_dict"])     # 维度对不上会在这里直接抛错
    model.eval()

    stoi = ck["stoi"]
    itos = {i: c for c, i in stoi.items()}

    with torch.no_grad():
        probe = model(torch.zeros(1, cfg["block"], dtype=torch.long))
    assert probe.shape == (1, cfg["V"]), f"forward shape {tuple(probe.shape)} != (1, {cfg['V']})"

    return model, stoi, itos


def sample(model, stoi, itos, prefix="", temperature=1.0, max_len=20, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    block = model.block
    idx = [stoi['.']] * block                   # 三个 '.' = 零上下文
    for ch in prefix:                           # 滑动窗口塞入前缀
        idx = (idx + [stoi[ch]])[-block:]

    out = []
    with torch.no_grad():
        for _ in range(max_len):
            logits = model(torch.tensor([idx]))  # [1, V]
            probs = F.softmax(logits / temperature, dim=1)
            nxt = torch.multinomial(probs, 1).item()

            if nxt == stoi['.']:                 # 抽到终止符
                break
            out.append(itos[nxt])
            idx = (idx + [nxt])[-block:]         # 更新前缀序列: 结果反过来影响下一步

    return "".join(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", type=str, default="")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--ckpt", type=str, default="mlp.pt")
    args = p.parse_args()

    model, stoi, itos = load_mlp(args.ckpt)
    print(f"loaded {args.ckpt}   vocab {len(stoi)}   params {sum(p.numel() for p in model.parameters())}")
    print(f"prefix '{args.prefix}'   temperature {args.temp}\n")

    for i in range(args.n):
        seed = None if args.seed is None else args.seed + i
        name = sample(model, stoi, itos, args.prefix, args.temp, seed=seed)
        print(f"  {i + 1:3d}. {name}")
