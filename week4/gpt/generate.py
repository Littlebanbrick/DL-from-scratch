'''
generate.py
推理
'''

import argparse

import torch

from model import GPT


def load_model(path="ckpt.pt"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(**ck["config"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck["stoi"], ck["itos"], ck["config"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="ckpt.pt")
    p.add_argument("--tokens", type=int, default=500)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--prompt", type=str, default="\n")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    model, stoi, itos, cfg = load_model(args.ckpt)
    print(f"loaded {args.ckpt}  {cfg}")

    enc = lambda s: [stoi[c] for c in s]
    dec = lambda ids: "".join(itos[int(i)] for i in ids)

    idx = torch.tensor(enc(args.prompt), dtype=torch.long).unsqueeze(0)
    out = model.generate(idx, max_new_tokens=args.tokens, temperature=args.temp)
    print("-" * 60)
    print(dec(out[0].tolist()))


if __name__ == "__main__":
    main()
