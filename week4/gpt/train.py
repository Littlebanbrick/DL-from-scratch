'''
train.py
训练, 保存模型
'''

import argparse
import gc
import time

import torch

from data import get_batch, load_split, load_text
from model import GPT


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, iters, device):
    model.eval()
    out = {}
    for split, d in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(iters)
        for i in range(iters):
            xb, yb = get_batch(d, block_size, batch_size, device)
            _, loss = model(xb, yb)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=67)
    p.add_argument("--ckpt", type=str, default="ckpt.pt")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = load_text()
    train_data, val_data, stoi, itos, encode, decode = load_split(text)
    vocab_size = len(stoi)
    print(f"device {device}  vocab {vocab_size}  train {len(train_data)}  val {len(val_data)}")

    model = GPT(vocab_size, args.block_size, args.n_embd, args.n_head, args.n_layer).to(device)
    print(f"params {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    gc.disable()
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        xb, yb = get_batch(train_data, args.block_size, args.batch_size, device)
        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.eval_every == 0:
            m = estimate_loss(model, train_data, val_data, args.block_size,
                              args.batch_size, args.eval_iters, device)
            print(f"step {step:6d}  train {m['train']:.4f}  val {m['val']:.4f}"
                  f"  ({time.perf_counter() - t0:.0f}s)")
    gc.enable()

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "vocab_size": vocab_size,
                "block_size": args.block_size,
                "n_embd": args.n_embd,
                "n_head": args.n_head,
                "n_layer": args.n_layer,
            },
            "stoi": stoi,
            "itos": itos,
        },
        args.ckpt,
    )
    print(f"saved {args.ckpt}")


if __name__ == "__main__":
    main()
