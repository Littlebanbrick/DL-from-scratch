"""加载 week4 训练好的 GPT，在固定的 val batch 上算 NLL。

用法：
    python eval_nll.py
    python eval_nll.py --n-batches 40

作为模块：
    from eval_nll import load_model, make_batches, val_nll
"""

import argparse
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
WEEK4 = ROOT / "week4" / "gpt"
sys.path.insert(0, str(WEEK4))

from data import load_split, load_text          # noqa: E402
from model import GPT                           # noqa: E402

DEFAULT_CKPT = WEEK4 / "ckpt.pt"
DEFAULT_TEXT = WEEK4 / "data" / "input.txt"


def load_model(ckpt_path=DEFAULT_CKPT, device="cpu"):
    """返回 (model, ckpt)。model 已在 eval 模式。"""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = GPT(**ck["config"])
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)
    return model, ck


def make_batches(val_data, block_size, batch_size=32, n_batches=20, seed=1234, device="cpu"):
    """用固定 seed 切一批 val 样本，之后所有实验复用同一批。"""
    g = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n_batches):
        ix = torch.randint(len(val_data) - block_size, (batch_size,), generator=g)
        x = torch.stack([val_data[i:i + block_size] for i in ix])
        y = torch.stack([val_data[i + 1:i + block_size + 1] for i in ix])
        batches.append((x.to(device), y.to(device)))
    return batches


@torch.no_grad()
def val_nll(model, batches):
    """按 token 数量加权的平均交叉熵。"""
    was = model.training
    model.eval()
    total_loss, total_tok = 0.0, 0
    for xb, yb in batches:
        _, loss = model(xb, yb)
        n = xb.numel()
        total_loss += loss.item() * n
        total_tok += n
    model.train(was)
    return total_loss / total_tok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    p.add_argument("--text", default=str(DEFAULT_TEXT))
    p.add_argument("--n-batches", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()

    torch.set_num_threads(args.threads)

    model, ck = load_model(args.ckpt)
    cfg = ck["config"]
    text = load_text(args.text)
    _, val_data, stoi, itos, _, _ = load_split(text)

    batches = make_batches(val_data, cfg["block_size"], args.batch_size, args.n_batches, args.seed)
    nll = val_nll(model, batches)

    print(f"ckpt      {args.ckpt}")
    print(f"config    {cfg}")
    print(f"params    {sum(q.numel() for q in model.parameters()):,}")
    print(f"batches   {args.n_batches} x {args.batch_size} x {cfg['block_size']} tokens = {args.n_batches * args.batch_size * cfg['block_size']:,}")
    print(f"val NLL   {nll:.4f}")


if __name__ == "__main__":
    main()
