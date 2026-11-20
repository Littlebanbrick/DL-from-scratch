"""实验 2：只量化 attention 权重，和 exp_ffn 的 FFN 结果对照。

attention 权重 = blocks.{i}.sa.heads.{j}.{key,query,value}.weight + blocks.{i}.sa.proj.weight
共 4 层 × (4 头 × 3 + 1) = 52 个张量。

用法：
    python exp_attn.py
"""

import argparse

import torch

from quantlib import quantize_dequantize
from eval_nll import DEFAULT_TEXT, load_model, load_split, make_batches, val_nll


def attn_tensors(model):
    """返回 [(名字, 张量)]，只要 attention 的 Linear weight（不含 bias、不含 tril）。"""
    out = []
    for i, blk in enumerate(model.blocks):
        for j, h in enumerate(blk.sa.heads):
            out.append((f"blocks.{i}.sa.heads.{j}.key.weight", h.key.weight))
            out.append((f"blocks.{i}.sa.heads.{j}.query.weight", h.query.weight))
            out.append((f"blocks.{i}.sa.heads.{j}.value.weight", h.value.weight))
        out.append((f"blocks.{i}.sa.proj.weight", blk.sa.proj.weight))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-batches", type=int, default=20)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()

    torch.set_num_threads(args.threads)

    model, ck = load_model()
    cfg = ck["config"]
    _, val_data, *_ = load_split(DEFAULT_TEXT.read_text(encoding="utf-8"))
    batches = make_batches(val_data, cfg["block_size"], n_batches=args.n_batches)

    wlist = attn_tensors(model)
    n_param = sum(w.numel() for _, w in wlist)
    total = sum(q.numel() for q in model.parameters())
    print(f"attention 张量 {len(wlist)} 个，参数 {n_param:,}（占全模型 {n_param / total:.1%}）")

    base = val_nll(model, batches)
    print(f"\nbaseline (fp32)                val NLL {base:.4f}\n")

    rows = []
    for bits in (8, 4):
        for gran in ("tensor", "channel", "group"):
            model.load_state_dict(ck["state_dict"])

            total_mse, n_scales = 0.0, 0
            with torch.no_grad():
                for _, w in attn_tensors(model):
                    wq, meta = quantize_dequantize(w.data, bits, gran, args.group_size)
                    w.copy_(wq)
                    total_mse += meta["mse"] * w.numel()
                    n_scales += meta["n_scales"]

            nll = val_nll(model, batches)
            size = n_param * bits / 8 + n_scales * 2
            rows.append((bits, gran, total_mse / n_param, n_scales, size, nll))

    model.load_state_dict(ck["state_dict"])

    print(f"{'bits':>5}{'granularity':>13}{'scales':>9}{'weight MSE':>13}{'size':>11}{'val NLL':>10}{'ΔNLL':>9}")
    print("-" * 70)
    print(f"{'-':>5}{'fp32':>13}{'-':>9}{'-':>13}{n_param * 4 / 1024:>10.0f}K{base:>10.4f}{0.0:>+9.4f}")
    for bits, gran, mse, ns, size, nll in rows:
        print(f"{bits:>5}{gran:>13}{ns:>9,}{mse:>13.3e}{size / 1024:>10.0f}K{nll:>10.4f}{nll - base:>+9.4f}")


if __name__ == "__main__":
    main()
