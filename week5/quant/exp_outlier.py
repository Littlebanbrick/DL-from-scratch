"""实验 3：人为注入离群值，看三种粒度各被伤到什么程度。

先记下有离群值但不量化的 NLL 作为参照。

用法：
    python exp_outlier.py
    python exp_outlier.py --magnitude 20 --target ffn
"""

import argparse

import torch

from quantlib import quantize_dequantize
from eval_nll import DEFAULT_TEXT, load_model, load_split, make_batches, val_nll
from exp_ffn import ffn_tensors
from exp_attn import attn_tensors


def pick(target):
    return {"ffn": ffn_tensors, "attn": attn_tensors}[target]


def inject(model, target, magnitude, n_tensors=1):
    """前 n_tensors 个目标张量的 [0,0] 位置设成 magnitude × 该张量的 |max|。"""
    vals = []
    with torch.no_grad():
        for _, w in pick(target)(model)[:n_tensors]:
            v = magnitude * w.abs().max().item()
            w[0, 0] = v
            vals.append(v)
    return vals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-batches", type=int, default=20)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--magnitude", type=float, default=10.0)
    p.add_argument("--target", choices=["ffn", "attn"], default="ffn")
    p.add_argument("--n-tensors", type=int, default=1)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()

    torch.set_num_threads(args.threads)

    model, ck = load_model()
    cfg = ck["config"]
    _, val_data, *_ = load_split(DEFAULT_TEXT.read_text(encoding="utf-8"))
    batches = make_batches(val_data, cfg["block_size"], n_batches=args.n_batches)

    n_tensor = len(pick(args.target)(model))
    print(f"目标 {args.target}：共 {n_tensor} 个张量，前 {args.n_tensors} 个各注入 1 个 {args.magnitude:g}× 离群值\n")

    # 参照 1：原始 fp32
    model.load_state_dict(ck["state_dict"])
    ref_clean = val_nll(model, batches)

    # 参照 2：注入离群值但不量化 —— 这才是量化误差的正确对照
    model.load_state_dict(ck["state_dict"])
    inject(model, args.target, args.magnitude)
    ref_dirty = val_nll(model, batches)

    print(f"参照  fp32 无离群值          val NLL {ref_clean:.4f}")
    print(f"参照  fp32 有离群值          val NLL {ref_dirty:.4f}   （离群值本身的扰动 {ref_dirty - ref_clean:+.4f}）\n")

    rows = []
    for dirty in (False, True):
        for bits in (8, 4):
            for gran in ("tensor", "channel", "group"):
                model.load_state_dict(ck["state_dict"])
                if dirty:
                    inject(model, args.target, args.magnitude, args.n_tensors)

                with torch.no_grad():
                    for _, w in pick(args.target)(model):
                        wq, _ = quantize_dequantize(w.data, bits, gran, args.group_size)
                        w.copy_(wq)

                nll = val_nll(model, batches)
                rows.append((dirty, bits, gran, nll))

    model.load_state_dict(ck["state_dict"])

    print(f"{'离群':>5}{'bits':>6}{'granularity':>13}{'val NLL':>10}{'量化损失':>11}")
    print("-" * 46)
    for dirty, bits, gran, nll in rows:
        ref = ref_dirty if dirty else ref_clean
        tag = "有" if dirty else "无"
        print(f"{tag:>5}{bits:>6}{gran:>13}{nll:>10.4f}{nll - ref:>+11.4f}")


if __name__ == "__main__":
    main()
