"""KV cache 测量：prefill 延迟、decode 每 token 延迟、加速比、cache 大小。

跑法：
    python bench_cache.py
    python bench_cache.py --n-gen 32 --threads 8

不加载训练权重 —— 只测时间，随机初始化即可，省掉 ckpt 不匹配的麻烦。
"""

import argparse
import gc
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch

from perf.bench import bench, device_info, set_threads
from model import GPT


def cache_bytes(caches):
    n = 0
    for layer in caches:
        for k, v in layer:
            n += k.numel() + v.numel()
    return n * 4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-gen", type=int, default=32)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    set_threads(args.threads)
    gc.disable()

    model = GPT(65, args.block_size, args.n_embd, args.n_head, args.n_layer).eval()
    nparam = sum(q.numel() for q in model.parameters())
    print(device_info())
    print(f"model  layers={args.n_layer} embd={args.n_embd} head={args.n_head} "
          f"block={args.block_size}  params={nparam:,}\n")

    # ---------- 1. prefill：一次性处理整个 prompt ----------
    print("== prefill (有 cache, 一次前向) ==")
    prefill = {}
    for L in (16, 64, 128, 256, 512):
        if L > args.block_size:
            continue
        x = torch.randint(0, 65, (1, L))
        with torch.no_grad():
            r = bench(lambda: model(x, use_cache=True), warmup=3, n=10, label=f"  L={L:4d}")
        prefill[L] = r["min_ms"]

    # ---------- 2. decode：逐 token 生成 ----------
    print(f"\n== decode  {args.n_gen} 个 token ==")
    decode = {}
    for L0 in (16, 64, 128, 256):
        if L0 + args.n_gen > args.block_size:
            continue
        idx = torch.randint(0, 65, (1, L0))
        with torch.no_grad():
            r_c = bench(lambda: model.generate(idx, args.n_gen, use_cache=True),
                        warmup=1, n=5, label=f"  L0={L0:4d}  +cache")
            r_n = bench(lambda: model.generate(idx, args.n_gen, use_cache=False),
                        warmup=1, n=5, label=f"  L0={L0:4d}  -cache")
        decode[L0] = (r_c["min_ms"], r_n["min_ms"])

    # ---------- 3. 汇总 ----------
    print("\n== 每 token 延迟 (ms) ==")
    print(f"{'prompt L0':>10}{'+cache':>12}{'-cache':>12}{'加速比':>10}")
    for L0, (c, n) in decode.items():
        print(f"{L0:>10}{c / args.n_gen:>12.3f}{n / args.n_gen:>12.3f}{n / c:>9.1f}x")

    # ---------- 4. cache 大小 ----------
    print("\n== cache 大小 ==")
    print(f"{'T':>6}{'实测':>12}{'公式 2*L*T*C*4':>18}")
    for L in (16, 64, 128, 256, 512):
        if L > args.block_size:
            continue
        x = torch.randint(0, 65, (1, L))
        with torch.no_grad():
            _, _, caches = model(x, use_cache=True)
        got = cache_bytes(caches)
        formula = 2 * args.n_layer * L * args.n_embd * 4
        print(f"{L:>6}{got / 1024:>10.0f}KB{formula / 1024:>16.0f}KB")

    gc.enable()


if __name__ == "__main__":
    main()
