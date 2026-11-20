"""
看权重分布

用法：
    python weight_stats.py
    python weight_stats.py --ckpt ../../week4/gpt/ckpt.pt
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# 峰度
def kurtosis(x):
    x = x.float()
    return ((x - x.mean()) ** 4).mean() / x.var() ** 2


def levels_used(w, bits):
    """per-tensor 量化后，实际用掉了多少个量化级。"""
    lv = 2 ** (bits - 1) - 1
    s = w.abs().max() / lv
    q = torch.round(w / s).clamp(-lv, lv)
    return len(torch.unique(q))


def group_of(name):
    if name.endswith("tril"):
        return "buffer"
    if "token_embedding" in name or "position_embedding" in name:
        return "embedding"
    if ".sa." in name:
        return "attention"
    if "ffwd" in name:
        return "ffn"
    if "lm_head" in name:
        return "lm_head"
    return "other"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(pathlib.Path(__file__).resolve().parents[2] / "week4/gpt/ckpt.pt"))
    p.add_argument("--out", default="weights.png")
    p.add_argument("--bins", type=int, default=120)
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    print(f"loaded {args.ckpt}")
    print(f"config {ck.get('config', {})}\n")

    rows = []
    for name, v in sd.items():
        if not v.is_floating_point() or name.endswith("tril"):
            continue
        w = v.flatten().float()
        mx = w.abs().max().item()
        std = w.std().item()
        rows.append({
            "name": name,
            "group": group_of(name),
            "numel": w.numel(),
            "min": w.min().item(),
            "max": w.max().item(),
            "std": std,
            "ratio": mx / std,
            "kurt": kurtosis(w).item(),
            "lv8": levels_used(w, 8),
            "lv4": levels_used(w, 4),
            "w": w,
        })

    # ---------- 逐张量指标 ----------
    print(f"{'张量':<34}{'元素':>8}{'std':>9}{'|max|/std':>11}{'峰度':>7}{'int8 级':>9}{'int4 级':>9}")
    print("-" * 88)
    for r in rows:
        print(f"{r['name']:<34}{r['numel']:>8,}{r['std']:>9.4f}{r['ratio']:>11.2f}"
              f"{r['kurt']:>7.2f}{r['lv8']:>7}/256{r['lv4']:>7}/16")

    # ---------- 按类聚合 ----------
    print(f"\n{'类别':<12}{'张量数':>8}{'元素':>10}{'std':>9}{'|max|/std':>11}{'峰度':>7}{'int8 级':>9}{'int4 级':>9}")
    print("-" * 78)
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    for g, rs in groups.items():
        allw = torch.cat([r["w"] for r in rs])
        print(f"{g:<12}{len(rs):>8}{allw.numel():>10,}{allw.std():>9.4f}"
              f"{allw.abs().max().item() / allw.std().item():>11.2f}{kurtosis(allw):>7.2f}"
              f"{levels_used(allw, 8):>7}/256{levels_used(allw, 4):>7}/16")

    # ---------- 判定 ----------
    print("\n判定（|max|/std < 5 且峰度 < 5 视为干净）:")
    dirty = [r for r in rows if r["ratio"] >= 5 or r["kurt"] >= 5]
    if dirty:
        for r in dirty:
            print(f"  ! {r['name']}  ratio {r['ratio']:.2f}  kurt {r['kurt']:.2f}")
    else:
        print("  全部张量接近正态，没有明显离群值")

    # ---------- 直方图 ----------
    picks = []
    for g in ("embedding", "attention", "ffn", "lm_head"):
        rs = groups.get(g, [])
        if rs:
            picks.append((g, torch.cat([r["w"] for r in rs])))

    fig, axes = plt.subplots(1, len(picks), figsize=(4 * len(picks), 3))
    if len(picks) == 1:
        axes = [axes]
    for ax, (g, w) in zip(axes, picks):
        ax.hist(w.numpy(), bins=args.bins, color="steelblue")
        ax.set_title(f"{g}\nstd={w.std():.4f}  |max|/std={w.abs().max() / w.std():.2f}")
        ax.set_xlabel("weight")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
