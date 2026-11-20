"""基础设施：算 scale、量化、反量化。三种粒度。

约定：
  - 权重张量形状 [out_features, in_features]
  - 对称量化（范围 -lv..lv），lv = 2**(bits-1) - 1
  - 量化结果存 int8

用法：
    from quantlib import quantize_dequantize
    wq, meta = quantize_dequantize(w, bits=4, granularity="group", group_size=64)
"""

import torch


def _levels(bits):
    return 2 ** (bits - 1) - 1


def quantize_per_tensor(w, bits=4):
    """整个张量共用一个 scale。离群值的影响最大。"""
    lv = _levels(bits)
    s = w.abs().max().clamp(min=1e-8) / lv
    q = torch.round(w / s).clamp(-lv, lv).to(torch.int8)
    return q, s


def quantize_per_channel(w, bits=4):
    """每行（每个输出通道）一个 scale。形状 [out, in] -> scale [out, 1]。"""
    assert w.ndim == 2, "per-channel 需要二维张量"
    lv = _levels(bits)
    s = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / lv
    q = torch.round(w / s).clamp(-lv, lv).to(torch.int8)
    return q, s


def quantize_per_group(w, bits=4, group_size=64):
    """每 group_size 个连续元素一个 scale。4bit 量化的标配。

    [out, in] -> 视作 [out, in//g, g]，最后那个 g 维共享一个 scale。
    """
    assert w.ndim == 2, "per-group 需要二维张量"
    out_f, in_f = w.shape
    assert in_f % group_size == 0, f"in_features {in_f} 不能被 {group_size} 整除"

    lv = _levels(bits)
    wg = w.reshape(out_f, in_f // group_size, group_size)
    s = wg.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / lv
    q = torch.round(wg / s).clamp(-lv, lv).to(torch.int8)
    return q.reshape(out_f, in_f), s


def dequantize(q, s):
    """按 scale 还原。q 形状 [out,in]，s 可以是标量 / [out,1] / [out,n_groups,1]。"""
    if s.ndim == 3:
        out_f, in_f = q.shape
        n_groups = s.shape[1]
        g = in_f // n_groups
        return (q.reshape(out_f, n_groups, g).float() * s).reshape(out_f, in_f)
    return q.float() * s


def quantize_dequantize(w, bits=4, granularity="tensor", group_size=64):
    """一步到位，返回 (反量化后的权重, 元信息)。用于测精度损失。"""
    w = w.detach().float()
    if granularity == "tensor":
        q, s = quantize_per_tensor(w, bits)
    elif granularity == "channel":
        q, s = quantize_per_channel(w, bits)
    elif granularity == "group":
        q, s = quantize_per_group(w, bits, group_size)
    else:
        raise ValueError(f"未知粒度 {granularity}")

    meta = {
        "bits": bits,
        "granularity": granularity,
        "n_scales": s.numel(),
        "mse": (w - dequantize(q, s)).pow(2).mean().item(),
    }
    return dequantize(q, s), meta


if __name__ == "__main__":
    # 自检：位数越低误差越大；粒度越细误差越小；离群值专门坑 per-tensor
    torch.manual_seed(0)
    w = torch.randn(64, 128)

    print("正常权重，同一个张量四种配置:")
    for bits in (8, 4):
        for g in ("tensor", "channel", "group"):
            _, m = quantize_dequantize(w, bits, g, group_size=64)
            print(f"  {bits}bit {g:<8} scales {m['n_scales']:>4}  MSE {m['mse']:.3e}")

    print("\n注入一个 10 倍离群值:")
    w2 = w.clone()
    w2[0, 0] = 10.0
    for bits in (8, 4):
        for g in ("tensor", "channel", "group"):
            _, m = quantize_dequantize(w2, bits, g, group_size=64)
            print(f"  {bits}bit {g:<8} scales {m['n_scales']:>4}  MSE {m['mse']:.3e}")
