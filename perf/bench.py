"""统一计时工具。所有实验引用这一份，数字才可比。

用法：
    from perf.bench import bench, device_info, peak_mem_mb, reset_peak_mem

    reset_peak_mem()
    bench(lambda: model(x), label="mlp forward")
    print(device_info(), peak_mem_mb())

规则（沿用 HPC101）：
    - warmup 后再计时
    - 计时前后都要 synchronize，否则测的是 kernel 入队时间
    - min-of-N（N >= 20）
    - 中位数判优劣，最小值报性能
"""

import statistics
import time

import torch


def sync() -> None:
    """CUDA 同步。CPU 阶段是空操作。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench(fn, *, warmup: int = 5, n: int = 20, label: str = "") -> dict:
    """跑 fn 并计时，返回 {"label", "median_ms", "min_ms", "n"}。

    中位数判优劣，最小值报性能。
    """
    for _ in range(warmup):
        fn()
    sync()

    ts = []
    for _ in range(n):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)

    med, mn = statistics.median(ts), min(ts)
    print(f"{label or 'run':<28} median {med * 1e3:9.3f} ms   min {mn * 1e3:9.3f} ms")
    return {"label": label, "median_ms": med * 1e3, "min_ms": mn * 1e3, "n": n}


def peak_mem_mb() -> float | None:
    """CUDA 峰值显存（MB）。CPU 返回 None。"""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 2**20


def reset_peak_mem() -> None:
    """每个实验开始前调一次，否则峰值会累积。"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def device_info() -> str:
    """一行硬件描述，直接抄进 RESULTS.md。"""
    thr = torch.get_num_threads()
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return f"{p.name} {p.total_memory / 2**30:.1f}GB | torch {torch.__version__} | cuda {torch.version.cuda} | cpu_threads {thr}"
    return f"CPU | torch {torch.__version__} | cpu_threads {thr}"


def set_threads(n: int = 8) -> int:
    """限制 PyTorch 的 CPU 线程数，返回设置后的值。

    为什么需要：默认值是 CPU 核数。在大小核混合的机器上把线程拉满会让调度
    抖动，同一配置的中位数能在 1 ms 和 90 ms 之间跳。实测 Ultra 9 285H：
    threads=8 稳定，threads=16 完全不可信。

    跑任何性能实验之前先调一次，并把结果记进 RESULTS.md 的测量条件。
    """
    torch.set_num_threads(n)
    return torch.get_num_threads()
