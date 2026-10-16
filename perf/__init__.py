"""统一的计时与硬件描述工具。

所有实验都引用这一份，数字才可比。各写各的计时 = 数字不可比。
CPU 阶段同样可用：sync() 什么都不做。
"""

from perf.bench import bench, device_info, peak_mem_mb, reset_peak_mem, sync

__all__ = ["bench", "device_info", "peak_mem_mb", "reset_peak_mem", "sync"]
