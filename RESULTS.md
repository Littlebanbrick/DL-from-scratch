# RESULTS

所有实测数字的唯一出处。每条记录必须带硬件与测量条件，缺一不录。

测量一律走 `perf/bench.py`：warmup ≥5 次、`torch.cuda.synchronize()` 后计时、min-of-N（N≥20）、中位数判优劣、最小值报性能。

当前硬件：Intel Core Ultra 9 285H（16 线程，30GB RAM，无 CUDA）· torch 2.14.1+cpu（Week 1–3）。GPU 定了之后记进 `notes/hardware.md`。

| 实验 | 指标 | 数值 | 硬件 | 测量条件 | 日期 |
|---|---|---|---|---|---|
| 工具链验证 | 512×512 fp32 matmul | median 6.020 ms / min 4.961 ms | Ultra 9 285H · torch 2.14.1+cpu | `bench()` warmup=5, N=20；单线程未限制 | 2026-10-16 |
