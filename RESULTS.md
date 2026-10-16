# RESULTS

所有实测数字的唯一出处。每条记录必须带硬件与测量条件，缺一不录。

测量一律走 `perf/bench.py`：warmup ≥5 次、`torch.cuda.synchronize()` 后计时、min-of-N（N≥20）、中位数判优劣、最小值报性能。

当前硬件：本地 CPU（Week 1–3）。GPU 定了之后记进 `notes/hardware.md`。

| 实验 | 指标 | 数值 | 硬件 | 测量条件 | 日期 |
|---|---|---|---|---|---|
| | | | | | |
