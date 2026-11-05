# RESULTS

所有实测数字的唯一出处。每条记录必须带硬件与测量条件，缺一不录。

测量一律走 `perf/bench.py`：warmup ≥5 次、min-of-N（N≥20）、中位数判优劣、最小值报性能。

当前硬件：Intel Core Ultra 9 285H（16 线程，30GB RAM，无 CUDA）· torch 2.14.1+cpu · seed=0（Week 1–3）。GPU 定了之后记进 `notes/hardware.md`。

## Week 1 · FashionMLP

默认配置：3 层 MLP（784→h→h→10），FashionMNIST，SGD，lr=1e-3，transform = `ToImage` + `ToDtype(float32)`。

| 实验 | 指标 | 数值 | 硬件 | 测量条件 | 日期 |
|---|---|---|---|---|---|
| 工具链验证 | 512×512 fp32 matmul | median 6.020 ms / min 4.961 ms | Ultra 9 285H · torch 2.14.1+cpu | `bench()` warmup=5, N=20；线程数未限制 | 2026-10-16 |
| Q1 profiler | `mm`+`addmm` 自身占比, h=32 | 24.5% | Ultra 9 285H · cpu_threads 8 | 2 步热身 + 8 步录制，record_shapes=True，bs=64 | 2026-11-05 |
| Q1 profiler | `mm`+`addmm` 自身占比, h=128 | 42.9% | 同上 | 同上 | 2026-11-05 |
| Q1 profiler | `mm`+`addmm` 自身占比, h=512 | 59.5% | 同上 | 同上 | 2026-11-05 |
| Q2 batch sweep | 单步 min / 实测 epoch, bs=32 | 0.342 ms / 4.21 s | 同上 | h=512，bench N=20 warmup=5，epoch 单次计时 | 2026-11-05 |
| Q2 batch sweep | 单步 min / 实测 epoch, bs=128 | 0.881 ms / 3.67 s | 同上 | 同上 | 2026-11-05 |
| Q2 batch sweep | 单步 min / 实测 epoch, bs=512 | 2.765 ms / 3.36 s | 同上 | 同上 | 2026-11-05 |
| Q2 batch sweep | 单步 min / 实测 epoch, bs=2048 | 10.648 ms / 3.41 s | 同上 | 同上 | 2026-11-05 |
| Q3 数据管线 | 6000 样本，无 transform | 24.3 ms | 同上 | 逐样本索引 `base[i]`，单次计时 | 2026-11-05 |
| Q3 数据管线 | 6000 样本，有 transform | 278.9 ms（21,511 样本/s） | 同上 | `ToImage` + `ToDtype(float32)` | 2026-11-05 |
| Q3 数据管线 | DataLoader 整轮 60000 样本 | 2.23 s (bs=32) / 2.26 s (bs=2048) | 同上 | shuffle=True，含 transform，空循环 | 2026-11-05 |
| Q4 hidden 缩放 | 单步 min, h=32 | 0.156 ms | 同上 | bs=128，bench N=20 warmup=5 | 2026-11-05 |
| Q4 hidden 缩放 | 单步 min, h=128 | 0.267 ms | 同上 | 同上 | 2026-11-05 |
| Q4 hidden 缩放 | 单步 min, h=512 | 0.888 ms | 同上 | 同上 | 2026-11-05 |
| Q4 hidden 缩放 | 单步 min, h=1024 | 2.398 ms | 同上 | 同上 | 2026-11-05 |
