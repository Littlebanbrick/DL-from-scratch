# RESULTS

> 本文件由 AI 生成，数字全部来自本机实测输出。声明详见 `README.md`。

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

## Week 2 · micrograd 引擎

配置：MLP 2→8→8→1（105 参数），moons 100 样本，hinge loss，`float64`，`cpu_threads 8`，**GC 关闭**（引擎的图有引用环，开 GC 中位数会跳 10 倍）。

### 正确性

| 实验 | 指标 | 数值 | 硬件 | 测量条件 | 日期 |
|---|---|---|---|---|---|
| M6 gradcheck | 测试通过数 | 16/16 | Ultra 9 285H | `python test.py`；8 单算子 + 5 组合 + 3 行为测试 | 2026-11-05 |
| M6 变异测试 | tanh 导数写错 | 2 个测试失败（14/16） | 同上 | `(1-res**2)` 改成 `(1-res)` | 2026-11-05 |
| M6 变异测试 | `grad +=` 改成 `grad =` | 4 个测试失败（12/16） | 同上 | 改 `__add__` 的累加 | 2026-11-05 |

### 性能对比（本引擎 vs torch.autograd）

| 实验 | 指标 | 数值 | 硬件 | 测量条件 | 日期 |
|---|---|---|---|---|---|
| M7 对比 | 纯前向（含建图） | mine 7.924 ms / torch 0.0211 ms（376×） | Ultra 9 285H · cpu_threads 8 | gc.disable()，bench N=20，torch 侧不用 no_grad | 2026-11-05 |
| M7 对比 | 前向 + 反向 | mine 15.077 ms / torch 0.0834 ms（181×） | 同上 | 同上 | 2026-11-05 |
| M7 对比 | 完整一步 | mine 15.028 ms / torch 0.1075 ms（140×） | 同上 | 同上 | 2026-11-05 |
| M7 对比 | 单步，200 步训练内测 | median 15.35 ms / min 15.14 ms | 同上 | `train.py`，gc disabled，计时不含 accuracy 日志 | 2026-11-05 |

### 测量方法副作用（两次踩坑记录）

| 实验 | 现象 | 结论 | 日期 |
|---|---|---|---|
| GC 开启 | 同代码三次 median：11.8 / 98.2 / 120.1 ms | 引擎图有引用环，GC 必须关，否则中位数不可信且慢 30% | 2026-11-05 |
| GC 关闭 | median 8.06 / 7.99 / 7.97 ms | 稳定 | 2026-11-05 |
