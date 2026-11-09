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

## Week 3 · 字符级语言模型（makemore）

配置：`names.txt` 去重后 29494 个名字，按名字 7:2:1 划分（seed=0），`float32`，`cpu_threads 8`。

### bigram：计数 vs 神经网络

| 实验 | 指标 | 数值 | 硬件 | 测量条件 | 日期 |
|---|---|---|---|---|---|
| 计数基线 | NLL（全量 32033 行） | 2.4540 | Ultra 9 285H | 228146 个 pair，MLE | 2026-11-08 |
| 单层网络 | 起点 nll | 3.3019 | 同上 | 600 步 Adam lr=0.1，`Linear(27,27,bias=False)` | 2026-11-08 |
| 单层网络 | 收敛 nll（无正则） | 2.4540 | 同上 | 6000 步，max\|W\| 10.12 | 2026-11-08 |
| 单层网络 | 收敛 nll（L2=0.01） | 2.4604 | 同上 | 6000 步，max\|W\| 4.31 | 2026-11-08 |

### MLP：split 上的实验

| 实验 | 指标 | 数值 | 硬件 | 测量条件 | 日期 |
|---|---|---|---|---|---|
| 计数基线 | dev NLL（+1 平滑） | 2.4581 | Ultra 9 285H | train 建表，dev 评估；无平滑时为 inf（25 个未见 pair） | 2026-11-08 |
| MLP emb=2 h=100 | dev / test | 2.0730 / 2.0791 | 同上 | 3,481 参数，Adam lr=0.01，best@2300，stale 20 | 2026-11-08 |
| MLP emb=2 h=100 | 最佳步 train NLL | 2.0252 | 同上 | 同 step 2300；gap(train−dev) 0.048 | 2026-11-08 |
| MLP emb=10 h=200 | dev / test | 2.0058 / 1.9992 | 同上 | 11,897 参数，Adam lr=0.01，best@300，stale 20 | 2026-11-08 |
| MLP emb=10 h=200 | 最佳步 train NLL | 1.8889 | 同上 | 同 step 300；gap(train−dev) 0.117 | 2026-11-08 |
| MLP emb=10 h=200 | 训练末期 train NLL | ~1.85 | 同上 | step 1800-2200 区间；gap 拉大到 ~0.19 | 2026-11-08 |

### 测量方法副作用（Week 3 记录）

| 实验 | 现象 | 结论 | 日期 |
|---|---|---|---|
| lr=0.1 + Adam 全批量 | 1500/3200/5100 步尖峰，2.18 → 12.34 | 学习率过大，判据是尖峰反复出现 | 2026-11-08 |
| lr=0.01 + 大模型 | step 2100 尖峰 2.85 | 小概率仍会踩空，记录不修 | 2026-11-08 |
| dev best 从 43 次评估中挑 | 小模型 gap +0.0061，大模型 −0.0066 | 选择偏差让 dev 偏低，test 更可信 | 2026-11-08 |
| 早停阈值 1e-4 | stale 被 0.0001 的噪声进步反复重置 | 阈值应设 0.005 以上，或 patience 降到 5 | 2026-11-08 |
