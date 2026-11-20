# Week 5 — 量化自己训练的 GPT

> **本周内容基本由 AI 生成。**
>
> 量化与本人 HPC101 Lab2（W8A8 MoE 向量化）、Lab5（Gemma4-12B GPTQ，ΔNLL=0.1018，非对称 + group_size=64）高度重叠，且 Lab5 已在 12B 参数、H800 上完成过更严格的版本。
>
> 所以本周**刻意缩小范围**：不追求压缩率与速度，只在自己从零写的 GPT 上把「量化误差从哪来」逐项拆开。所有数字见 [`../../RESULTS.md`](../../RESULTS.md) 的 Week 5 一节。

## 这周做了什么

在自己的 81 万参数 GPT 上，量化 FFN 与 attention 的 Linear 权重，测 val NLL 变化。

```
weight_stats.py   权重分布：直方图 + 离群值指标 + 量化会用掉多少级
quantlib.py       量化核心：per-tensor / per-channel / per-group 的量化与反量化
eval_nll.py       载入 week4 checkpoint，在固定 batch 上算 val NLL
exp_ffn.py        实验 1：只量化 FFN
exp_attn.py       实验 2：只量化 attention
exp_outlier.py    实验 3：注入离群值，看三种粒度的抗性
weights.png       权重直方图
```

## 跑法

```bash
python weight_stats.py     # 先看分布
python eval_nll.py         # 基线 1.5946
python exp_ffn.py          # 实验 1
python exp_attn.py         # 实验 2
python exp_outlier.py      # 实验 3
```

依赖 week4 的 `ckpt.pt`，路径写死在 `eval_nll.py` 里。

## 三条核心结论

**1. int8 基本无损，int4 才需要考虑精度。**

三种粒度下 int8 的 ΔNLL 都在 ±0.0002。即使注入一个 ×10 离群值，也只到 +0.0008。

**2. 粒度细的主要价值是抗离群值，不是提精度。**

| int4 | 无离群值 | 有离群值 |
|---|---|---|
| per-tensor | +0.0289 | **+0.4076** |
| per-channel | +0.0106 | +0.0114 |
| per-group(64) | +0.0053 | +0.0062 |

无离群值时 per-group 只比 per-tensor 好 5 倍；有离群值时好 66 倍。

**一个离群值就让 per-tensor int4 的损失放大 14 倍**，而 per-group 几乎不动。

**3. 对本周这个模型，per-group(64) int4 足够。**

体积 2048 KB → 272 KB（4 倍压缩），有离群值也只损失 +0.0062 val NLL。

## 两个反直觉的观察

**attention 比 FFN 更容易量化。** int4 per-tensor 下 0.0184 vs 0.0289。这和 `weight_stats.py` 里 `|max|/std` 的差异一致（5.21 vs 6.96）—— 尾部越窄，per-tensor 越不亏。

**通道数少时 per-channel 不输 per-group。** attention 的头权重是 `[32,128]`，只有 32 个通道，per-channel 的粒度损失不明显（0.0084 vs per-group 0.0079）。而 FFN 是 `[512,128]`，per-channel 就明显不如 per-group（0.0106 vs 0.0053）。

**「per-group 一定更好」这个直觉在小张量上不成立。**

## 踩到的三个坑

**1. 注入多个离群值会让参照点失效**

第一版给 8 个 FFN 张量各注入一个 ×10 离群值，结果是 `fp32 本身`就从 1.5946 崩到 3.3267（+1.73）。扰动盖过了量化误差，无法归因。

**改成只注入 1 个张量**，扰动降到 +0.0014，对比才有意义。

**2. `tril` 掩码被误当成权重**

第一次跑 `weight_stats.py`，attention 组的峰度是 7.04（异常）。原因是 `blocks.*.sa.heads.*.tril` 是 float 类型的 buffer（值只有 0/1），被 `group_of` 归进了 attention 组。排除后峰度 3.04，正常。

**3. `inspect.py` 与标准库重名**

`matplotlib` 内部会 `import inspect`，同目录下的 `inspect.py` 抢先被导入，触发循环导入报错。改名 `weight_stats.py`。

## 已知局限

1. **只测精度，不测速度。** `quantize_dequantize` 返回 fp32，计算仍走 fp32。CPU 没有整数矩阵乘指令，真加速必须上 GPU。
2. **体积是算出来的，不是量出来的。** `n_param * bits / 8 + n_scales * 2`，权重实际仍以 fp32 存在内存里。
3. **没有权重补偿。** 只有朴素四舍五入，没做 GPTQ 的 Hessian 误差补偿。
4. **权重量化，不涉及激活。** W4A16，不是 W4A4。激活的离群值比权重严重得多，需要 SmoothQuant 那一类方法。
5. **模型太小，权重太干净。** 峰度 3.0–3.3，接近正态。真实大模型的重尾在这里看不到，离群值只能人为注入。

## 待探索（需 GPU）

| 项 | 现在测不了的原因 |
|---|---|
| 真实 int4/int8 加速比 | CPU 无整数矩阵乘指令，模拟量化不改计算路径 |
| 端到端内存占用 | 权重仍以 fp32 存放，节省只在理论体积上 |
| 大模型上的重尾分布 | 本模型权重接近正态，需要 12B 级模型才有参考价值 |
| GPTQ 式补偿 | 需要 Hessian 近似与逐层校准，收益在 CPU 上无法验证 |
